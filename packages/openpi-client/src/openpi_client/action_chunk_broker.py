"""
动作块分发器（Action Chunk Broker）模块

本模块定义了 ActionChunkBroker 类，它是一个"策略装饰器"——在不改变底层策略
行为的前提下，为上层提供"动作块逐帧分发"的能力。

什么是"动作块"（Action Chunk）？
  在机器人学习中，策略模型通常一次预测多步未来的动作，而不是只预测一步。
  这叫做"动作块预测"（Action Chunking / Action Chunk Prediction）：

    传统 one-step 策略：                      预测
      观测 → [模型] → 动作[ₜ] → 执行
                          ↑ 每步都要重新推理，计算量大

    动作块策略（Chunking）：                  预测一个块
      观测 → [模型] → 动作[ₜ, ₜ₊₁, ..., ₜ₊ₕ]
                       ↓ 逐步分发
      stepₜ: 执行 动作[ₜ]      ← 从缓存中取
      stepₜ₊₁: 执行 动作[ₜ₊₁]   ← 从缓存中取
      ...
      stepₜ₊ₕ: 缓存用完 → 重新推理

  这样做的好处：
    1. 减少推理频率（每 H 步推理一次 → 计算量降低到 1/H）
    2. 平滑控制（模型可以看到更长远的规划，避免短视）
    3. 容忍延迟（即使推理偶尔延迟，缓存中还有动作可以用）

类的工作方式：
  它包装一个底层策略（BasePolicy），截获其 infer() 调用，实现"缓存-分发"逻辑。
  对于上层（如 Runtime 和 PolicyAgent），它看上去就是一个普通的策略。

数据流：
  PolicyAgent（每步调用）
      │
      ▼
  ActionChunkBroker.infer(obs)
      │
      ├─ 缓存中有剩余动作？→ 返回当前步的动作 → 步数 +1
      │
      └─ 缓存已用完？ → 调用底层策略.infer(obs) → 缓存整个动作块
                        → 返回第一步动作 → 步数 +1
      │
      ▼
  底层策略（如 WebSocket 远程策略服务）

示例（action_horizon=3）：
  时间轴    上层调用         ActionChunkBroker         底层策略
  ──────────────────────────────────────────────────────────────
  step₀     infer(obs₀)  →  缓存为空 → 调底层     →  infer(obs₀)
                              返回 actions[0] ← 输出 [a₀, a₁, a₂]
                              缓存 = [a₀, a₁, a₂], cur_step=1

  step₁     infer(obs₁)  →  缓存有剩余 → 直接用
                              返回 actions[1] ← 从缓存取 a₁
                              缓存 = [a₀, a₁, a₂], cur_step=2

  step₂     infer(obs₂)  →  缓存有剩余 → 直接用
                              返回 actions[2] ← 从缓存取 a₂
                              缓存 = [a₀, a₁, a₂], cur_step=3

  step₃     infer(obs₃)  →  cur_step ≥ horizon(3) → 缓存清空
                              缓存为空 → 调底层     →  infer(obs₃)
                              返回 actions[0] ← 输出 [a₃, a₄, a₅]
                              缓存 = [a₃, a₄, a₅], cur_step=1

注意事项：
  - 底层策略的 infer() 仅在缓存用尽时才被调用。
    因此观测 obs 可能已经"过时"了（缓存期间环境已经变了 H-1 步）。
    这意味动作块策略天然有一定"延迟"，但在实践中通常不是问题，
    因为动作块足够短（通常 H=10~50 步，每步 20ms = 200ms~1s 的预测范围）。
  - reset() 必须被调用以清空缓存（例如新的 episode 开始时）。
  - 底层策略的输出必须满足：第一个维度是块大小（action_horizon）。
    即形状为 [action_horizon, action_dim]。
"""

from typing import Dict

import numpy as np  # 数值计算库，用于数组操作
import tree  # 树结构工具库（tree.map_structure 可遍历字典/列表/元组的叶子节点）
from typing_extensions import override  # 类型提示：显式标记方法覆盖

from openpi_client import base_policy as _base_policy  # 基础策略接口


class ActionChunkBroker(_base_policy.BasePolicy):
    """动作块分发器：将策略输出的"动作块"逐个分发给调用者。

    这个类实现了"缓存-分发"模式（Cache-and-Dispatch Pattern）。
    它包装一个底层策略，拦截 infer() 调用，实现以下逻辑：

      1. 第一次调用 → 触发底层推理 → 缓存整个动作块
      2. 后续 H-1 次调用 → 从缓存逐步取出动作，不触发推理
      3. 缓存用完 → 再次触发底层推理 → 重复

    这种模式其实是一种"滚动时域控制"（Receding Horizon Control）的简化实现：
    - 每次预测未来 H 步的完整轨迹
    - 只执行第一步
    - 然后重新规划（当缓存用完时）

    但这里有一个简化：不是每步都重新规划，而是每 H 步才重新规划。
    这减少了推理频率，但也意味着在两次推理之间策略对环境变化是"盲"的。

    适用场景：
      - 高频控制（如 50Hz）与昂贵推理（如大模型）之间的矛盾 — 用缓存来缓冲
      - 动作平滑 — 多步预测天然具有连续性和平滑性
      - 网络延迟容忍 — WebSocket 策略服务器的通信延迟可以被缓存吸收

    不适用场景：
      - 需要对观测变化立即响应的任务（如快速避障）
      - 动作必须精确跟随每步观测变化的精密操作
    """

    def __init__(self, policy: _base_policy.BasePolicy, action_horizon: int):
        """初始化动作块分发器。

        Args:
            policy:         底层策略（BasePolicy 的实现）。
                            可以是 WebsocketClientPolicy（远程推理）
                            或其他任何实现了 BasePolicy 接口的策略。
            action_horizon: 动作块大小（horizon）。
                            - 底层策略每次 infer() 应该输出 action_horizon 步
                            - 分发器每 action_horizon 次调用才会触发一次底层推理
                            - 例如 action_horizon=10: 推理 1 次管 10 步
        """
        self._policy = policy  # 底层策略（真正的推理引擎）
        self._action_horizon = action_horizon  # 动作块大小
        self._cur_step: int = 0  # 当前在块中的位置（0 到 action_horizon-1）

        # 上次推理结果的缓存。形状：{key: [action_horizon, ...]}
        # None 表示缓存为空，需要重新推理。
        self._last_results: Dict[str, np.ndarray] | None = None

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        """执行策略推理（含缓存逻辑）。

        这个方法对外表现为"每步推理"（和普通策略的 infer 一样），
        但内部通过缓存实现了"每 H 步才真正推理一次"。

        Args:
            obs: 观测字典，包含摄像头图像、关节状态等。

        Returns:
            动作字典，包含第一步动作。形状示例：
            {
                "actions": np.ndarray [action_dim],  # 当前步的动作
                "state": np.ndarray [...],           # 对应的状态
                ...
            }
            注意：返回的 actions 是"单步"而不是"整个块"。
        """
        # ====================================================================
        # 检查缓存
        #
        # 如果 _last_results 为 None，说明：
        #   1. 刚刚初始化，还没有推理过
        #   2. 上一轮缓存已经用完
        #   3. reset() 被调用过
        # 这时需要真正调用底层策略进行推理。
        #
        # 如果 _last_results 不为 None，说明缓存中还有剩余动作，
        # 直接跳过底层推理，从缓存中取。
        # ====================================================================
        if self._last_results is None:
            # ── 触发底层推理 ──
            # 底层策略（如 WebsocketClientPolicy）可能发起网络请求，
            # 到远程推理服务器执行模型前向传播。
            # 返回结果应该包含 action_horizon 步的未来动作。
            self._last_results = self._policy.infer(obs)

            # 重置步数计数器（从块的第一步开始）
            self._cur_step = 0

        # ====================================================================
        # 从缓存中提取当前步的动作
        #
        # 缓存结果中的每个 NumPy 数组，第一个维度都是 action_horizon。
        # 我们需要取第 _cur_step 个元素（即当前步对应的那一行）。
        #
        # tree.map_structure(slicer, self._last_results) 的作用：
        #   遍历 _last_results 字典的每个叶子节点（每个 NumPy 数组），
        #   对每个数组应用 slicer 函数——取第 _cur_step 个元素。
        #
        # 示例 (_cur_step=0):
        #   输入: {"actions": np.array([[1,2], [3,4], [5,6]]),
        #          "state":   np.array([[1],   [2],   [3]]  )}
        #   输出: {"actions": np.array([1,2]),
        #          "state":   np.array([1])}
        # ====================================================================
        def slicer(x):
            """从数组的第一个维度取第 _cur_step 个元素。"""
            if isinstance(x, np.ndarray):
                return x[self._cur_step, ...]  # 索引当前步
            else:
                return x  # 非数组类型（如字符串）原样返回

        # 对缓存结果逐叶子应用 slicer，得到当前步的动作
        results = tree.map_structure(slicer, self._last_results)

        # 步数计数器 +1，指向块中的下一步
        self._cur_step += 1

        # ====================================================================
        # 检查缓存是否用完
        #
        # 如果 _cur_step >= action_horizon，说明已经取了 H 步，
        # 缓存已经耗尽。将 _last_results 设为 None，
        # 下次调用 infer() 时会触发新一轮推理。
        # ====================================================================
        if self._cur_step >= self._action_horizon:
            # 缓存耗尽，清空标志位
            # 下一次 infer() 将触发新的底层推理
            self._last_results = None

        return results

    @override
    def reset(self) -> None:
        """重置分发器的内部状态。

        这个方法在以下时机被调用：
          - 新 episode 开始时（由 Runtime 的 _run_episode() 通过
            agent.reset() → PolicyAgent.reset() 间接调用）
          - 需要清除缓存，重新开始

        重置的内容：
          1. 清空缓存（_last_results = None）
          2. 重置步数计数器（_cur_step = 0）
          3. 同时调用底层策略的 reset()（清理其内部状态）

        为什么需要 reset()？
          如果不重置，缓存中会残留上一个 episode 的动作。
          新 episode 开始时的观测完全不同，使用旧缓存的动作
          会导致机器人行为异常甚至危险。
        """
        self._policy.reset()  # 让底层策略也重置（可能清空其内部缓存）
        self._last_results = None  # 清空动作块缓存
        self._cur_step = 0  # 重置步数计数器

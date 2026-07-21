"""
策略智能体（PolicyAgent）模块 —— Agent 接口的标准实现

本模块定义了 PolicyAgent 类，它是 Runtime 框架中 Agent 接口的"标准/默认实现"。

PolicyAgent 的职责很简单：作为一个"适配器"（Adapter），将 BasePolicy 接口
适配到 Agent 接口。它不添加任何额外的逻辑——get_action 就是调用 policy.infer()，
reset 就是调用 policy.reset()。

为什么需要这一层适配？
  Runtime 主循环只认识 Agent 接口（get_action / reset），不认识 BasePolicy。
  但实际做推理的是 BasePolicy。PolicyAgent 就是连接这两者的桥梁：

    Runtime 主循环                    Agent 接口                  实现
    ┌──────────────┐    调用     ┌──────────────────┐    ┌──────────────────────┐
    │  _step()     │ ──────────→ │  Agent.get_action │ ──→│  PolicyAgent         │
    │              │             │  (observation)    │    │    ↓                 │
    │  1. get_obs  │             └──────────────────┘    │  policy.infer(obs)   │
    │  2. get_act  │                                      │    ↓                 │
    │  3. apply    │                                      │  BasePolicy 实现      │
    │  4. notify   │                                      │  (本地/远程/缓存...)  │
    └──────────────┘                                      └──────────────────────┘

架构关系：
  这实际上是三个层次的抽象：
    1. Agent 接口        (agent.py)          —— "谁做决策"的抽象
    2. BasePolicy 接口   (base_policy.py)    —— "如何推理"的抽象
    3. Policy 具体实现    (policy.py / ...)   —— "在哪里推理"的实现
                                                     本地模型 / 远程服务 / 缓存装饰器

  PolicyAgent 位于第 1 层和第 2 层之间，是一个"转接器"。

设计决策：
  PolicyAgent 为什么这么简单？它存在的意义是什么？
    1. 职责分离：Runtime 只依赖 Agent 接口，不依赖 BasePolicy 接口。
       未来如果 Agent 接口变化（如增加 pre_action 钩子），只需要修改
       PolicyAgent，不需要修改 Runtime。
    2. 扩展点：如果你想在"观测 → 动作"之间加入预处理或后处理
       （如输入校验、动作限幅、日志记录），可以继承 PolicyAgent
       重写 get_action() 方法，而不需要修改 Runtime。
    3. 统一 reset 链：Runtime → Agent.reset() → PolicyAgent.reset()
       → Policy.reset() → ActionChunkBroker.reset() → 清空缓存
       这条链确保了 episode 切换时所有层级的缓存都被清理。
"""

from typing_extensions import override  # 类型提示：显式标记方法覆盖

from openpi_client import base_policy as _base_policy  # 基础策略接口（推理引擎）
from openpi_client.runtime import agent as _agent  # 智能体接口（决策者）


class PolicyAgent(_agent.Agent):
    """策略智能体 —— 将 BasePolicy 包装为 Agent 的适配器。

    这个类的实现极其简单——它就是"把 Agent 的方法调用转发给 Policy"。

    但它的意义不止于此：
      - 它是 Runtime 框架中 Agent 接口的"规范实现"
      - 它定义了"Agent 应该如何持有和使用 Policy"的标准模式
      - 它是整条 reset 调用链中的关键一环

    如果你需要自定义 Agent 行为（如在推理前后添加逻辑），
    可以继承 PolicyAgent 并重写对应方法：

        class MyAgent(PolicyAgent):
            @override
            def get_action(self, observation):
                # 预处理：检查观测完整性
                if "images" not in observation:
                    observation["images"] = self._fallback_image
                # 调用父类方法（即 policy.infer）
                action = super().get_action(observation)
                # 后处理：限制关节速度
                action["actions"] = np.clip(action["actions"], -max_speed, max_speed)
                return action

    与 ActionChunkBroker 的关系：
      注意这里的 PolicyAgent 内部持有的是 BasePolicy，而 BasePolicy 的
      实现可能是 ActionChunkBroker（装饰了远程策略）。所以调用链是：

      Runtime
        → PolicyAgent.get_action(obs)          ← 智能体层
          → ActionChunkBroker.infer(obs)        ← 缓存层
            → WebsocketClientPolicy.infer(obs)  ← 网络层
              → [远程服务器推理]                  ← 模型层

      每一层只关心自己的职责，通过接口组合起来。
    """

    def __init__(self, policy: _base_policy.BasePolicy) -> None:
        """初始化策略智能体。

        Args:
            policy: 底层策略对象（BasePolicy 的任何实现）。
                    可以是：
                    - ActionChunkBroker（包装了远程策略，带缓存分发）
                    - WebsocketClientPolicy（直接远程推理）
                    - Policy（本地模型推理，src/openpi/policies/policy.py）
                    - PolicyRecorder（带记录功能的策略装饰器）
                    等等——只要是 BasePolicy 的实现都可以。
        """
        self._policy = policy

    @override
    def get_action(self, observation: dict) -> dict:
        """根据观测做出决策 —— 直接委托给底层策略。

        这个方法非常简单——它就是调用 self._policy.infer(observation)。
        所有的复杂逻辑（模型推理、缓存管理、网络通信）都在 policy 内部。

        为什么没有预处理或后处理？
          因为 PolicyAgent 被设计为"纯适配器"——它不做任何策略相关的事情。
          如果需要额外逻辑，推荐的方式是：
          - 预处理（观测变换）：在 Environment 中完成（如 env.py 的 _convert_observation）
          - 后处理（动作变换）：在 Policy 的输出变换中完成（如反归一化）
          - 装饰器逻辑：使用 PolicyRecorder 或自定义 BasePolicy 包装器

        Args:
            observation: 环境观测字典（由 Runtime 从 Environment 获取并传递）。

        Returns:
            dict: 动作字典（由底层策略的 infer 方法返回）。
                  至少包含 "actions" 键，其值为 NumPy 数组。
        """
        # 委托给底层策略的 infer 方法
        # 这里的 self._policy 可能是任何 BasePolicy 实现：
        #   - 如果是 ActionChunkBroker，它会检查缓存，可能触发或不触发远程推理
        #   - 如果是 WebsocketClientPolicy，它会通过网络请求远程服务器
        #   - 如果是本地 Policy，它会直接在本进程的模型上运行推理
        return self._policy.infer(observation)

    @override
    def reset(self) -> None:
        """重置智能体状态 —— 即重置底层策略的状态。

        这个方法在以下场景被调用：
          - 每个 episode 开始时（由 Runtime._run_episode() 调用）
          - 在 environment.reset() 之后

        它做的事情就是调用 self._policy.reset()，然后这个调用会
        沿着 BasePolicy 的包装链逐层传递下去：

        PolicyAgent.reset()
          → ActionChunkBroker.reset()    ← 清空动作块缓存
            → WebsocketClientPolicy.reset()  ← 无操作（远程连接不需要重置）
              或者
            → Policy.reset()             ← JAX 模型可能有内部状态需要重置

        为什么要级联 reset？
          考虑一个没有 reset 的场景：
            Episode 1: 策略推理了动作块 [抓取, 抬升, 移动, 放置]
                       执行了前两步：抓取、抬升 → 机器人把 cube 抓起来了
                       → Episode 结束，机器人复位

            Episode 2: 没有 reset → ActionChunkBroker 缓存中还有
                       [移动, 放置] 两个动作 → 直接执行 → 机器人做出
                       诡异的动作（因为环境和上一轮结束时的状态完全不同）

          这就是为什么 reset() 必须逐层传递，确保所有缓存都被清空。
        """
        self._policy.reset()

"""
环境接口抽象定义模块

本模块定义了 Environment 抽象基类（Abstract Base Class），它是 openpi
运行时框架（Runtime）中"环境"组件的标准接口。

什么是"环境"（Environment）？
  在机器人学习的上下文中，"环境"代表机器人和它所在的物理或仿真世界。
  它是 Runtime 主循环的数据源和动作执行端：

    Runtime 主循环:
        ┌──────────────────────────────────┐
        │  1. get_observation() ← 读取状态 │
        │  2. agent.get_action(obs) ← 推理 │
        │  3. apply_action(action) ← 执行   │
        │  4. is_episode_complete() ← 检查  │
        └──────────────────────────────────┘

为什么使用抽象类？
  通过定义标准接口，我们可以：
    - 在"仿真环境"和"真实机器人"之间无缝切换（都实现同一个接口）  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    - 在 Runtime 中统一处理所有类型的环境，无需关心具体实现
    - 方便测试：用仿真环境调试代码，再切换到真实环境部署

本模块的实现者指南：
  要创建一个新的环境，只需继承 Environment 并实现四个抽象方法：
    reset()                — 重置环境到初始状态
    get_observation()      — 返回当前观测
    apply_action(action)   — 执行一个动作
    is_episode_complete()  — 判断当前 episode 是否结束

参考实现：
  - AlohaSimEnvironment（examples/aloha_sim/env.py）— ALOHA MuJoCo 仿真环境
"""

import abc  # Python 抽象基类库，用于定义接口契约


class Environment(abc.ABC):
    """环境（Environment）抽象基类。

    一个 Environment 对象代表"机器人和它所在的物理世界"。
    它的核心契约很简单：
      - 可以被查询当前状态（get_observation）
      - 可以接受动作来改变状态（apply_action）

    生命周期（由 Runtime 主循环管理）：
      ┌────────────────────────────────────────────────────┐
      │  runtime.run()                                     │
      │    for each episode:                               │
      │      environment.reset()          ← 初始化环境     │
      │        repeat:                                     │
      │          obs = environment.get_observation() ← 读取│
      │          act = agent.get_action(obs)        ← 推理 │
      │          environment.apply_action(act)      ← 执行 │
      │        until environment.is_episode_complete()     │
      │      environment.reset()          ← 清理复位      │
      └────────────────────────────────────────────────────┘

    关键设计理念：
      1. 简单性：只定义最基本的四个方法，不假设任何具体机器人或传感器的存在
      2. 通用性：同样适用于仿真环境和真实机器人
      3. 无状态契约：Environment 自己管理内部状态，Runtime 不保存环境的历史数据
         （但 get_observation() 可以在内部缓存最新观测，避免重复计算）
    """

    @abc.abstractmethod
    def reset(self) -> None:
        """重置环境到初始状态。

        这个方法在以下时机被调用：
          1. 每个 episode 开始时（runtime._run_episode() 的第一行）
          2. 所有 episode 运行完毕后（runtime.run() 最后一行，用于安全复位）

        实现者应该在这个方法中：
          - 将机器人/仿真恢复到起始位姿
          - 重置任务目标（如随机初始化物体的位置）
          - 清空 episode 相关的内部状态（累计奖励、步数计数等）

        注意：
          - reset() 不应该返回观测值。观测值应该通过后续的 get_observation()
            调用来获取（因为 reset 后 Runtime 内部还有一些初始化工作要做）。
          - 在真实机器人上，reset() 可能涉及将机器人移动到"安全起始位姿"
            （例如抬起到一个不会碰撞的位置）。
          - reset() 可能会被多次调用，实现应该能够安全地重复调用。

        设计决策：
          为什么 reset() 不直接返回观测？
            因为 Runtime 的设计中，reset() 和 get_observation() 是分离的。
            reset() 只负责"重置状态"，而观测的获取和缓存由具体实现自由决定。
            这使得 reset() 可以专注于它自己的职责。
        """

    @abc.abstractmethod
    def is_episode_complete(self) -> bool:
        """检查当前 episode 是否已经完成。

        Runtime 在每一步 apply_action() 之后调用此方法，
        以决定是继续当前 episode 还是进入下一轮。

        返回 True 的情况可能包括：
          - 任务成功完成（如成功抓取并放置物体）
          - 任务失败（如机器人翻倒、物体掉落）
          - 达到步数上限（超时截断）
          - 用户手动终止

        返回 False 表示 episode 仍在进行中，Runtime 将继续循环。

        实现注意事项：
          - 这个方法应该轻量、快速（每次 step 都会被调用）
          - 应该幂等（连续调用多次应返回一致的结果，直到 apply_action 被再次调用）
          - 通常内部维护一个 bool 标志（如 self._done），在 apply_action 中更新

        Returns:
            bool: True 表示 episode 已完成，False 表示仍在进行中。
        """

    @abc.abstractmethod
    def get_observation(self) -> dict:
        """获取当前环境的观测。

        这是 Runtime 获取"环境状态"的唯一途径。
        返回的观测字典会被直接传递给 agent.get_action() 进行推理。

        返回的字典应该包含策略模型（Policy）所需的所有输入数据：
          - images:    摄像头图像（如 cam_high, cam_left_wrist 等）
          - state:     机器人关节角度、夹爪位置等状态信息
          - 以及其他模型需要的字段

        注意：
          - 这个方法在每一步都会被调用，而且通常紧跟在 apply_action() 之后。
          - 返回值应该是一个"快照"——代表调用时刻的环境状态。
          - 实现通常会缓存最近一次 apply_action() 后的观测，避免重复计算。
          - 如果 reset() 后还没有 apply_action()，应返回重置后的初始观测。

        性能考虑：
          - 对于仿真环境：可能需要渲染图像、读取物理状态——相对昂贵
          - 对于真实机器人：可能需要从传感器读取数据——可能涉及 I/O
          - 建议在 apply_action 中预先计算并缓存观测，get_observation 只做返回

        Returns:
            dict: 观测数据字典。具体键值对由具体环境和策略模型约定。
                  典型结构：
                  {
                      "state": np.ndarray,        # 关节角度等状态
                      "images": {
                          "cam_high": np.ndarray,  # 摄像头图像
                      },
                  }

        Raises:
            RuntimeError: 如果在 reset() 之前调用此方法（_last_obs 为 None）。
        """

    @abc.abstractmethod
    def apply_action(self, action: dict) -> None:
        """在环境中执行一个动作。

        这是 Runtime 向环境"下达指令"的唯一途径。
        动作字典通常包含机器人各关节的目标位置或速度。

        Args:
            action: 动作字典。包含策略模型预测的控制指令。
                    典型结构：
                    {
                        "actions": np.ndarray,   # 关节角度/速度指令
                        "states": np.ndarray,    # （可选）对应的期望状态
                    }

        这个方法应该：
          1. 解析动作指令（从 action 字典中提取控制量）
          2. 执行动作（在仿真中更新物理状态，或向真实机器人发送指令）
          3. 更新内部状态（步数计数、累计奖励等）
          4. 检查 episode 是否结束（更新 done 标志，供 is_episode_complete 返回）
          5. 预处理下一帧的观测（缓存以便 get_observation 快速返回）

        与 Gymnasium 的差异：
          Gymnasium 的 step() 同时返回 (obs, reward, terminated, truncated, info)，
          而这里 apply_action 只负责"执行"不负责"返回"。
          观测、结束标志等通过单独的接口方法获取。
          这种分离的设计让环境的具体实现更灵活。
        """

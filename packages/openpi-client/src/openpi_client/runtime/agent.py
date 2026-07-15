"""
智能体（Agent）接口抽象定义模块

本模块定义了 Agent 抽象基类——Runtime 框架中"决策者"组件的标准接口。

什么是"智能体"（Agent）？
  在 Runtime 框架中，智能体是"做决策的实体"——它接收来自环境的观测，
  输出要执行的动作。简单来说，Agent = 策略的代言人。

  Runtime 主循环中的位置：
    ┌────────────────────────────────────┐
    │  environment.get_observation()     │ ← 环境：提供观测
    │  agent.get_action(observation)     │ ← Agent：做出决策 ←——— 你现在在这里
    │  environment.apply_action(action)  │ ← 环境：执行动作
    └────────────────────────────────────┘

Agent vs BasePolicy 的区别：
  初学者容易混淆这两个概念，它们的职责不同：

  ┌────────────────────────────────────────────────────────────┐
  │  Runtime 的视角:                                            │
  │                                                            │
  │  Environment    Agent              Policy                   │
  │  (感知-执行)     (决策者)           (推理引擎)               │
  │                                                            │
  │  传感器 ──→ observation ──→ get_action()                    │
  │                               │                             │
  │                               ▼                             │
  │                          Policy.infer(obs)                  │
  │                               │                             │
  │                               ▼                             │
  │                          动作指令 ──→ apply_action()         │
  └────────────────────────────────────────────────────────────┘

  - Agent：更高层的抽象，代表"谁在做决策"。它内部可能包含一个或多个 Policy，
          也可能包含额外的逻辑（如动作选择、安全过滤、模式切换等）。
  - BasePolicy：更底层的抽象，代表"推理引擎"。它只负责"观测 → 动作"的映射。

  打个比方：
    Agent  = 飞行员（做决策的人）
    Policy = 飞机的操纵杆和仪表（执行决策的工具）

  在 openpi 中，Agent 的唯一标准实现是 PolicyAgent（policy_agent.py）。
  它做的事情很简单：把 get_action(obs) 直接委托给内部的 Policy.infer(obs)。

  为什么还要多这一层？
    1. 抽象灵活性：未来可以实现更复杂的 Agent，如：
       - SwitchingAgent：根据场景在多个 Policy 之间切换
       - SafetyAgent：在 Policy 输出的动作上做安全约束检查
       - EnsembleAgent：同时运行多个 Policy，投票或平均输出
    2. 关注点分离：Runtime 只和 Agent 打交道，不直接接触 Policy。
       这使得 Runtime 的代码对"策略的实现方式"完全无感知。
    3. 测试友好：可以轻松创建 MockAgent（固定输出）用于测试 Runtime。

接口设计：
  Agent 定义了三个抽象方法：
    - get_action(observation) → action：核心决策方法
    - reset()：重置内部状态
"""

import abc  # Python 抽象基类库，用于定义接口契约


class Agent(abc.ABC):
    """智能体抽象基类 —— Runtime 框架中的"决策者"。

    智能体是 Runtime 主循环中的"大脑"——它接收观测，返回动作。
    整个 Runtime 只有一个 Agent，它在每一步被调用一次。

    生命周期（由 Runtime 管理）：
      reset()         ← episode 开始时调用
      get_action()    ← 每步调用一次（主循环的核心）
      get_action()    ← ...
      ...             ← 直到 episode 结束
      reset()         ← 下一个 episode 开始时再次调用

    Agent 内部不一定只有一个 Policy。它可以：
      - 包装一个 Policy（最简单的情况，如 PolicyAgent）
      - 组合多个 Policy（如主策略 + 备用策略）
      - 在 Policy 输出上添加后处理（如平滑、安全约束）
      - 完全不用 Policy（如硬编码规则、键盘控制）

    但不管内部多复杂，对 Runtime 来说，Agent 就是一个
    "接收观测 → 返回动作"的黑盒。

    实现注意事项：
      - get_action() 应该是纯计算函数（无副作用），但这不是强制要求
      - reset() 应该清理所有与 episode 相关的内部状态
      - 如果 Agent 内部持有 Policy，应该在其 reset() 中调用 policy.reset()
      - get_action() 的输入输出格式需要和 Environment/Policy 协商一致
    """

    @abc.abstractmethod
    def get_action(self, observation: dict) -> dict:
        """根据观测做出决策，返回要执行的动作。

        这是 Agent 唯一的核心方法——"思考"的入口。

        调用时机：Runtime 主循环的每一步，在 get_observation() 之后、
        apply_action() 之前调用。

        典型实现（PolicyAgent）：
            1. 接收观测（从 Environment 来的 dict）
            2. 将观测传给内部 Policy.infer(obs)
            3. 返回 Policy 输出的动作 dict

        输入输出格式：
          observation 的格式由 Environment 决定——可以是仿真环境的
          标准化观测，也可以是真实机器人的传感器数据。

          action 的格式需要和 Environment.apply_action() 的期望一致。

        Args:
            observation: 环境观测字典。包含机器人当前状态的所有信息，
                        如摄像头图像、关节角度、力传感器等。
                        具体键值对由 Environment 的实现决定。

        Returns:
            dict: 动作字典。至少包含 "actions" 键，
                  其值为一个 NumPy 数组，表示要执行的控制指令。
                  还可能包含其他元数据（如推理耗时等）。

        Raises:
            NotImplementedError: 子类未实现此抽象方法（由 abc 触发）。
        """

    @abc.abstractmethod
    def reset(self) -> None:
        """重置智能体到初始状态。

        调用时机：
          - 每个 episode 开始时（由 Runtime._run_episode() 调用）
          - 在 environment.reset() 之后被调用

        重置内容取决于 Agent 的内部实现：
          - PolicyAgent：调用内部 Policy.reset()（如清空 ActionChunkBroker 缓存）
          - 有状态的 Agent：清空所有内部状态变量
          - 无状态的 Agent（如纯规则 Agent）：可以是空实现

        为什么需要 reset()？
          如果不重置，Agent 可能保留上一个 episode 的"记忆"，
          导致新的 episode 开始时行为异常。对于 ActionChunkBroker 来说，
          缓存中残留的上一个 episode 的动作会被用于新的 episode，
          可能导致灾难性的后果。

        实现注意事项：
          - 这个方法应该幂等——连续调用多次应该和调用一次效果相同
          - 必须清空所有与特定 episode 相关的状态
          - 如果需要，同时级联重置内部的子组件（如 Policy）
        """

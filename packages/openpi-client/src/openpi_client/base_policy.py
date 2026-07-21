"""
基础策略接口定义模块

本模块定义了 BasePolicy 抽象基类——它是 openpi 中"策略"（Policy）的统一接口，
是整个系统中复用度最高的抽象层。

什么是"策略"（Policy）？
  在机器人学习中，"策略"是一个函数（或模型），它接收观测（observation），
  输出动作（action）。简单说就是"看到什么 → 做什么"的映射。

  策略 = π(a | o)
        策略  动作  观测

BasePolicy 的定位：
  它是整个策略体系的"最小公分母"——所有与策略交互的代码都通过这个接口进行。
  具体来说，它有以下几层实现：

    ┌──────────────────────────────────────────────────────┐
    │                    上层调用者                          │
    │  Runtime / PolicyAgent / 用户脚本                     │
    ├──────────────────────────────────────────────────────┤
    │                  BasePolicy 接口                       │
    │                   infer() + reset()                   │
    ├──────────────────────────────────────────────────────┤
    │            实现层（从简单到复杂）                       │
    │                                                       │
    │  ActionChunkBroker    ←── 装饰器：缓存+分发            │
    │      │                                                 │
    │      ▼                                                 │
    │  WebsocketClientPolicy ── 远程推理（通过 WebSocket）   │
    │                                                       │
    │  Policy (src/openpi/policies/policy.py)  ←── 本地推理  │
    │      │                            （直接加载模型）    │
    │      ▼                                                 │
    │  PolicyRecorder        ←── 装饰器：记录输入输出        │
    └──────────────────────────────────────────────────────┘

  无论是远程推理还是本地模型、无论是带缓存还是带记录，所有策略都
  实现相同的 infer(obs) → action 接口——这是"策略模式"（Strategy Pattern）
  的经典应用。

为什么用抽象类而不是 Protocol？
  这里用 abc.ABC 而不是 typing.Protocol 是故意的：
  - reset() 提供了默认实现（空操作），实现者可以按需覆盖
  - infer() 是抽象方法，如果不实现就会在实例化时报错
  - 这种设计让"最小实现"只需要写 infer() 一个方法

参考实现：
  - ActionChunkBroker（action_chunk_broker.py）— 动作块缓存分发
  - WebsocketClientPolicy（websocket_client_policy.py）— 远程策略调用
  - Policy（src/openpi/policies/policy.py）— 本地模型推理
  - PolicyRecorder（src/openpi/policies/policy.py）— 推理记录
"""

import abc  # Python 抽象基类库，用于定义接口契约
from typing import Dict  # 类型提示：字典类型


class BasePolicy(abc.ABC):
    """策略基类 —— 所有策略实现的统一接口。

    这是 openpi 中最核心的抽象定义。它规定了一个"策略"必须能做两件事：
      1. infer(obs) → action：根据观测输出动作（必须实现）
      2. reset()：重置策略的内部状态（可选覆盖）

    无论底层是：
      - 一个在 GPU 上运行的 JAX / PyTorch 模型（本地推理）
      - 一个通过 WebSocket 连接的远程推理服务（远程推理）
      - 一个缓存分发的装饰器（ActionChunkBroker）
      - 一个记录输入输出的调试包装器（PolicyRecorder）

    它们都表现为 BasePolicy——调用者无需关心具体实现。

    使用方式（多态）：
        def run_episode(policy: BasePolicy, env):
            policy.reset()
            obs = env.reset()
            while not env.done:
                action = policy.infer(obs)   # ← 不管是什么策略，都这样调用
                obs, done = env.step(action)
    """

    @abc.abstractmethod
    def infer(self, obs: Dict) -> Dict:
        """根据观测输出动作 —— 策略的核心方法。

        这是策略的"前向传播"——接收传感器数据，输出控制指令。

        输入 obs 的典型结构：
            {
                "state": np.ndarray,          # 关节角度、夹爪位置等 [state_dim]
                "images": {
                    "cam_high": np.ndarray,    # 顶部摄像头 [C, H, W]
                    "cam_left_wrist": ...,     # （可选）左腕摄像头
                    "cam_right_wrist": ...,    # （可选）右腕摄像头
                },
                "prompt": str,                 # （可选）语言指令如 "Transfer cube"
            }

        输出 action 的典型结构：
            {
                "actions": np.ndarray,         # 关节目标位置/速度 [action_dim]
                "state": np.ndarray,           # （透传）对应的状态
                "policy_timing": {             # （可选）推理耗时统计
                    "infer_ms": 12.5,
                },
            }

        Args:
            obs: 观测字典。具体键值对由环境和策略模型协商决定，
                 但至少应包含 images 和 state。

        Returns:
            动作字典。至少应包含 "actions" 键，值是一个一维或多维数组。

        Raises:
            NotImplementedError: 子类未实现此方法（由 abc 自动触发）。
        """

    def reset(self) -> None:
        """重置策略到初始状态。

        这个方法在以下场景被调用：
          - 新 episode 开始时：清空策略的内部状态，防止上一轮的数据
            影响新的 episode（例如 ActionChunkBroker 需要清空动作缓存）
          - PolicyAgent.reset() 被调用时：代理将重置传递给底层策略

        默认实现是什么都不做（pass）。
        子类应该在此方法中重置所有与 episode 相关的内部状态：
          - ActionChunkBroker：清空动作块缓存，重置步数计数器
          - Policy（src/openpi/policies/policy.py）：重置 JAX RNG key（可选）
          - WebsocketClientPolicy：不需要重置，保持连接

        为什么不是抽象方法？
          因为不是所有策略都有需要重置的"内部状态"。
          例如 WebsocketClientPolicy 只是通过网络收发数据，
          服务器端不维护与 episode 相关的状态，reset() 就是空操作。
          让 reset() 有默认实现可以避免子类被迫编写空方法。
        """
        pass

"""
ALOHA 真机环境适配器

本模块定义了 AlohaRealEnvironment —— 将真实 ALOHA 机器人（Interbotix VX300S 双臂）
接入 openpi 运行时框架的"环境适配器"。

它在架构中的位置：
  ┌─────────────────────────────────────────────────────────────────┐
  │  Runtime（主循环调度器）                                          │
  │    │                                                             │
  │    ├─ reset()              ──→  环境复位                         │
  │    ├─ get_observation()    ──→  读取机器人状态（循环步）           │
  │    ├─ apply_action(action) ──→  执行动作（循环步）                │
  │    └─ is_episode_complete()─→  检查结束条件（循环步）              │
  │                                                                 │
  │  ┌──────────────────────────────────────────────────────────┐   │
  │  │  AlohaRealEnvironment（本模块）                            │   │
  │  │    │  实现对 Environment 抽象接口的所有方法                  │   │
  │  │    │  职责：适配 Runtime 接口 + 图像预处理                   │   │
  │  │    │                                                        │   │
  │  │    ▼                                                        │   │
  │  │  RealEnv（real_env.py — 底层硬件控制）                       │   │
  │  │    │  通过 Interbotix SDK 直接控制真实的 VX300S 机械臂       │   │
  │  │    │  reset() → 关节归位                                     │   │
  │  │    │  step(action) → 发送电机指令 + 读取传感器               │   │
  │  └──────────────────────────────────────────────────────────┘   │
  └─────────────────────────────────────────────────────────────────┘

两层设计的原因（AlohaRealEnvironment vs RealEnv）：
  - RealEnv：直接与硬件交互的"驱动层"。它知道如何读写电机、控制夹爪等。
    它的 reset/step 直接操作真实的物理机器人。
  - AlohaRealEnvironment：环境和 Runtime 之间的"适配器层"。
    它处理：
      • 从 RealEnv 的观测中提取和转换数据
      • 对图像进行缩放、格式转换（HWC → CHW）、可选深度图过滤
      • 将 Runtime 规范的 action dict 映射为 RealEnv 可以理解的数组

数据流（一次 step）：
  物理世界 ──传感器──→ RealEnv.get_observation()
                          │ qpos (14维关节角度)
                          │ images (4个摄像头 HWC uint8)
                          ▼
                      AlohaRealEnvironment.get_observation()
                          │ 过滤深度图、缩放图像到224x224
                          │ 转换 HWC → CHW
                          │ 提取 state = qpos
                          ▼
                      Runtime → PolicyAgent → 策略推理
                          │
                          │ action["actions"] (14维动作数组)
                          ▼
                      AlohaRealEnvironment.apply_action()
                          │
                          ▼
                      RealEnv.step(action)
                          │ set_joint_positions（左臂/右臂）
                          │ set_gripper_pose（左夹爪/右夹爪）
                          │ time.sleep(0.02) 等待20ms
                          ▼
                      物理世界 ──执行动作──→ 下一个状态
"""

from typing import List, Optional  # noqa: UP035

import einops  # 强大的张量操作库（类似 einstein operations 的缩写）
from openpi_client import image_tools  # 图像工具函数（缩放、格式转换）
from openpi_client.runtime import environment as _environment  # Environment 抽象接口
from typing_extensions import override  # 类型提示：显式标记方法覆盖

from examples.aloha_real import real_env as _real_env  # ALOHA 真实机器人底层驱动


class AlohaRealEnvironment(_environment.Environment):
    """ALOHA 真机环境适配器 —— 将真实 ALOHA 机器人包装为 Runtime 可用的 Environment。

    这个类是对真实 ALOHA 机器人的"标准化包装"。
    它实现了 Runtime 期望的 Environment 接口，隐藏了底层硬件的所有细节。
    从 Runtime 的角度看，它和仿真环境（AlohaSimEnvironment）没有区别。

    适配器模式（Adapter Pattern）：
      这是经典的适配器模式应用 —— RealEnv 的接口（reset/fake/step 返回 dm_env.TimeStep）
      与 Runtime 要求的接口（get_observation/apply_action）不匹配，
      AlohaRealEnvironment 负责在这两者之间做转换：

        RealEnv 接口                     AlohaRealEnvironment 接口
      ┌──────────────┐                 ┌──────────────────────┐
      │ reset(fake)  │                 │ reset()              │
      │   → TimeStep │                 │   → 调用 reset()     │
      │              │ 适配器模式       │   → 缓存 TimeStep    │
      │ step(action) │ ──────────────→ │                      │
      │   → TimeStep │                 │ get_observation()    │
      │              │                 │   → 从缓存的 TimeStep│
      │ get_qpos()   │                 │     提取 qpos+images │
      │ get_images() │                 │     缩放+转置        │
      └──────────────┘                 │     返回标准 dict    │
                                       │                      │
                                       │ apply_action(action) │
                                       │   → step(action)     │
                                       │   → 缓存 TimeStep    │
                                       └──────────────────────┘

    动作空间（14 维）：
      [左臂关节1, ..., 左臂关节6, 左夹爪位置, 右臂关节1, ..., 右臂关节6, 右夹爪位置]
       0           5           6          7           12          13

    观测空间：
      - state:  14 维关节状态（位置/角度）
      - images: 4 个摄像头视图，经 resize + pad 到 (3, 224, 224)
        - cam_high:       顶部广角摄像头
        - cam_low:        底部摄像头
        - cam_left_wrist:  左腕部摄像头
        - cam_right_wrist: 右腕部摄像头
    """

    def __init__(
        self,
        reset_position: Optional[List[float]] = None,  # noqa: UP006,UP007
        render_height: int = 224,
        render_width: int = 224,
    ) -> None:
        """初始化 ALOHA 真机环境。

        这个构造函数做了三件事：
          1. 创建底层硬件驱动（RealEnv）—— 连接到真实的机械臂
          2. 设置图像处理参数 —— 模型要求的输入尺寸
          3. 初始化状态缓存

        关于 ROS 节点初始化（init_node）：
          ALOHA 机器人使用 ROS（Robot Operating System）进行通信。
          每个 ROS 程序都需要先初始化一个节点才能订阅和发布话题。
          这里的 init_node=True 表示由我们负责创建 ROS 节点。
          RealEnv 内部的 Recorder、ImageRecorder 等传入 init_node=False，
          表示它们复用已有的 ROS 节点（由 RealEnv 的 init_node 创建的主节点）。

        关于复位位姿（reset_position）：
          这是机器人每个 episode 开始前的"归位"姿态。
          从训练数据中统计得到，通常是将机器人抬起到一个安全位置：
          - 既不会碰撞到其他物体
          - 又不会让关节限位或拉扯线缆
          - 且能让摄像头看到完整的工作区域

        Args:
            reset_position: 机器人的起始关节位姿，长度为 6（只有臂，不包括夹爪）。
                            每个元素是关节角度（弧度）。
                            示例：[0, -0.96, 1.16, 0, -0.3, 0]
                            - None 则使用 RealEnv 中的默认值（出厂默认位姿）。
                            通常从远程策略服务器的 metadata["reset_pose"] 获取。

            render_height:  输出图像高度（像素），默认 224。
                            策略模型通常使用固定尺寸的输入，这里负责将
                            原始摄像头图像（480x640）缩放到模型要求的尺寸。

            render_width:   输出图像宽度（像素），默认 224。
                            224 是常见视觉模型（如 ViT、SigLIP）的输入尺寸。
        """
        # ── 创建底层硬件驱动 ──
        # make_real_env() 返回 RealEnv 实例，其中包含真正的 Interbotix 机器人控制对象。
        # init_node=True 表示：由 RealEnv 的构造函数创建 ROS 节点（roscore 初始化）。
        # 如果已有一个 ROS 节点在运行，这里应传 init_node=False 以避免冲突。
        #
        # reset_position 只取前 6 个值（臂关节），夹爪的复位在 _reset_gripper() 中处理。
        self._env = _real_env.make_real_env(init_node=True, reset_position=reset_position)

        # ── 图像处理参数 ──
        self._render_height = render_height  # 模型期望的图像高度
        self._render_width = render_width  # 模型期望的图像宽度

        # ── 状态缓存 ──
        # _ts（timestep）保存最近一次 reset() 或 step() 返回的 dm_env.TimeStep。
        # 这是观察者模式的应用：apply_action 时缓存观测，get_observation 时直接返回。
        # 这样可以避免重复从硬件读取数据，也保证了观测和动作的时序一致性。
        #
        # None 表示还没有调用过 reset()，此时 get_observation() 会抛出错误。
        self._ts = None

    @override
    def reset(self) -> None:
        """重置环境到初始状态 —— 让真实机器人回到起始位姿。

        这个方法由 Runtime 在以下时机调用：
          1. 每个 episode 开始时（让机器人从相同位姿开始执行任务）
          2. 所有 episode 结束后（安全归位，避免机器人悬在半空）

        内部流程：
          1. 重启夹爪电机（RealEnv 的 reset 中执行）
          2. 移动两个机械臂到复位位姿（_reset_joints）
          3. 先关闭夹爪、再打开夹爪（_reset_gripper）
             ── 这是为了清除夹爪的 PWM 控制模式，切回位置模式
          4. 读取复位后的观测（qpos、qvel、images等）
          5. 包装为 dm_env.TimeStep 并缓存

        安全说明：
          真实机器人的 reset() 是一个物理操作 —— 关节在电机的驱动下
          移动到指定位姿。如果复位速度太快或路径上有障碍物，可能导致碰撞。
          因此 RealEnv 中设置了 move_time=1（1 秒内缓慢到达），
          并先关闭夹爪再打开（避免夹爪在复位过程中夹到东西）。

        Notes:
          - 这个方法不返回任何值。需要观测的话后续调用 get_observation()。
          - reset() 必须先于任何 get_observation() 或 apply_action() 调用。
          - 可以多次调用，每次都会让机器人重新归位。
        """
        # 调用底层 RealEnv 的 reset()，返回包含初始观测的 TimeStep
        # RealEnv.reset(fake=False)：实际上是物理移动机器人
        # TimeStep.step_type = FIRST（标记这是一个 episode 的开始）
        # TimeStep.observation 包含完整的状态快照
        self._ts = self._env.reset()

    @override
    def is_episode_complete(self) -> bool:
        """检查当前 episode 是否完成。

        对于真实机器人，这个方法的"标准答案"是：总是返回 False。

        为什么不检测任务完成？
          1. 真机环境没有"任务成功"的自动检测机制。
             视觉物体检测、力反馈判断都需要额外的传感器和算法。
          2. episode 的结束由 Runtime 的 max_episode_steps 控制。
             这是一种"超时截断"策略 —— 无论任务是否完成，跑到预设步数就停止。
             这种方式简单可靠，不需要额外的感知能力。

        替代策略：
          如果需要检测任务完成（如"成功抓到物体就提前结束"），
          可以在这里添加基于视觉的检测逻辑，或从机器人电流检测抓取状态。

        Returns:
            bool: 总是返回 False。episode 由 Runtime 的 max_episode_steps 参数控制结束。
        """
        return False

    @override
    def get_observation(self) -> dict:
        """获取当前环境观测 —— 将 RealEnv 的原始数据转换为策略模型的输入格式。

        这个方法将 RealEnv 返回的 dm_env.TimeStep.observation 转换为
        策略模型期望的标准 dict 格式。

        原始数据 -> 转换后数据：
          ┌─────────────────────────────────────────────────────┐
          │ RealEnv 原始观测（来自机器人传感器）                    │
          │  {                                                   │
          │    "qpos":   np.ndarray [14]   ← 关节位置/夹爪位置    │
          │    "qvel":   np.ndarray [14]   ← 关节速度/夹爪速度    │
          │    "effort": np.ndarray [14]   ← 关节力矩             │
          │    "images": {                                       │
          │      "cam_high":        np.ndarray [480,640,3] ← HWC │
          │      "cam_low":         np.ndarray [480,640,3]       │
          │      "cam_left_wrist":  np.ndarray [480,640,3]       │
          │      "cam_right_wrist": np.ndarray [480,640,3]       │
          │    }                                                 │
          │  }                                                   │
          │                        ↓                              │
          │ AlohaRealEnvironment 转换输出（给策略模型）             │
          │  {                                                   │
          │    "state":  np.ndarray [14]   ← 只取 qpos 作为状态  │
          │    "images": {                                       │
          │      "cam_high":        np.ndarray [3,224,224] ← CHW │
          │      "cam_low":         np.ndarray [3,224,224]       │
          │      "cam_left_wrist":  np.ndarray [3,224,224]       │
          │      "cam_right_wrist": np.ndarray [3,224,224]       │
          │    }                                                 │
          │  }                                                   │
          └─────────────────────────────────────────────────────┘

        图像预处理流程：
          原始图像 (480x640x3, uint8)
            │
            ├─ 跳过深度图（如果摄像头有深度通道，删除 _depth 图像）
            │
            ├─ resize_with_pad (→ 224x224x3)
            │   • 保持宽高比缩放（避免图像变形）
            │   • 不足的部分用黑边填充（padding）
            │   • 因为 480:640 ≈ 3:4，目标 224:224 = 1:1
            │     所以先缩放到 168x224，再上下各填充 28 个黑像素
            │
            ├─ convert_to_uint8
            │   • 如果是浮点图（0.0~1.0），转换为 uint8（0~255）
            │   • 减少网络传输数据量
            │
            └─ einops.rearrange("h w c -> c h w")
                • 将通道维度从最后移到最前
                • 因为 PyTorch 模型使用 CHW 格式（通道、高、宽）
                • OpenCV / PIL 使用 HWC 格式（高、宽、通道）

        为什么只提取 qpos 作为 state？
          策略模型通常只需要当前关节位置来推理下一步动作。
          qvel 和 effort 信息在大多数策略中不是必须的。
          如果模型需要，可以在这里添加。

        Returns:
            dict: 包含以下键的字典：
                - "state":  np.ndarray shape=(14,)
                    14 维关节状态：[左臂6关节, 左夹爪, 右臂6关节, 右夹爪]
                - "images": dict[str, np.ndarray]
                    4 个摄像头图像，每个 shape=(3, 224, 224)，dtype=uint8

        Raises:
            RuntimeError: 如果在调用 reset() 之前调用此方法。
        """
        # ── 安全检查 ──
        # _ts 在 reset() 和 apply_action() 中设置，
        # 如果为 None，说明还没有调用过 reset()。
        # 这是一个常见的错误：忘记在循环开始前先调用 reset()。
        if self._ts is None:
            raise RuntimeError("Timestep is not set. Call reset() first.")

        # ── 提取观测数据 ──
        # self._ts 是 dm_env.TimeStep，其中 observation 是 OrderedDict
        obs = self._ts.observation

        # ── 过滤深度图像 ──
        # 有些摄像头配置同时输出 RGB 和 Depth（深度）图，
        # 深度图键名通常包含 "_depth" 后缀。
        # 策略模型（如 π₀）只用 RGB 图像，不需要深度信息。
        # 在这里删除深度图可以：
        #   1. 避免后续图像处理出错（深度图是单通道，resize 可能报错）
        #   2. 减少传入模型的数据量（少传 1~N 张图）
        #
        # list(obs["images"].keys()) 而不是直接 for k in obs["images"]：
        #   这是 Python 的常见陷阱 —— 遍历字典时不能修改字典（增删键）。
        #   list() 创建了一个键的副本，这才允许我们在循环中删除元素。
        for k in list(obs["images"].keys()):
            if "_depth" in k:
                del obs["images"][k]

        # ── 图像预处理 ──
        # 逐摄像头处理：缩放 + 类型转换 + 通道重排
        for cam_name in obs["images"]:
            # 1. 缩放图像到模型输入尺寸，保持宽高比并用黑边填充
            #    resize_with_pad 模仿 TensorFlow 的 tf.image.resize_with_pad
            #    返回的图像尺寸： (224, 224, 3)，仍然是 HWC 格式
            img = image_tools.resize_with_pad(
                obs["images"][cam_name],
                self._render_height,
                self._render_width,
            )

            # 2. 确保图像是 uint8 格式
            #    原始摄像头输出通常是 uint8（0~255），但某些仿真或处理步骤
            #    可能产生浮点图（0.0~1.0），需要转换以减少数据量
            img = image_tools.convert_to_uint8(img)

            # 3. 重排通道顺序：HWC → CHW
            #    OpenCV/PIL 使用 HWC（高、宽、通道）格式：
            #       图像形状: [480, 640, 3]  ← 高480，宽640，3通道
            #    PyTorch 模型使用 CHW（通道、高、宽）格式：
            #       图像形状: [3, 224, 224]   ← 3通道，高224，宽224
            #
            #    einops.rearrange 用字符串表达式描述重排操作：
            #      "h w c -> c h w" 表示将第0维（h）移到第1维，
            #      第1维（w）移到第2维，第2维（c）移到第0维。
            #
            #    等价实现：
            #      img = np.transpose(img, (2, 0, 1))
            obs["images"][cam_name] = einops.rearrange(img, "h w c -> c h w")

        # ── 组合最终输出 ──
        # 返回策略模型需要的标准格式：
        #   state:  关节位置（agent 的"本体感觉"）
        #   images: 摄像头图像（agent 的"视觉"）
        #
        # 注意：这里只提取了 qpos（位置），没有 qvel（速度）和 effort（力矩）。
        # 这是因为 Aloha 策略模型通常只用位置作为状态。
        # 如果模型需要速度/力矩信息，可以在训练配置中定义对应的输入变换。
        return {
            "state": obs["qpos"],
            "images": obs["images"],
        }

    @override
    def apply_action(self, action: dict) -> None:
        """在真实机器人上执行一个动作。

        这个方法将策略模型输出的动作指令发送到真实 ALOHA 机器人。

        动作格式：
          action["actions"] 是一个 14 维的 NumPy 数组：
            [左臂6关节角度] + [左夹爪位置] + [右臂6关节角度] + [右夹爪位置]
            0           5         6            7          12         13

          每个元素的意义：
            - 臂关节：绝对关节角度（弧度），直接写入电机位置控制器
            - 夹爪：归一化位置（0=闭合, 1=张开），先反归一化为关节值再写入

        执行过程（RealEnv.step）：
          1. 分割动作：前 7 维给左臂，后 7 维给右臂
          2. 设置关节位置：
             - puppet_bot_left.arm.set_joint_positions(left_action[:6])
             - puppet_bot_right.arm.set_joint_positions(right_action[:6])
             - 使用 blocking=False（非阻塞），立即返回
          3. 设置夹爪位置：
             - 通过 ROS 话题发送 JointSingleCommand
             - 先反归一化：归一化值 → 实际关节角度值
          4. 等待 20ms（time.sleep(constants.DT)）
             - 这 20ms 对应 50Hz 的控制频率
             - 给电机足够的时间开始运动
             - 也让物理世界有时间响应

        为什么 blocking=False？
          blocking=True 会等待电机到达目标位置才返回，但对于多步规划的策略，
          每步都是小幅移动（20ms 内的微小增量），不需要等待到位。
          非阻塞调用让电机在后台继续运动，主循环可以立即处理下一帧。

        时序保证：
          apply_action() 中的 time.sleep(0.02) 确保：
            - 控制频率稳定在 ~50Hz
            - 即使前序步骤（读取传感器、网络推理）有小的波动，执行频率仍保持稳定
            - 给电机一个"稳定的控制流"而非突发的指令突发

        Args:
            action: 动作字典，必须包含 "actions" 键。
                    action["actions"] 的形状通常为 (14,) 或 (action_horizon, 14)。
                    对于 ActionChunkBroker 已经取单步的情况，这里是 (14,)。
        """
        # 将策略动作发送到底层 RealEnv 执行
        # RealEnv.step() 返回新的 TimeStep（包含执行后的观测），
        # 将其缓存到 _ts 中，供下一次 get_observation() 使用。
        #
        # 这样设计的好处：step 后立刻获取新观测，保证观测和动作的时序同步。
        # 比如 step t 的动作执行后，_ts 中保存的就是 t+1 时刻的观测。
        self._ts = self._env.step(action["actions"])

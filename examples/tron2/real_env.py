"""
Tron2 真实机器人环境（Tron2 Real Robot Environment）

【设计目标】
提供一个高层次的机器人环境封装，屏蔽底层硬件通信细节，让策略（Policy）可以
通过简单的 reset() / step() / get_obs() 接口与真实 Tron2 机器人交互。

【核心功能】
1. 机器人控制 —— 通过 TCP/UDP 与 Tron2 机器人控制器通信，下发关节角度指令
2. 多相机图像采集 —— 同时从头部、左腕、右腕三个 RealSense 相机获取 RGB 图像
3. 观测与动作的时间同步 —— 确保图像和关节状态在时间上对齐
4. 轨迹插值 —— 在两次策略输出之间进行线性插值，使机器人运动更平滑

【整体架构】
    ┌──────────────────────────────────────────────────┐
    │                  Tron2Env                         │
    │                                                    │
    │  ┌──────────┐    ┌────────────────┐               │
    │  │  Tron2   │    │MultiCameraManager│              │
    │  │ (机器人)  │    │   (多相机管理)   │              │
    │  └────┬─────┘    └───────┬────────┘               │
    │       │                  │                         │
    │       ▼                  ▼                         │
    │  ┌──────────────────────────────────┐              │
    │  │   时间同步 + 数据组装             │              │
    │  └──────────────────────────────────┘              │
    │       │                                            │
    │       ▼                                            │
    │  ┌──────────┐                                     │
    │  │ get_obs()│ → {"state": [...], "images": {...}}  │
    │  └──────────┘                                     │
    └──────────────────────────────────────────────────┘
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import cv2
import numpy as np
from PIL import Image

from robot_utils import Tron2, Tron2Config, JointIndex


# ============================================================================
# 配置数据结构
# ============================================================================

@dataclass
class CameraConfig:
    """相机配置 —— 定义所有与相机相关的参数。

    【相机命名约定】
    物理相机（通过序列号识别） → 内部名称（由 CameraManager 管理） → 观测名称（对外暴露）

    例如：
        RealSense 序列号 "245022302696" → head_camera_image → cam_high
        这是因为：
        - 头部摄像头安装在机器人头部，提供"高角度"（high）全局视角
        - 在机器人学中，"cam_high" 是标准命名约定（相对于 "cam_low"）
    """
    # 相机内部名称列表（CameraManager 使用的名称标识）
    camera_names: List[str] = field(default_factory=lambda: [
        "head_camera_image",     # 头部摄像头（全局视角，相当于人眼）
        "left_wrist_image",      # 左腕部摄像头（第一人称视角，观察左手操作区域）
        "right_wrist_image"      # 右腕部摄像头（第一人称视角，观察右手操作区域）
    ])

    # 对外暴露的观测名称（策略/模型端看到的名称）
    # 注意：pi0/pi0.5 模型预定义的标准摄像头命名空间是 "cam_high" 等，
    # 所以这里做了一个重命名映射。
    obs_camera_names: List[str] = field(default_factory=lambda: [
        "cam_high",             # 对应 head_camera_image
        "cam_left_wrist",       # 对应 left_wrist_image
        "cam_right_wrist"       # 对应 right_wrist_image
    ])

    # 相机分辨率 (H, W, C) —— 高×宽×通道数
    # 这里是 (480, 640, 3)，即 VGA 分辨率的 RGB 图像
    # 【为什么不用更高分辨率？】
    # - 机器人控制需要低延迟（通常 10-50Hz），高分辨率会降低帧率
    # - 视觉策略通常会将图像缩放到 224x224，过高的原始分辨率浪费带宽
    resolution: Tuple[int, int, int] = (480, 640, 3)

    # 相机帧队列的最大缓冲区大小
    # 如果队列满了，旧的帧会被丢弃，确保总是使用最新的帧
    max_queue_size: int = 10

    # 是否保存调试图像（用于排查视觉问题）
    save_debug_images: bool = True
    debug_image_dir: str = "./debug_images"


@dataclass
class EnvConfig:
    """环境配置 —— 汇集所有控制机器人环境行为的参数。

    这个类是配置的"总入口"，它将机器人配置、相机配置、控制参数
    集中到一个地方，简化了 API 调用。
    """
    # 机器人硬件配置（IP、初始关节位置、PID 参数等）
    robot_config: Tron2Config = field(default_factory=Tron2Config)

    # 多相机配置
    camera_config: CameraConfig = field(default_factory=CameraConfig)

    # 轨迹插值点数 —— 控制动作执行的平滑度
    # 【工作原理】
    # 策略以低频（如 10Hz）输出一个"目标动作"，
    # 插值算法在这个目标和当前实际位置之间插入 interp_points 个中间点，
    # 机器人控制循环逐一执行这些中间点。
    # 【值的选择】
    # - 值越大 → 运动越平滑，但执行动作的总时间越长
    # - 值越小 → 运动越"生硬"，响应越快
    # - 典型值：6~10 之间
    #
    # 【图解】
    #   策略输出:     A ────────────────────────────→ B  （只有起点和终点）
    #   interp_points=3:  A ──── P1 ──── P2 ──── B   （插了 2 个中间点）
    #   实际执行:     A → P1 → P2 → B                  （逐点执行）
    interp_points: int = 8

    # 时间同步容差（秒）—— 关节状态和图像时间戳的最大允许差异
    # 相机和关节传感器硬件不同步，它们的时钟有微小差异。
    # 如果时间差小于此值，认为数据是"同步的"。
    # 如果超出，会尝试重新获取关节数据来对齐。
    # 【为什么需要同步？】
    # 机器人控制需要"在某个时刻 t，我看到的是这样，我的关节位置是这样"。
    # 如果图像是 t=0 时刻的，但关节状态是 t=0.1 时刻的，那么
    # 策略的推理结果 (在 t=0 状态上执行动作 A) 与实际状态 (t=0.1) 不匹配。
    # 【典型值】
    # - 0.01 秒（10 毫秒）：严格同步，适合精密操作
    # - 0.05 秒：宽松同步，适合对时间不敏感的任务
    time_sync_tolerance: float = 0.01
    # 时间同步的最大重试次数
    time_sync_max_retries: int = 3

    # 夹爪初始化开口度（归一化到 0-1 范围）
    # 0 = 完全闭合，1 = 完全张开
    # 在 reset() 时，两个夹爪都会移动到此开口度。
    # 【为什么设为 0.9 而不是 1.0？】
    # 完全张开可能会使夹爪碰到机械限位，留一点余量更安全。
    init_gripper_opening: float = 0.9

    # 原始配置字典 —— 用于透传给其他底层组件
    # 这是一个"逃生舱"，当配置参数无法被 EnvConfig 的字段覆盖时，
    # 可以通过这个字典直接传递给 MultiCameraManager.from_config()。
    # 例如，摄像头序列号到名称的映射就是通过这个字典传入的。
    raw_config: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Tron2 机器人环境
# ============================================================================

class Tron2Env:
    """Tron2 机器人环境 —— 机器人控制的核心抽象层。

    【设计模式：环境封装（Environment Wrapper）】
    类似 OpenAI Gym 的 Env 接口设计，提供了标准化的交互方式：
        reset()  →  重置环境，返回初始观测
        step()   →  执行动作，更新环境状态
        get_obs() → 获取当前观测（图像 + 关节状态）

    【典型用法】
        with Tron2Env(config) as env:
            obs = env.reset()
            while True:
                action = policy(obs)  # 策略推理
                env.step(action)      # 执行动作
                obs = env.get_obs()   # 获取新观测

    Examples:
        >>> config = EnvConfig(robot_config=Tron2Config(robot_ip="10.192.1.2"))
        >>> env = Tron2Env(config)
        >>> obs = env.reset()
        >>> action = np.zeros(16)  # 16维动作
        >>> env.step(action)
    """

    def __init__(self, config: Optional[EnvConfig] = None):
        """初始化环境 —— 建立与真实机器人和相机的连接。

        初始化流程：
        1. 设置日志系统
        2. 初始化机器人控制器（建立 TCP 连接）
        3. 初始化多相机管理器（启动相机流）
        4. 创建调试图像保存目录

        注意：这个构造函数会阻塞，直到相机预热完成（约 3 秒）。

        Args:
            config: 环境配置。如果为 None，使用所有默认参数。
                    大多数情况下你需要自定义 robot_config.robot_ip。
        """
        self.config = config or EnvConfig()

        # 设置日志系统（便于调试和监控）
        self._setup_logger()

        # ================================================================
        # 初始化机器人控制器
        # Tron2 是一个与机器人底层硬件通信的封装。
        # 它通过 TCP/UDP 协议向机器人控制器发送关节角度指令，
        # 并接收编码器反馈的当前关节状态。
        # ================================================================
        self.logger.info("正在初始化机器人控制器...")
        self.robot = Tron2(self.config.robot_config)

        # ================================================================
        # 初始化多相机管理器
        # 使用 RealSense 相机（也可能是其他 USB 相机），
        # 同时从多个视角采集 RGB 图像。
        # ================================================================
        self.logger.info("正在初始化相机...")
        self.camera_manager = self._init_camera()

        # ---- 状态管理 ----
        # 上一次执行的动作（用于轨迹插值的起点）
        # 初始为 None，首次 step() 时不进行插值
        self.last_action: Optional[np.ndarray] = None

        # 初始关节位置（来自配置，在 reset() 中会使用）
        self.init_joints = self.config.robot_config.init_joints

        # ---- 创建调试目录 ----
        if self.config.camera_config.save_debug_images:
            Path(self.config.camera_config.debug_image_dir).mkdir(parents=True, exist_ok=True)

        self.logger.info("环境初始化完成")

    def _setup_logger(self):
        """设置日志系统。

        配置日志输出的格式，包括时间戳（毫秒级精度）、模块名、日志级别。
        格式示例：[2026-07-16 14:30:00.123] [Tron2Env] [INFO] 消息内容

        毫秒级时间戳对于调试机器人控制的时间同步问题非常关键。
        """
        self.logger = logging.getLogger("Tron2Env")
        if not self.logger.handlers:
            # 避免重复添加 handler（当多次创建 Tron2Env 实例时）
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(asctime)s.%(msecs)03d] [%(name)s] [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _init_camera(self):
        """初始化多相机管理器。

        【相机初始化流程】
        1. 导入 MultiCameraManager（延迟导入，避免没有安装 RealSense SDK 时报错）
        2. 如果配置中有 raw_config，通过 from_config() 加载自定义相机映射
           （例如摄像头序列号到名称的映射）
        3. 否则使用默认初始化方式
        4. 启动相机流（开始采集图像）
        5. 等待相机预热（约 3 秒），让自动曝光和自动白平衡稳定下来

        Returns:
            MultiCameraManager: 多相机管理器实例

        Raises:
            ImportError: 如果未安装 realsense_image 模块
        """
        try:
            from realsense_image import MultiCameraManager
        except ImportError:
            self.logger.error("无法导入 MultiCameraManager，请确保 realsense_image 模块已安装")
            raise

        # 如果配置中包含原始配置字典，传递给 MultiCameraManager
        # from_config() 可以解析 YAML 格式的配置文件
        if hasattr(self.config, 'raw_config'):
            camera_manager = MultiCameraManager.from_config(self.config.raw_config)
        else:
            camera_manager = MultiCameraManager(
                max_queue_size=self.config.camera_config.max_queue_size
            )

        # 启动相机采集线程
        camera_manager.start_capture()

        # 【为什么需要等待 3 秒？】
        # RealSense 相机启动后需要一段时间让自动曝光（AE）收敛。
        # 刚启动时的图像可能过亮或过暗，等待 3 秒可以获得稳定亮度的图像。
        self.logger.info("相机预热中...")
        time.sleep(3.0)

        return camera_manager

    # ========================================================================
    # 环境接口（Environment Interface）
    # ========================================================================

    def reset(self) -> Dict:
        """重置环境到初始状态。

        【做什么？】
        1. 获取当前观测（包括图像和关节状态）
        2. 验证图像分辨率是否与配置一致
        3. 如果机器人不在初始位置，通过 move 指令移动到初始位置
        4. 将夹爪打开到配置的初始开口度

        【为什么 reset() 要做这么多事？】
        机器人不是仿真环境，不能"一键重置"。reset() 的作用是：
        - 确保机器人在控制循环开始时处于已知的安全姿态
        - 验证所有传感器正常工作（图像尺寸、关节反馈）
        - 为后续的控制循环建立"初始条件"

        Returns:
            初始观测字典 {
                'state': np.ndarray,   # 关节状态向量
                'images': Dict         # 多视角图像字典
            }
        """
        self.logger.info("重置环境...")

        # ================================================================
        # 第 1 步：获取当前观测
        # ================================================================
        obs = self.get_obs()

        # ================================================================
        # 第 2 步：验证图像尺寸
        # ================================================================
        expected_shape = self.config.camera_config.resolution
        for cam_name in self.config.camera_config.obs_camera_names:
            actual_shape = obs['images'][cam_name].shape
            if actual_shape != expected_shape:
                # 图像尺寸不匹配可能意味着相机配置错误或硬件问题
                # 这里只打印警告而不中断执行，因为某些情况下可以容忍
                self.logger.warning(
                    f"{cam_name} 分辨率不匹配: 期望{expected_shape}, 实际{actual_shape}"
                )

        # ================================================================
        # 第 3 步：验证并移动到初始位置
        # ================================================================
        if self.init_joints is not None:
            # 获取当前机械臂的实际关节角度
            current_state = obs['state']
            arm_states = np.concatenate([
                current_state[JointIndex.LEFT_ARM],
                current_state[JointIndex.RIGHT_ARM]
            ])
            init_arm = np.array(self.init_joints)

            # 计算机械臂当前姿态与目标初始姿态的最大误差
            error = np.abs(arm_states - init_arm).max()

            if error > 0.05:
                # 误差超过 0.05 弧度（约 2.86 度），需要移动到初始位置
                self.logger.warning(f"机器人未在初始位置，最大误差: {error:.4f}")
                # wait_until_reached 会阻塞直到机器人到达目标位置
                self.robot.wait_until_reached(self.init_joints, tolerance=0.05)

        # ================================================================
        # 第 4 步：初始化夹爪开口度
        # ================================================================
        # 将夹爪打开到配置的初始开口度
        # 这里通过 step() 来执行，而不是直接调用 set_gripper()，
        # 因为 step() 会处理插值和日志记录
        test_action = obs['state'].copy()
        test_action[JointIndex.LEFT_GRIPPER] = self.config.init_gripper_opening
        test_action[JointIndex.RIGHT_GRIPPER] = self.config.init_gripper_opening
        self.step(test_action)

        self.logger.info("环境重置完成")
        return obs

    def step(self, action: Union[List[float], np.ndarray]):
        """执行一个动作 —— 将策略输出的动作发送给机器人。

        【动作维度说明】
        动作向量的长度可以是 14、16 或 18，含义如下：

        ┌─────────────┬──────┬──────────────────────────────────────┐
        │  维度       │ 长度  │ 内容                                 │
        ├─────────────┼──────┼──────────────────────────────────────┤
        │ MOVEJ_DIM   │  14  │ [左臂关节7, 右臂关节7]                │
        │ SERVOJ_DIM  │  16  │ [左臂7, 左夹爪1, 右臂7, 右夹爪1]      │
        │ STATE_DIM   │  18  │ [左臂7, 左夹爪1, 右臂7, 右夹爪1, 头2] │
        └─────────────┴──────┴──────────────────────────────────────┘

        【执行流程】
        1. 提取机械臂关节动作（14 维）
        2. 提取头部动作（2 维）：如果动作不包含头部，使用当前头部位置
        3. 组合为 16 维伺服动作
        4. 提取并缩放夹爪动作（乘以 100）
        5. 如果是首次调用，直接下发动作（不插值）
        6. 否则，通过线性插值生成平滑轨迹，逐点下发

        【为什么需要轨迹插值？】
        如果不插值，机器人会尝试"跳"到目标位置。
        当两次动作差异较大时，这会导致机器人剧烈抖动或触发安全保护。
        插值让机器人在多个控制周期内逐步到达目标，运动更平滑、更安全。

        Args:
            action: 动作向量，支持 14/16/18 维

        Raises:
            ValueError: 如果动作维度不是 14/16/18
        """
        # ---- 输入验证 ----
        if isinstance(action, list):
            action = np.array(action)

        if len(action) not in [JointIndex.MOVEJ_DIM, JointIndex.SERVOJ_DIM, JointIndex.STATE_DIM]:
            raise ValueError(
                f"动作维度应为{JointIndex.MOVEJ_DIM}/{JointIndex.SERVOJ_DIM}/{JointIndex.STATE_DIM}, "
                f"实际{len(action)}"
            )

        # ================================================================
        # 第 1 步：提取机械臂关节动作（前 14 维）
        # 结构：[左臂7, 右臂7]
        # 注意这里跳过了夹爪维度（action[7] 是左夹爪，action[15] 是右夹爪）
        # ================================================================
        arm_action = np.concatenate([
            action[JointIndex.LEFT_ARM],
            action[JointIndex.RIGHT_ARM]
        ])

        # ================================================================
        # 第 2 步：提取头部动作（2 维：[俯仰, 偏航]）
        # ================================================================
        if len(action) >= JointIndex.STATE_DIM:
            # 如果动作包含头部维度，直接使用
            head_action = action[JointIndex.HEAD]
        else:
            # 如果动作不包含头部，使用当前头部位置
            # 这样即使策略不输出头部动作，头部也能保持在当前位置
            # 注意：这里直接访问了 robot._state_lock，是一个"逃生舱"操作
            with self.robot._state_lock:
                curr_states = self.robot.joint_states['states']
                if len(curr_states) >= JointIndex.STATE_DIM_WITH_HEAD:
                    head_action = np.array(curr_states[JointIndex.HEAD])
                else:
                    # 如果连状态数据都不可用，默认头部保持水平
                    head_action = np.array([0.0, 0.0])

        # ================================================================
        # 第 3 步：组合为 16 维伺服动作（14 臂 + 2 头）
        # ================================================================
        full_servo_action = np.concatenate([arm_action, head_action])

        # ================================================================
        # 第 4 步：提取并处理夹爪动作
        # 【为什么乘以 100？】
        # 策略输出的夹爪动作是归一化的 [0, 1] 范围，
        # 但机器人底层控制器期望的是 [0, 100] 的百分比值。
        # 0% = 完全闭合，100% = 完全张开。
        # ================================================================
        gripper_action = np.array([
            action[JointIndex.LEFT_GRIPPER],
            action[JointIndex.RIGHT_GRIPPER]
        ]) * 100.0
        gripper_action = np.clip(gripper_action, 0, 100)  # 安全限幅

        # ================================================================
        # 第 5 步：首次调用处理
        # 如果是 reset() 后的第一次 step()，last_action 为 None，
        # 此时没有"上一次位置"来做插值，所以直接下发动作。
        # ================================================================
        if self.last_action is None:
            # servoj: 关节速度控制模式（非阻塞，持续发送位置指令）
            # 相对于 movej（点到点运动），servoj 可以实现更平滑的连续控制
            self.robot.servoj(full_servo_action)
            # 设置夹爪开度
            self.robot.set_gripper(
                left_opening=gripper_action[0],
                right_opening=gripper_action[1]
            )
            self.last_action = full_servo_action
            return

        # ================================================================
        # 第 6 步：非首次调用 —— 设置夹爪并进行轨迹插值
        # ================================================================

        # 可选：夹爪死区处理（注释掉了）
        # 当夹爪动作小于 10% 时，强制设为 0（完全闭合）
        # 这可以防止微小抖动导致夹爪误动作
        # gripper_action[0] = gripper_action[0] if gripper_action[0] > 10 else 0
        # gripper_action[1] = gripper_action[1] if gripper_action[1] > 10 else 0

        # 设置夹爪（夹爪通常不需要插值，可以快速响应）
        self.robot.set_gripper(
            left_opening=gripper_action[0],
            right_opening=gripper_action[1]
        )

        # 【核心：轨迹插值执行】
        # 在 last_action（当前位置）和 full_servo_action（目标位置）之间
        # 生成 interp_points 个中间点，然后逐一发送给机器人。
        # 这样做的好处：
        # 1. 运动更平滑（机器人在每个控制周期只移动一小步）
        # 2. 更安全（避免大幅度跳跃动作）
        # 3. 符合机器人控制频率要求
        # ================================================================
        interpolated_traj = self._interpolate_trajectory(
            start=self.last_action,
            end=full_servo_action,
            num_points=self.config.interp_points
        )

        # 执行插值轨迹（从索引 1 开始，跳过起点，因为起点就是当前位置）
        for i in range(1, len(interpolated_traj)):
            time_servoj = time.time()
            self.robot.servoj(interpolated_traj[i])
            time_servoj2 = time.time()
            dt = time_servoj2 - time_servoj
            # 可选的频率监控日志（注释掉了）
            # self.logger.info(f"time1 : {time_servoj}, time2 : {time_servoj2}, control rate: {1/dt}")

        # 更新历史动作（供下一次插值使用）
        self.last_action = full_servo_action

    def get_obs(self) -> Dict:
        """获取当前观测 —— 同步采集图像和关节状态。

        【观测采集流水线】
        get_images()      get_qpos()
            │                 │
            ▼                 ▼
        RGB 图像 ────────── 关节状态
            │                 │
            ▼                 ▼
        时间同步检查 ──── 如果时间差 > 容差 → 重新获取关节状态
            │
            ▼
        组装修观测字典

        【为什么需要时间同步？】
        机器人的关节状态和图像来自不同的硬件传感器：
        - 关节状态：来自电机编码器，通过 TCP 读取（延迟较低）
        - 图像：来自 RealSense 相机，通过 USB 传输（延迟较高）

        这两个数据流到达的时间不同步，但策略推理需要"同一时刻"的
        观测数据。_sync_observation() 负责对齐它们的时间戳。

        Returns:
            观测字典: {
                'state': np.ndarray,       # 关节状态 (16 维)
                'images': Dict[str, np.ndarray]  # {摄像头名: 图像}
            }
        """
        # ================================================================
        # 第 1 步：获取图像（同时记录图像的时间戳）
        # 先获取图像，因为图像的延迟通常比关节状态大，
        # 后续再获取时间戳更"新鲜"的关节状态来对齐。
        # ================================================================
        rgb_images = self._get_images()
        img_timestamp = rgb_images['head_camera_image_timestamp']

        # 可选：保存调试图像到磁盘（排查视觉问题）
        if self.config.camera_config.save_debug_images:
            self._save_debug_images(rgb_images)

        # ================================================================
        # 第 2 步：获取关节状态（读取编码器反馈）
        # get_joint_states() 从机器人控制器获取当前关节角度，
        # 返回的数据中包含硬件时间戳（毫秒级）。
        # ================================================================
        qpos_dict = self._get_qpos()
        joint_timestamp = qpos_dict['timestamp'] / 1000.0  # 毫秒 → 秒

        # 日志：打印时间戳差异，用于调试同步问题
        self.logger.debug(
            f"时间戳 - 关节: {joint_timestamp:.3f}s, 图像: {img_timestamp:.3f}s, "
            f"差值: {joint_timestamp - img_timestamp:.3f}s"
        )

        # ================================================================
        # 第 3 步：时间同步
        # 如果关节和图像的时间戳差异过大，尝试重新获取关节状态
        # ================================================================
        synced_qpos = self._sync_observation(
            img_timestamp=img_timestamp,
            initial_qpos=qpos_dict,
            using_sync=True,
        )

        # ================================================================
        # 第 4 步：构建标准化的观测字典
        # ================================================================
        obs = {
            # 只取前 16 维状态（左臂7+左夹爪1+右臂7+右夹爪1）
            # 注意：这里丢弃了头部状态（索引 16-17）
            # 因为 π₀ 模型的 state 输入设计是 16 维，不包含头部
            "state": np.array(synced_qpos['states'][:16]),
            "images": {
                # 将内部的摄像头名称映射为策略期望的名称
                self.config.camera_config.obs_camera_names[0]: rgb_images['head_camera_image'],
                self.config.camera_config.obs_camera_names[1]: rgb_images['left_wrist_image'],
                self.config.camera_config.obs_camera_names[2]: rgb_images['right_wrist_image']
            }
        }

        return obs

    # ========================================================================
    # 私有方法
    # ========================================================================

    def _interpolate_trajectory(
        self,
        start: np.ndarray,
        end: np.ndarray,
        num_points: int
    ) -> np.ndarray:
        """线性插值生成运动轨迹。

        【数学原理】
        在起始位置和结束位置之间均匀插入 num_points 个中间点。
        对每一个维度（每个关节）独立进行线性插值。

        公式：p(t) = start + t * (end - start),  t ∈ {0, 1/(n-1), 2/(n-1), ..., 1}

        【为什么是线性插值？】
        - 简单、计算快（机器人控制需要高频执行，不能有过多的计算开销）
        - 对于大部分操作任务，线性插值已经足够平滑
        - 如果需要更平滑的轨迹，可以考虑三次样条插值（Cubic Spline），
          但计算量更大

        【图解（num_points=5）】
        dim_i:  s ──── p1 ──── p2 ──── p3 ──── e
        t:      0    0.25    0.50    0.75     1

        Args:
            start: 起始位置 (16 维：14 臂 + 2 头)
                  通常是上一次执行的动作（last_action）
            end: 结束位置 (16 维)
                  通常是策略当前输出的目标动作
            num_points: 插值点的数量（包括起点和终点）

        Returns:
            形状为 (num_points, 16) 的插值轨迹数组
            索引 0 = 起点，索引 -1 = 终点
        """
        # 生成均匀分布的插值参数 t
        # linspace(0, 1, num_points) 例如：num_points=5 → [0, 0.25, 0.5, 0.75, 1]
        t = np.linspace(0, 1, num_points)

        # 初始化插值轨迹数组
        interpolated = np.zeros((num_points, JointIndex.SERVOJ_DIM))

        # 对每个关节维度独立进行一维线性插值
        # np.interp(t, [0, 1], [start[i], end[i]]) 实现了：
        #   p(t) = start[i] + t * (end[i] - start[i])
        for i in range(JointIndex.SERVOJ_DIM):
            interpolated[:, i] = np.interp(t, [0, 1], [start[i], end[i]])

        return interpolated

    def _get_qpos(self) -> Dict:
        """获取当前关节状态（位置/角度）。

        通过机器人控制器读取电机编码器的反馈值。
        这些值反映了机器人当前实际的关节角度（而非目标角度）。

        Returns:
            dict: 包含 'states'（关节角度数组）和 'timestamp'（硬件时间戳，毫秒）的字典
        """
        return self.robot.get_joint_states(timeout=0.5)

    def _get_images(self) -> Dict:
        """获取所有相机的最新帧。

        【图像处理流程】
        1. 从 CameraManager 获取所有相机的最新帧
        2. 对每个相机的帧：
           a. 提取彩色图像
           b. 将 BGR 格式转换为 RGB 格式
              【为什么需要 BGR → RGB？】
              大多数相机驱动（如 OpenCV 的 VideoCapture、RealSense SDK）
              默认输出 BGR 格式，但机器学习模型通常期望 RGB 格式。
              如果直接使用 BGR 图像训练/推理，红色和蓝色通道会颠倒，
              导致模型性能下降（因为颜色信息被错误解读）。
           c. 提取时间戳（用于与关节状态同步）
        3. 组装为图像字典

        Returns:
            图像字典: {
                'head_camera_image': np.ndarray,      # [H, W, C] RGB 图像
                'head_camera_image_timestamp': float,  # 时间戳（秒）
                'left_wrist_image': np.ndarray,
                'left_wrist_image_timestamp': float,
                'right_wrist_image': np.ndarray,
                'right_wrist_image_timestamp': float,
            }
        """
        all_frames = self.camera_manager.get_all_latest_frames()
        image_dict = {}

        for camera_name, frame_data in all_frames.items():
            if frame_data is not None:
                # BGR → RGB：反转最后一个维度的通道顺序
                # [:, :, ::-1] 表示取所有行、所有列、通道逆序（BGR→RGB）
                image_dict[camera_name] = frame_data['color'][:, :, ::-1]
                # 保存时间戳（用于后续的时间同步）
                image_dict[f'{camera_name}_timestamp'] = frame_data['timestamp']

        return image_dict

    def _sync_observation(
        self,
        img_timestamp: float,
        initial_qpos: Dict,
        using_sync: bool = False
    ) -> Dict:
        """时间同步：以图像时间戳为基准，对齐关节状态。

        【为什么需要这个函数？】
        在机器人系统中，不同传感器的时间戳天然不同步：
        - 相机在时刻 T1 曝光 → 图像到达 PC 端时间 T1 + Δt1
        - 关节编码器在时刻 T2 采样 → 数据到达 PC 端时间 T2 + Δt2
        - 一般来说 Δt1（图像传输）>> Δt2（状态传输）
        - 所以图像时间戳比关节时间戳"更老"

        时间同步的目标是找到与图像时间戳最接近的关节状态，
        确保"图像中看到的姿态"和"关节读数反映的姿态"是同一时刻的。

        【同步策略】
        1. 如果时间差在容差范围内 → 直接使用（数据足够新）
        2. 如果关节比图像晚 → 使用当前数据（图像已经是最新的）
        3. 如果关节比图像早得多 → 重新获取（等待新的关节数据到达）

        Args:
            img_timestamp: 图像时间戳（秒）
            initial_qpos: 初始关节状态
            using_sync: 是否启用同步（用于调试，可以关闭同步检查）

        Returns:
            同步后的关节状态字典
        """
        joint_timestamp = initial_qpos['timestamp'] / 1000.0
        time_diff = abs(joint_timestamp - img_timestamp)
        time_dif = joint_timestamp - img_timestamp  # 带符号的差值，用于日志

        # 如果关闭同步，直接返回（用于调试/性能测试）
        if not using_sync:
            self.logger.info(f"(sync disabled, diff={time_dif:.4f}s)")
            return initial_qpos

        # Case 1: 时间差在容差范围内 → 同步成功
        if time_diff <= self.config.time_sync_tolerance:
            self.logger.debug(f"✅ 时间同步成功 (diff={time_dif:.4f}s)")
            return initial_qpos

        # Case 2: 关节时间戳 > 图像时间戳
        # 说明关节数据比图像更新，但误差超出容差。
        # 这里只记录警告，不做特殊处理。
        # 因为"最新的关节数据 + 稍旧的图像"比"等关节数据变旧"更好。
        if joint_timestamp > img_timestamp:
            self.logger.warning(f"⚠️ joint later than img (diff={time_dif:.4f}s)")
            return initial_qpos

        # Case 3: 关节时间戳 < 图像时间戳（关节数据太旧）
        # 需要等待新的关节数据到达
        self.logger.debug(f"⚠️ 时间差过大 ({time_dif:.4f}s), 尝试重新同步...")

        for retry in range(self.config.time_sync_max_retries):
            time.sleep(0.005)  # 等待 5 毫秒，让新数据有机会到达

            # 重新获取关节状态
            qpos_dict = self._get_qpos()
            joint_timestamp = qpos_dict['timestamp'] / 1000.0
            time_diff = abs(joint_timestamp - img_timestamp)

            # 检查是否满足容差要求
            if time_diff <= self.config.time_sync_tolerance:
                self.logger.debug(
                    f"✅ 时间同步成功 (尝试{retry+1}次, diff={time_dif:.4f}s)"
                )
                return qpos_dict

        # 所有重试都失败：使用最新获取的数据
        # 注意：这里的时间差可能仍然很大，但与其阻塞等待，不如
        # 使用最新数据继续运行。时间同步是一个"尽力而为"的过程。
        self.logger.warning(
            f"⚠️ 时间同步失败 ({self.config.time_sync_max_retries}次重试后), "
            f"使用最新数据 joint{joint_timestamp}-img{img_timestamp}(diff={time_dif:.4f}s)"
        )
        return qpos_dict

    def _save_debug_images(self, rgb_images: Dict):
        """将当前帧保存为 JPEG 图像文件，用于调试。

        这些图像可以帮助排查：
        - 相机是否正常工作（黑屏、花屏等）
        - 曝光是否合适（过曝/欠曝）
        - 相机视角是否调整到位
        - 场景中是否有策略难以处理的光照条件

        Args:
            rgb_images: 包含 RGB 图像数据的字典
        """
        debug_dir = Path(self.config.camera_config.debug_image_dir)
        timestamp = time.time()

        for key in ['head_camera_image', 'left_wrist_image', 'right_wrist_image']:
            if key in rgb_images:
                img = Image.fromarray(rgb_images[key])
                save_path = debug_dir / f"{key}.jpg"
                img.save(save_path)

    def close(self):
        """关闭环境并释放所有资源。

        这个方法应该在任何时候退出机器人控制循环时调用。
        它在上下文管理器退出时自动调用。

        释放的资源包括：
        - 机器人控制器连接（TCP socket）
        - 相机采集线程
        """
        self.logger.info("关闭环境...")

        # 断开机器人连接（向机器人发送停止指令，释放控制权）
        if hasattr(self, 'robot'):
            self.robot.disconnect()

        # 停止相机采集（释放相机资源）
        if hasattr(self, 'camera_manager'):
            self.camera_manager.stop_capture()

        self.logger.info("环境已关闭")

    # ---- 上下文管理器支持 ----
    # 通过 __enter__ 和 __exit__，Tron2Env 可以用于 with 语句：
    #   with Tron2Env(config) as env:
    #       obs = env.get_obs()
    # 这样即使在控制循环中发生异常，close() 也会被自动调用，
    # 确保机器人安全停止。

    def __enter__(self):
        """上下文管理器入口 —— 进入 with 块时返回 self。"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出 —— 离开 with 块时自动释放资源。

        Args:
            exc_type: 异常类型（如果发生异常）
            exc_val: 异常值
            exc_tb: 异常回溯
        """
        self.close()


# ============================================================================
# 策略包装器
# ============================================================================

class PolicyWrapper:
    """策略包装器基类。

    定义了一个统一的 get_action() 接口，具体的策略实现
    （本地策略、WebSocket 远程策略等）通过继承这个基类来提供
    不同的推理后端。

    类似"策略模式"（Strategy Pattern）：
    - 环境（Tron2Env）只关心调用 get_action(obs) 得到动作
    - 具体的推理方式由 PolicyWrapper 的子类决定
    """

    def get_action(self, observation: Dict) -> np.ndarray:
        """根据观测获取动作。

        Args:
            observation: 观测字典（来自 env.get_obs()）

        Returns:
            动作数组，形状为 [action_horizon, action_dim]
            其中 action_horizon 是策略一次预测的多个未来动作（动作块），
            action_dim 是动作空间维度。

        Raises:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError


class WebsocketPolicyWrapper(PolicyWrapper):
    """基于 WebSocket 的远程策略客户端。

    通过 WebSocket 连接到一个远程策略推理服务器。
    典型的部署模式：
    - 策略服务器（serve_policy.py）在一台 GPU 机器上运行
    - 机器人控制代码在另一台工控机上运行
    - 两者通过 WebSocket 通信

    【工作流程】
        1. 图像预处理（缩放填充 + 类型转换 + 通道重排）
        2. 通过 WebSocket 发送观测数据到策略服务器
        3. 接收策略预测的动作序列
    """

    def __init__(self, host: str = "localhost", port: int = 8000):
        """初始化 WebSocket 策略客户端。

        Args:
            host: 策略服务器地址（"localhost" 表示同一台机器）
            port: 策略服务器端口（默认 8000，需与服务器一致）

        Raises:
            ImportError: 如果未安装 openpi_client 包
        """
        try:
            from openpi_client import websocket_client_policy, image_tools
            # WebsocketClientPolicy 在构造时会主动连接服务器，
            # 如果连接失败会重试（最多 30 秒）
            self.ws_client = websocket_client_policy.WebsocketClientPolicy(
                host=host,
                port=port
            )
            self.image_tools = image_tools
        except ImportError as e:
            raise ImportError(f"无法导入 openpi_client: {e}")

        self.logger = logging.getLogger("WebsocketPolicy")

    def get_action(self, observation: Dict) -> np.ndarray:
        """通过 WebSocket 远程获取策略动作。

        【图像预处理的完整流水线】
        原始图像 (480x640, HWC)         ← 来自 env.get_obs()
            │
            ▼ resize_with_pad(224, 224) ← 等比例缩放 + 填充到模型输入尺寸
        缩放后图像 (224x224, HWC)
            │
            ▼ convert_to_uint8()        ← 确保像素值类型为 uint8 [0,255]
        uint8 图像 (224x224, HWC)
            │
            ▼ einops.rearrange(HWC→CHW) ← 调整通道顺序，适配模型输入格式
        模型输入 (3x224x224, CHW)
            │
            ▼ ws_client.infer(obs)      ← 发送到策略服务器
        推理结果 (actions)

        Args:
            observation: 原始观测字典（来自 env.get_obs()）

        Returns:
            动作序列，形状为 [action_horizon, action_dim]
            action_horizon 取决于策略配置（如 20 个未来时间步）
        """
        import einops

        # 预处理图像
        obs = observation.copy()
        for cam_name in obs["images"]:
            # 1. 缩放并填充到 224x224（保持宽高比）
            img = self.image_tools.convert_to_uint8(
                self.image_tools.resize_with_pad(obs["images"][cam_name], 224, 224)
            )
            # 2. 通道重排：[H, W, C] → [C, H, W]
            #    π₀ 模型期望的输入格式
            obs["images"][cam_name] = einops.rearrange(img, "h w c -> c h w")

        # 推理：发送到远程服务器，接收动作预测
        result = self.ws_client.infer(obs)
        # 将动作列表堆叠为数组
        actions = np.stack(result['actions'], axis=0)

        return actions


# ============================================================================
# 示例用法
# ============================================================================

if __name__ == "__main__":
    """独立运行此文件的示例。

    这个示例演示了 Tron2Env + WebsocketPolicyWrapper 的完整使用流程：
    配置 → 初始化 → 重置 → 控制循环 → 关闭

    运行方式：
        python examples/tron2/real_env.py
    """
    # ---- 1. 配置 ----
    # 初始关节位置（"挂衣服"任务的预设姿态）
    init_joints = [
        0.026899, 0.2612, -0.02709991, -1.5477003, 0.265, 0.0180999, -0.0614999,
        0.008999, -0.269, 0.02069998, -1.5567001, -0.254, -0.02309972, 0.06469989
    ]

    init_head = [1.0567, -0.0139998]  # 头部初始姿态 [俯仰, 偏航]

    robot_config = Tron2Config(
        robot_ip="10.192.1.2",
        init_joints=init_joints,
        init_head=init_head
    )

    # 摄像头序列号到名称的映射
    # 每个 RealSense 相机有唯一的序列号，通过这个映射告诉
    # MultiCameraManager 哪个序列号的相机对应哪个逻辑名称。
    serial_to_name = {
        'serial_to_name': {
            "245022302696": 'head_camera_image',    # 序列号 → 头部摄像头
            "409122274385": 'left_wrist_image',      # 序列号 → 左腕摄像头
            "230322276915": 'right_wrist_image'      # 序列号 → 右腕摄像头
        }
    }

    env_config = EnvConfig(
        robot_config=robot_config,
        interp_points=6,           # 少一点的插值点使动作响应更快
        time_sync_tolerance=0.01,  # 10ms 的同步容差
        raw_config={'camera': serial_to_name}  # 透传相机配置
    )

    # ---- 2. 初始化环境 ----
    with Tron2Env(env_config) as env:
        # 重置环境（移动到初始位置、初始化夹爪）
        obs = env.reset()
        print(f"✅ 环境重置完成")
        print(f"   状态维度: {obs['state'].shape}")
        print(f"   图像数量: {len(obs['images'])}")

        # ---- 3. 初始化策略 ----
        try:
            policy = WebsocketPolicyWrapper(host='0.0.0.0', port=8000)
            print(f"✅ 策略加载完成")
        except ImportError:
            print("⚠️ 无法加载策略，使用随机动作")
            policy = None

        # ---- 4. 控制循环 ----
        max_steps = 100
        for step in range(max_steps):
            print(f"\n{'='*50}")
            print(f"Step {step+1}/{max_steps}")
            print(f"{'='*50}")

            # 获取观测
            obs = env.get_obs()
            print(f"✅ 获取观测: state={obs['state'][:4]}...")

            # 获取动作并执行
            if policy is not None:
                try:
                    actions = policy.get_action(obs)
                    print(f"✅ 策略推理完成: 动作序列形状 {actions.shape}")

                    # 执行动作序列中的每个动作
                    # 注意：这里的 actions 是一个"动作块"（action chunk），
                    # 包含多个时间步的连续动作（如 action_horizon=20）。
                    # 逐一执行这些动作可以实现开环控制
                    # （在 action_horizon 步内不进行新的推理）。
                    for action in actions:
                        env.step(action)

                except Exception as e:
                    print(f"⚠️ 策略推理失败: {e}")
                    break
            else:
                # 没有策略时使用随机动作进行测试
                # 在初始位置附近添加小扰动，测试机器人响应
                action = obs['state'].copy()
                action[:JointIndex.MOVEJ_DIM] += np.random.randn(JointIndex.MOVEJ_DIM) * 0.01
                env.step(action)

            time.sleep(0.1)  # 控制循环频率，避免过于密集

        print(f"\n✅ 测试完成")

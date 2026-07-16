"""
Tron2 机器人控制模块（Tron2 Robot Control Module）

【模块定位】
这是 Tron2 机器人最底层的硬件通信模块，负责通过 WebSocket 协议与机器人控制器
进行实时通信。上层的 Tron2Env 环境封装依赖于本模块提供的原始控制接口。

【系统架构中的位置】
┌──────────────────────────────────────────────────────────────────┐
│                        pi_client.py (主控脚本)                     │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │                   real_env.py (环境封装)                    │   │
│  │  ┌──────────────────────────────────────────────────────┐  │   │
│  │  │               robot_utils.py (硬件通信)                │  │   │
│  │  │  ┌─────────────┐  ┌──────────┐  ┌──────────────────┐ │  │   │
│  │  │  │ Tron2 类     │  │Tron2Config│  │ 工具类           │ │  │   │
│  │  │  │ (WebSocket) │  │ (配置)    │  │(JointIndex等)    │ │  │   │
│  │  │  └─────────────┘  └──────────┘  └──────────────────┘ │  │   │
│  │  └──────────────────────────────────────────────────────┘  │   │
│  └────────────────────────────────────────────────────────────┘   │
│                           │                                        │
│                           ▼                                        │
│                    Tron2 机器人控制器                               │
│                    (IP: 10.192.1.2:5000)                           │
└──────────────────────────────────────────────────────────────────┘

【通信协议】
本模块通过 WebSocket 与机器人控制器通信，发送 JSON 格式的请求消息。
消息结构：
{
    "accid": "xxx",        // 账户标识（会话 ID）
    "title": "request_xxx", // 请求类型（命令名称）
    "timestamp": 123456789, // 毫秒级时间戳
    "guid": "xxxx-xxxx",   // 唯一请求 ID
    "data": {}             // 请求参数
}

【支持的运动模式】
- MoveJ:  关节空间运动（带插值）—— 点到点运动，适合大范围移动
- ServoJ: 关节空间伺服（无插值）—— 连续实时控制，适合高频闭环
- MoveP:  笛卡尔空间运动（带插值）—— 末端执行器直线运动
- ServoP: 笛卡尔空间伺服（无插值）—— 末端执行器连续控制

【关节空间说明】
Tron2 机器人是双臂构型，共有 18 个自由度：
- 左臂: 7 个旋转关节（肩3+肘2+腕2）
- 左夹爪: 1 个自由度
- 右臂: 7 个旋转关节（肩3+肘2+腕2）
- 右夹爪: 1 个自由度
- 头部: 2 个自由度（俯仰 + 偏航）
"""

import json
import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import time
import math
import numpy as np
import websocket


# ============================================================================
# 频率限制器（Rate Limiter）
# ============================================================================

class RateLimiter:
    """频率限制器 —— 基于绝对时间戳的固定频率控制。

    【为什么需要这个类？】
    在机器人伺服控制中，控制指令的发送频率必须保持稳定。
    如果发送太快，机器人控制器可能过载；
    如果发送太慢，机器人运动会出现卡顿。

    传统的 time.sleep() 无法精确控制频率，因为：
    - sleep() 的精度受操作系统调度影响
    - 控制指令的执行时间会波动（每次耗时不同）
    - 累积延迟会逐渐偏离目标频率

    这个类通过"绝对时间戳"来解决这些问题：
    无论执行耗时多少，下一次执行总是基于"上一次应该执行的时间 + 周期"，
    而不是"上一次实际结束的时间 + 周期"。

    【图解】
    理想情况（频率=100Hz，周期=0.01s）：
    tick: 0.00  0.01  0.02  0.03  0.04  ...
    exec: |─────|─────|─────|─────|─────|→

    实际情况（一次执行耗时 0.003s）：
    tick: 0.00   0.01   0.02   0.03   0.04  ...
    exec: |───xxx|──────|───xxx|──────|───xxx|→
          ↑ 执行耗时  ↑ 只等了 0.007s
          不阻塞总周期

    如果某次执行超时（耗时 > 周期），自动跳过错过的周期，
    从下一个正确的 tick 继续，避免"追赶"导致频率异常。

    Examples:
        >>> limiter = RateLimiter(rate_hz=100.0)
        >>> for i in range(100):
        ...     send_command()
        ...     limiter.sleep()  # 等待到下一个周期
    """

    def __init__(self, rate_hz: float):
        """初始化频率限制器。

        Args:
            rate_hz: 目标频率（Hz）。例如 100 Hz 表示每秒执行 100 次。
        """
        self.rate_hz = rate_hz
        self.period = 1.0 / rate_hz  # 周期 = 1/频率
        self.next_tick = time.monotonic()  # 下一个周期的时间点

    def sleep(self):
        """等待到下一个周期。

        使用 time.monotonic()（单调时钟）而不是 time.time()，
        因为单调时钟不受系统时间调整（如 NTP 同步）的影响。

        算法：
        1. 计算当前时间与下一个 tick 的差值
        2. 如果差值为正（还没到时间），sleep 等待
        3. 更新 next_tick（基于绝对时间增加周期）
        4. 如果 next_tick 已经落后于当前时间（说明执行超时了），
           重置 next_tick 到当前时间 + 周期（跳过错过的周期）
        """
        current_time = time.monotonic()
        sleep_time = self.next_tick - current_time

        if sleep_time > 0:
            time.sleep(sleep_time)

        # 更新下一个 tick（基于绝对时间）
        self.next_tick += self.period

        # 【防累积延迟机制】
        # 如果 next_tick 落后于当前时间，说明执行耗时超过了一个周期。
        # 此时重置 next_tick，从当前时间开始新的周期，
        # 防止为了追赶上而连续快速执行。
        if self.next_tick < time.monotonic():
            self.next_tick = time.monotonic() + self.period

    def reset(self):
        """重置时钟到当前时间。

        通常在切换运动模式（如 MoveJ → ServoJ）时调用，
        因为模式切换后需要重新建立时间基准。
        """
        self.next_tick = time.monotonic()


# ============================================================================
# 关节索引常量
# ============================================================================

class JointIndex:
    """关节索引常量 —— 集中管理所有关节索引的魔数（Magic Number）。

    【设计目的】
    避免在代码中硬编码索引数字（如 hardcode 7、13 等），
    当机器人构型变化时只需修改此处一处。

    【状态数组布局】
    索引:  0   1   2   3   4   5   6  |  7  |  8   9  10  11  12  13  14  | 15  | 16  17
                  左臂(7)             左夹爪          右臂(7)             右夹爪   头部(2)

    STATE_DIM         = 18  (7+1+7+1+2)  完整状态维度
    SERVOJ_DIM        = 16  (14+2)       伺服控制维度（臂 + 头，不含夹爪）
    MOVEJ_DIM         = 14  (7+7)        关节移动维度（仅臂）
    MOVEP_DIM         = 14  (3+4+3+4)    笛卡尔移动维度（左 xyz+wxyz + 右 xyz+wxyz）
    """
    # ---------- 基础维度 ----------
    ARM_DIM = 7             # 单臂关节数
    GRIPPER_DIM = 1         # 单侧夹爪自由度
    HEAD_DIM = 2            # 头部自由度（pitch, yaw）

    # ---------- 复合维度 ----------
    STATE_DIM = ARM_DIM + GRIPPER_DIM + ARM_DIM + GRIPPER_DIM + HEAD_DIM  # = 18
    SERVOJ_DIM = ARM_DIM * 2 + HEAD_DIM  # = 16（14臂 + 2头）
    MOVEJ_DIM = ARM_DIM * 2  # = 14（仅臂关节）
    MOVEP_DIM = 14  # 7（左 xyz+wxyz）+ 7（右 xyz+wxyz）
    SERVOP_DIM = (7, 7)  # 左右各自 7 维（xyz+wxyz）

    # ---------- 左臂关节（索引 0-6） ----------
    LEFT_ARM_START = 0
    LEFT_ARM_END = ARM_DIM  # = 7
    LEFT_ARM = slice(LEFT_ARM_START, LEFT_ARM_END)  # slice(0, 7)

    # ---------- 左夹爪（索引 7） ----------
    LEFT_GRIPPER = LEFT_ARM_END  # = 7

    # ---------- 右臂关节（索引 8-14） ----------
    RIGHT_ARM_START = LEFT_GRIPPER + GRIPPER_DIM  # = 8
    RIGHT_ARM_END = RIGHT_ARM_START + ARM_DIM     # = 15
    RIGHT_ARM = slice(RIGHT_ARM_START, RIGHT_ARM_END)  # slice(8, 15)

    # ---------- 右夹爪（索引 15） ----------
    RIGHT_GRIPPER = RIGHT_ARM_END  # = 15

    # ---------- 头部关节（索引 16-17） ----------
    HEAD_START = RIGHT_GRIPPER + GRIPPER_DIM  # = 16
    HEAD_END = HEAD_START + HEAD_DIM          # = 18
    HEAD = slice(HEAD_START, HEAD_END)        # slice(16, 18)
    # 头部子维度
    HEAD_PITCH = HEAD_START        # = 16（俯仰：点头）
    HEAD_YAW = HEAD_START + 1      # = 17（偏航：摇头）

    # ---------- 兼容性别名 ----------
    STATE_DIM_WITHOUT_HEAD = STATE_DIM - HEAD_DIM  # = 16
    STATE_DIM_WITH_HEAD = STATE_DIM                 # = 18
    ARM_JOINT_DIM = ARM_DIM          # = 7
    TOTAL_ARM_DIM = ARM_DIM * 2      # = 14


# ============================================================================
# 配置数据结构
# ============================================================================

@dataclass
class Tron2Config:
    """Tron2 机器人配置。

    集中管理所有与机器人硬件相关的参数，包括网络连接、初始姿态、
    PID 控制参数等。

    【PID 参数说明】
    ServoKp（比例增益）和 ServoKd（微分增益）是伺服控制器的参数。
    - Kp 越大，机器人对位置偏差的响应越强（但过大会引起震荡）
    - Kd 越大，机器人的运动越"阻尼"（可以抑制震荡，但过大会让运动迟钝）
    - 大关节（肩部）Kp/Kd 较大，小关节（腕部）Kp/Kd 较小
    - 头部 Kp/Kd 最小（头部惯性小，不需要强控制力）
    """
    # ---- 网络连接 ----
    robot_ip: str = "10.192.1.2"  # 机器人控制器的 IP 地址
    port: int = 5000               # WebSocket 服务端口

    # ---- 初始位置 ----
    init_joints: Optional[List[float]] = None  # 14 维：左臂7 + 右臂7
    init_head: Optional[List[float]] = None    # 2 维：[俯仰, 偏航]

    # ---- 状态轮询 ----
    state_queue_maxlen: int = 7    # 状态队列的最大长度
    polling_rate: float = 200.0    # 状态轮询频率（Hz）
                                   # 200 Hz = 每 5 毫秒查询一次
                                   # 高频轮询确保状态数据的实时性

    # ---- 连接控制 ----
    connection_timeout: float = 5.0  # 连接超时（秒）

    # ---- 状态选择 ----
    include_head_state: bool = True  # 状态数据中是否包含头部关节

    # ---- Servo 模式 PID 参数 ----
    # 格式：16 维 [左臂7kp, 右臂7kp, 头部2kp]
    # 每个关节的 PID 参数不同，取决于关节的负载和动力学特性
    servo_kp: List[float] = field(default_factory=lambda: [
        420, 420, 300, 300, 200, 200, 200,  # 左臂（肩→肘→腕，依次递减）
        420, 420, 300, 300, 200, 200, 200,  # 右臂（对称）
        60, 60                                # 头部
    ])

    servo_kd: List[float] = field(default_factory=lambda: [
        12, 12, 15, 15, 10, 10, 10,  # 左臂
        12, 12, 15, 15, 10, 10, 10,  # 右臂
        3, 3                          # 头部
    ])

    def __post_init__(self):
        """初始化后验证配置参数。

        dataclass 的 __post_init__ 在 __init__ 完成后自动调用。
        """
        if self.init_joints is not None and len(self.init_joints) != JointIndex.MOVEJ_DIM:
            raise ValueError(
                f"init_joints 应有 {JointIndex.MOVEJ_DIM} 个元素，实际 {len(self.init_joints)}"
            )

        if self.init_head is not None and len(self.init_head) != JointIndex.HEAD_DIM:
            raise ValueError(
                f"init_head 应有 {JointIndex.HEAD_DIM} 个元素，实际 {len(self.init_head)}"
            )


# ============================================================================
# 枚举类型
# ============================================================================

class MotionMode(Enum):
    """运动控制模式枚举。

    四种运动模式的区别：
    ┌─────────┬─────────────┬────────────┬─────────────────────────┐
    │ 模式    │ 空间        │ 插值       │ 用途                    │
    ├─────────┼─────────────┼────────────┼─────────────────────────┤
    │ MoveJ   │ 关节空间    │ ✅ 有插值   │ 点到点移动（初始化归位） │
    │ ServoJ  │ 关节空间    │ ❌ 无插值   │ 连续控制（策略推理循环） │
    │ MoveP   │ 笛卡尔空间  │ ✅ 有插值   │ 直线运动（抓取放置）    │
    │ ServoP  │ 笛卡尔空间  │ ❌ 无插值   │ 连续末端控制（跟踪）    │
    └─────────┴─────────────┴────────────┴─────────────────────────┘

    【关节空间 vs 笛卡尔空间】
    - 关节空间（Joint Space）：指定每个关节的目标角度
    - 笛卡尔空间（Cartesian Space）：指定末端执行器的目标位置和姿态（x, y, z, 四元数）

    【插值 vs 无插值】
    - 带插值（Move）：机器人控制器内部自动规划平滑轨迹，适合大范围运动
    - 无插值（Servo）：直接发送目标位置，不规划轨迹，适合高频闭环控制
    """
    MOVEJ = "movej"     # 关节空间带插值
    SERVOJ = "servoj"   # 关节空间无插值（实时伺服）
    MOVEP = "movep"     # 笛卡尔空间带插值
    SERVOP = "servop"   # 笛卡尔空间无插值（实时伺服）


# ============================================================================
# 异常定义
# ============================================================================

class Tron2Error(Exception):
    """Tron2 基础异常 —— 所有 Tron2 相关异常的基类。"""
    pass


class ConnectionError(Tron2Error):
    """连接异常 —— WebSocket 连接失败或意外断开。"""
    pass


class CommandError(Tron2Error):
    """命令执行异常 —— 发送指令时参数错误或通信失败。"""
    pass


class StateError(Tron2Error):
    """状态获取异常 —— 获取关节状态时超时或数据异常。"""
    pass


# ============================================================================
# 主机器人控制器
# ============================================================================

class Tron2:
    """Tron2 机器人控制类 —— 最底层的硬件通信接口。

    【设计职责】
    1. 通过 WebSocket 与机器人控制器建立双向通信
    2. 发送运动指令（MoveJ/ServoJ/MoveP/ServoP）
    3. 异步接收并缓存机器人的实时状态反馈
    4. 管理多种运动模式的切换

    【并发模型】
    主线程（用户代码）              WebSocket 线程              轮询线程
        │                              │                          │
        │───── servoj() ──────────────→│                          │
        │                              │──── 发送指令 ──────────→│
        │                              │                          │
        │                              │←──── 状态回传 ──────────│
        │                              │                          │
        │←── get_joint_states() ─────│←── 存入队列              │
        │    (从队列读取最新状态)      │                          │

    【状态数据流】
    机器人控制器 → WebSocket → _on_message() → _handle_joint_state()
        → 原子更新 joint_states → _try_commit_state() → joint_state_queue
        → get_joint_states() → 用户代码

    Examples:
        >>> config = Tron2Config(robot_ip="10.192.1.2")
        >>> robot = Tron2(config)
        >>> robot.set_movej_mode()
        >>> robot.movej([0.0]*14, move_time=2.0)
        >>> states = robot.get_joint_states()
        >>> robot.disconnect()
    """

    def __init__(self, config: Optional[Tron2Config] = None):
        """初始化 Tron2 机器人控制器。

        初始化流程（按顺序）：
        1. 保存配置并设置日志
        2. 初始化 WebSocket 连接
        3. 初始化状态缓冲区
        4. 启动后台状态轮询线程
        5. 移动到初始位置

        Args:
            config: 机器人配置对象。如果为 None，使用默认配置。
                    注意：即使使用默认配置，也需要修改 robot_ip 才能连接。
        """
        self.config = config or Tron2Config()
        self.time_recoder = time.time()

        # ---- 日志系统 ----
        self._setup_logger()

        # ---- WebSocket 连接状态 ----
        self.accid: Optional[str] = None          # 会话标识
        self.ws_client: Optional[websocket.WebSocketApp] = None  # WebSocket 客户端
        self.ws_thread: Optional[threading.Thread] = None        # WebSocket 后台线程
        self.connected = False        # 连接状态标志
        self.should_exit = False      # 退出标志（用于控制轮询线程退出）

        # ---- 运动模式 ----
        self.motion_mode: Optional[MotionMode] = None

        # ---- 状态缓冲区初始化 ----
        self._init_state_buffers()

        # ---- ServoJ 模式参数 ----
        self.servoj_joint_num = JointIndex.SERVOJ_DIM  # = 16
        self.servoj_args = self._init_servoj_args()

        # ---- 频率限制器 ----
        # ServoJ 和 ServoP 都需要稳定的控制频率
        self.servoj_rate_limiter = RateLimiter(rate_hz=100.0)
        self.servop_rate_limiter = RateLimiter(rate_hz=100.0)

        # ---- 建立连接 ----
        self._connect()
        time.sleep(1)  # 等待连接稳定

        # ---- 启动状态轮询 ----
        self._start_polling_threads()

        # ---- 移动到初始位置 ----
        if self.config.init_joints is not None or self.config.init_head is not None:
            self._move_to_init_pose()

    def _setup_logger(self):
        """设置日志系统。

        日志格式：[时间戳] [模块名] [级别] 消息
        时间戳精确到毫秒，对于调试机器人控制的时序问题非常关键。
        """
        self.logger = logging.getLogger(f"Tron2-{self.config.robot_ip}")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(asctime)s.%(msecs)03d] [%(name)s] [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG)

    # =======================================================================
    # 状态缓冲区初始化
    # =======================================================================

    def _init_state_buffers(self):
        """初始化状态缓冲区。

        【为什么需要缓冲区？】
        机器人的状态数据（关节角度、末端位姿）是通过 WebSocket 异步推送的。
        为了不丢失数据并确保主线程和 WebSocket 线程之间的线程安全，
        我们使用：
        - 一个"当前状态"字典（由 _state_lock 保护）
        - 一个"历史队列"（由 _queue_lock 保护）
        每当完整的状态数据到达时，原子性地提交到队列。

        【两个锁的设计意图】
        - _state_lock: 保护关节数据 + 夹爪数据的"半更新"状态
        - _queue_lock: 保护队列的读写并发
        这种"两段锁"设计允许频繁的关节更新和较少的队列操作并存。
        """
        # ---- 当前关节状态 ----
        self.joint_states = {
            'timestamp': -1,  # 硬件时间戳（毫秒）
            'states': [-1.0] * JointIndex.STATE_DIM_WITH_HEAD,  # 18 维状态向量
            'joint_updated': False,   # 关节数据是否已更新（用于原子提交判断）
            'gripper_updated': False  # 夹爪数据是否已更新
        }

        # ---- 末端执行器位姿 ----
        self.ee_pose_states = {
            'timestamp': -1,
            "left_position": [-1.0, -1.0, -1.0],       # 左臂末端位置 (x, y, z)
            "left_quat": [-1.0, -1.0, -1.0, -1.0],     # 左臂末端姿态 (四元数 w, x, y, z)
            "right_position": [-1.0, -1.0, -1.0],      # 右臂末端位置 (x, y, z)
            "right_quat": [-1.0, -1.0, -1.0, -1.0]     # 右臂末端姿态 (四元数 w, x, y, z)
        }

        # 当前状态聚合模式（用于日志调试）
        self.states_mode = None

        # ---- 历史队列（线程安全） ----
        # 使用 deque（双端队列）实现环形缓冲区
        # maxlen 限制了最大缓存量，防止内存无限增长
        self.joint_state_queue = deque(maxlen=self.config.state_queue_maxlen)
        self.ee_pose_queue = deque(maxlen=self.config.state_queue_maxlen)

        # 线程锁
        self._queue_lock = threading.Lock()   # 保护队列的并发访问
        self._state_lock = threading.Lock()   # 保护 joint_states 的原子性

    def _init_servoj_args(self) -> Dict:
        """初始化 ServoJ 模式参数。

        ServoJ 模式下，机器人控制器使用这些参数进行实时轨迹跟踪。
        参数含义：
        - v: 速度前馈（0 表示不启用速度前馈）
        - kp: 比例增益（位置误差 → 力矩的比例系数）
        - kd: 微分增益（速度误差 → 力矩的比例系数）
        - tau: 力矩前馈（0 表示不需要额外力矩补偿）
        - mode: 控制模式（0 表示标准位置控制）
        - na: 关节数量

        Returns:
            dict: ServoJ 参数字典
        """
        return {
            "v": [0.0] * self.servoj_joint_num,                                  # 速度
            "kp": self.config.servo_kp[:self.servoj_joint_num],                  # 比例增益
            "kd": self.config.servo_kd[:self.servoj_joint_num],                  # 微分增益
            "tau": [0.0] * self.servoj_joint_num,                                 # 力矩
            "mode": [0.0] * self.servoj_joint_num,                                # 模式
            "na": self.servoj_joint_num                                           # 数量
        }

    # =======================================================================
    # WebSocket 连接管理
    # =======================================================================

    def _generate_guid(self) -> str:
        """生成唯一的请求标识符（GUID）。

        每次 WebSocket 请求都需要一个唯一的 GUID，
        用于将响应与请求匹配（请求-响应模式）。
        使用 UUID4（随机 UUID），冲突概率极低。

        Returns:
            str: 格式为 "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx" 的 36 字符字符串
        """
        return str(uuid.uuid4())

    def _send_request(self, title: str, data: Optional[Dict] = None) -> bool:
        """通过 WebSocket 发送请求到机器人控制器。

        【消息协议】
        所有命令都封装为统一的 JSON 消息格式：
        {
            "accid": "会话ID",
            "title": "命令名称",
            "timestamp": 毫秒级时间戳,
            "guid": "唯一请求ID",
            "data": {命令特定参数}
        }

        Args:
            title: 请求标题/命令名称。
                   例如："request_movej"、"request_servoj"、"request_get_joint_state"
            data: 请求数据。不同命令需要不同字段。

        Returns:
            bool: 是否发送成功。False 表示 WebSocket 未连接。
        """
        if data is None:
            data = {}

        if not self.ws_client or not self.connected:
            self.logger.warning(f"WebSocket 未连接，无法发送: {title}")
            return False

        try:
            message = {
                "accid": self.accid,                  # 会话标识
                "title": title,                       # 命令名称
                "timestamp": int(time.time() * 1000), # 当前时间戳（毫秒）
                "guid": self._generate_guid(),        # 唯一标识
                "data": data                          # 命令参数
            }

            message_str = json.dumps(message)  # 序列化为 JSON 字符串
            self.ws_client.send(message_str)   # 通过 WebSocket 发送
            return True

        except Exception as e:
            self.logger.error(f"发送请求失败 ({title}): {e}")
            return False

    def _on_open(self, ws):
        """WebSocket 连接建立成功的回调函数。

        当 TCP 连接建立并且 WebSocket 握手完成后自动调用。

        Args:
            ws: WebSocketApp 实例
        """
        self.logger.info(f"机器人连接成功: {self.config.robot_ip}:{self.config.port}")
        self.connected = True

    def _on_message(self, ws, message: str):
        """WebSocket 接收到消息的回调函数。

        所有来自机器人控制器的响应都通过这个函数处理。
        它是一个"消息分发器"，根据 title 字段将消息路由到不同的处理函数。

        【消息路由】
        "response_get_joint_state"       → _handle_joint_state()
        "response_get_limx_2fclaw_state" → _handle_gripper_state()
        "response_get_move_pose"         → _handle_ee_pose()
        其他消息                         → 记录日志（调试）

        Args:
            ws: WebSocketApp 实例
            message: 接收到的原始 JSON 字符串
        """
        try:
            root = json.loads(message)  # 反序列化 JSON
            title = root.get("title", "")
            self.accid = root.get("accid", self.accid)  # 更新会话标识

            # 根据消息类型分发处理
            if title == "response_get_joint_state":
                self._handle_joint_state(root)
            elif title == "response_get_limx_2fclaw_state":
                self._handle_gripper_state(root)
            elif title == "response_get_move_pose":
                self._handle_ee_pose(root)
            elif title not in [
                "notify_robot_info",
                "response_servoj",
                "response_set_limx_2fclaw_cmd"
            ]:
                # 忽略高频的伺服响应和夹爪响应，只打印其他消息
                self.logger.debug(f"收到消息: {title}")

        except json.JSONDecodeError:
            self.logger.error(f"无法解析消息: {message}")
        except Exception as e:
            self.logger.error(f"处理消息异常: {e}")

    def _handle_joint_state(self, root: Dict):
        """处理关节状态消息（原子更新）。

        【数据格式】
        机器人控制器返回的关节状态数据格式：
        {
            "data": {
                "q": [j1, j2, ..., j16]  # 16 维关节角度（7左 + 7右 + 2头）
            },
            "timestamp": 123456789  # 硬件时间戳（毫秒）
        }

        这个函数将 16 维数据拆解为 18 维状态向量：
        索引 0-6:   左臂关节
        索引 7:     左夹爪（由 _handle_gripper_state 填充）
        索引 8-14:  右臂关节
        索引 15:    右夹爪（由 _handle_gripper_state 填充）
        索引 16-17: 头部关节

        【原子更新策略】
        关节数据和夹爪数据来自两个不同的 WebSocket 消息。
        为了确保状态的一致性，我们使用"两阶段提交"策略：
        1. 收到关节数据 → 更新关节部分，标记 joint_updated=True
        2. 收到夹爪数据 → 更新夹爪部分，标记 gripper_updated=True
        3. 两者都更新后 → 提交完整状态到队列

        Args:
            root: 解析后的 JSON 数据字典
        """
        self.states_mode = 'joint'
        states = root.get("data", {})
        joint_q = states.get("q", [])  # 16 维数组
        joint_timestamp = root.get("timestamp", -1)

        with self._state_lock:
            self.joint_states["timestamp"] = joint_timestamp

            # 左臂关节 [0:7]
            self.joint_states["states"][JointIndex.LEFT_ARM] = \
                joint_q[:JointIndex.ARM_JOINT_DIM]

            # 右臂关节 [8:15]（在 16 维反馈中索引 7-13）
            self.joint_states["states"][JointIndex.RIGHT_ARM] = \
                joint_q[JointIndex.ARM_JOINT_DIM:JointIndex.TOTAL_ARM_DIM]

            # 头部关节 [16:18]（在 16 维反馈中索引 14-15）
            self.joint_states["states"][JointIndex.HEAD_PITCH] = joint_q[14]
            self.joint_states["states"][JointIndex.HEAD_YAW] = joint_q[15]

            temp = self.joint_states["states"][JointIndex.RIGHT_ARM]

            # 标记关节数据已更新
            self.joint_states["joint_updated"] = True
            # 尝试提交完整状态（如果夹爪数据也已到达）
            self._try_commit_state()

    def _handle_gripper_state(self, root: Dict):
        """处理夹爪状态消息（原子更新）。

        【数据格式】
        {
            "data": {
                "left_opening": 50,    # 左夹爪开口度 [0-100]
                "right_opening": 50    # 右夹爪开口度 [0-100]
            }
        }

        【归一化】
        机器人控制器返回的夹爪开口度是 [0, 100] 的百分比值，
        这里将其转换为 [0, 1] 的归一化值，方便与 π₀ 模型的输出对齐。

        Args:
            root: 解析后的 JSON 数据字典
        """
        self.states_mode = 'gripper'
        claw_data = root.get("data", {})

        with self._state_lock:
            # 将 [0, 100] → [0, 1] 归一化
            self.joint_states["states"][JointIndex.LEFT_GRIPPER] = \
                claw_data.get("left_opening", -1) / 100.0
            self.joint_states["states"][JointIndex.RIGHT_GRIPPER] = \
                claw_data.get("right_opening", -1) / 100.0

            # 标记夹爪数据已更新
            self.joint_states["gripper_updated"] = True
            # 尝试提交完整状态
            self._try_commit_state()

        self.states_mode = None

    def _try_commit_state(self):
        """尝试将完整状态提交到历史队列。

        【为什么需要这个函数？】
        关节数据和夹爪数据来自不同的 WebSocket 消息，到达时间不同。
        如果在关节数据到达时直接提交，夹爪数据还是旧值；
        如果在夹爪数据到达时直接提交，关节数据还是旧值。
        所以需要等待两者都更新后，才提交"完整"的状态快照。

        这个函数需要在 _state_lock 的保护下调用（由调用者保证）。

        提交条件：
        1. 关节数据已更新（joint_updated == True）
        2. 夹爪数据已更新（gripper_updated == True）
        3. 状态数据有效（第一个关节值 != -1）
        4. 时间戳有效（timestamp != -1）
        """
        # 检查四个条件是否都满足
        if (self.joint_states["joint_updated"] and
            self.joint_states["gripper_updated"] and
            self.joint_states["states"][JointIndex.LEFT_ARM_START] != -1 and
            self.joint_states["timestamp"] != -1):

            # 提交到线程安全的队列
            with self._queue_lock:
                self.joint_state_queue.append(self.joint_states.copy())

            # 重置更新标志（等待下一轮更新）
            self.joint_states["joint_updated"] = False
            self.joint_states["gripper_updated"] = False

    def _handle_ee_pose(self, root: Dict):
        """处理末端执行器位姿消息。

        【数据格式】
        {
            "data": {
                "left_position": [x, y, z],     # 左臂末端位置（米）
                "left_quat": [w, x, y, z],      # 左臂末端姿态（四元数）
                "right_position": [x, y, z],    # 右臂末端位置（米）
                "right_quat": [w, x, y, z]      # 右臂末端姿态（四元数）
            },
            "timestamp": 123456789
        }

        Args:
            root: 解析后的 JSON 数据字典
        """
        self.states_mode = 'ee_pose'
        ee_pose_data = root.get("data", {})
        ee_pose_timestamp = root.get("timestamp", -1)

        self.ee_pose_states["timestamp"] = ee_pose_timestamp
        self.ee_pose_states["left_position"] = ee_pose_data.get("left_position", [-1, -1, -1])
        self.ee_pose_states["left_quat"] = ee_pose_data.get("left_quat", [-1, -1, -1, -1])
        self.ee_pose_states["right_position"] = ee_pose_data.get("right_position", [-1, -1, -1])
        self.ee_pose_states["right_quat"] = ee_pose_data.get("right_quat", [-1, -1, -1, -1])

        temp = self.ee_pose_states["right_quat"]

        with self._queue_lock:
            self.ee_pose_queue.append(self.ee_pose_states.copy())

    def _on_close(self, ws, close_status_code, close_msg):
        """WebSocket 连接关闭的回调函数。

        Args:
            ws: WebSocketApp 实例
            close_status_code: 关闭状态码
            close_msg: 关闭原因描述
        """
        self.logger.warning(f"机器人连接已关闭: {close_status_code} - {close_msg}")
        self.connected = False

    def _on_error(self, ws, error):
        """WebSocket 错误的回调函数。

        Args:
            ws: WebSocketApp 实例
            error: 错误对象
        """
        self.logger.error(f"WebSocket 错误: {error}")

    def _connect(self):
        """建立与机器人控制器的 WebSocket 连接。

        连接流程：
        1. 构造 WebSocket URL（ws://{ip}:{port}）
        2. 创建 WebSocketApp 实例，注册回调函数
        3. 在后台线程中启动 run_forever()（事件循环）
        4. run_forever() 会自动处理重连（如有配置）

        【为什么用后台线程？】
        websocket-client 库的 run_forever() 是一个阻塞调用，
        它会持续监听 WebSocket 消息直到连接关闭。
        如果放在主线程中，会阻塞后续代码的执行。
        所以放在后台线程中运行。
        """
        ws_url = f"ws://{self.config.robot_ip}:{self.config.port}"
        self.logger.info(f"正在连接机器人: {ws_url}")

        self.ws_client = websocket.WebSocketApp(
            ws_url,
            on_open=self._on_open,          # 连接成功回调
            on_message=self._on_message,    # 消息接收回调
            on_close=self._on_close,        # 连接关闭回调
            on_error=self._on_error         # 错误回调
        )

        # 在后台线程中运行 WebSocket 事件循环
        self.ws_thread = threading.Thread(target=self._run_websocket, daemon=True)
        self.ws_thread.start()

    def _run_websocket(self):
        """运行 WebSocket 客户端事件循环。

        这是一个阻塞调用，会持续运行直到 WebSocket 连接关闭。
        在后台线程中执行。
        """
        try:
            self.ws_client.run_forever()
        except Exception as e:
            self.logger.error(f"WebSocket 运行异常: {e}")

    # =======================================================================
    # 状态轮询
    # =======================================================================

    def _start_polling_threads(self):
        """启动后台状态轮询线程。

        轮询线程以固定的频率（默认 200 Hz）向机器人控制器请求三种状态信息：
        1. 关节角度（get_joint_state）
        2. 夹爪开度（get_limx_2fclaw_state）
        3. 末端位姿（get_move_pose）

        【为什么需要轮询？】
        机器人控制器不会主动推送状态更新。
        需要周期性发送请求来获取最新的状态数据。
        轮询频率越高，状态数据越实时，但也会增加通信负载。
        200 Hz（5 毫秒一次）对于机器人控制来说是一个合理的选择。
        """
        self.joint_polling_thread = threading.Thread(
            target=self._poll_feedback,
            daemon=True
        )
        self.joint_polling_thread.start()
        self.logger.info(f"状态轮询已启动 ({self.config.polling_rate} Hz)")

    def _poll_feedback(self):
        """轮询机器人状态的循环函数。

        每次循环发送三个请求：
        1. request_get_joint_state: 获取关节角度
        2. request_get_limx_2fclaw_state: 获取夹爪开度
        3. request_get_move_pose: 获取末端位姿

        控制轮询频率的方式是通过 sleep_time 来调整。
        注意这里用 max(0, ...) 确保不会因为执行超时而 sleep 负数。

        【为什么分开发送三个请求？】
        虽然理论上可以把三个状态合并为一个请求，
        但机器人控制器的 API 是分开设计的。
        三个请求各自返回独立的数据。
        """
        sleep_time = 1.0 / self.config.polling_rate  # 200 Hz → 0.005s

        while not self.should_exit:
            start_time = time.time()

            # 并行请求三种状态
            self._send_request("request_get_joint_state")         # 关节角度
            self._send_request("request_get_limx_2fclaw_state")   # 夹爪开度
            self._send_request("request_get_move_pose")           # 末端位姿

            # 控制轮询频率
            elapsed = time.time() - start_time
            time.sleep(max(0, sleep_time - elapsed))

    # =======================================================================
    # 状态获取接口
    # =======================================================================

    def get_joint_states(self, timeout: float = 1.0) -> Dict:
        """获取当前关节状态。

        从 joint_state_queue 中取出最近提交的完整状态快照。
        如果队列为空（还没有接收到完整的状态），则等待直到超时。

        Args:
            timeout: 超时时间（秒）。如果在此时间内没有获取到状态，抛出异常。

        Returns:
            包含关节状态的字典:
            {
                'timestamp': int,        # 硬件时间戳（毫秒）
                'states': List[float],   # 18 维状态向量
                'joint_updated': bool,   # 内部标记（历史数据）
                'gripper_updated': bool  # 内部标记（历史数据）
            }

        Raises:
            StateError: 超时未获取到状态
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            with self._queue_lock:
                if self.joint_state_queue:
                    # pop() 从队列尾部取出最新的数据（LIFO）
                    # 使用 LIFO（后进先出）而不是 FIFO（先进先出）是因为
                    # 我们总是想要最新的状态，而不是最旧的。
                    states = self.joint_state_queue.pop()

                    # 验证状态维度
                    expected_dim = JointIndex.STATE_DIM_WITH_HEAD  # = 18
                    if len(states['states']) != expected_dim:
                        raise StateError(
                            f"状态维度错误: 期望 {expected_dim}, "
                            f"实际 {len(states['states'])}"
                        )

                    return states

            time.sleep(0.001)  # 1 毫秒轮询间隔，避免忙等待

        raise StateError(f"获取关节状态超时 ({timeout}s)")

    def get_ee_poses(self, timeout: float = 1.0) -> Dict:
        """获取当前末端执行器位姿。

        Args:
            timeout: 超时时间（秒）

        Returns:
            包含末端位姿的字典:
            {
                'timestamp': int,
                'left_position': [x, y, z],
                'left_quat': [w, x, y, z],
                'right_position': [x, y, z],
                'right_quat': [w, x, y, z]
            }

        Raises:
            StateError: 超时未获取到状态
        """
        if self.motion_mode != MotionMode.MOVEP:
            self.set_movep_mode()  # 获取位姿前需要切换到 MoveP 模式

        start_time = time.time()

        while time.time() - start_time < timeout:
            with self._queue_lock:
                if self.ee_pose_queue:
                    return self.ee_pose_queue.popleft()  # FIFO

            time.sleep(0.001)

        raise StateError(f"获取末端位姿超时 ({timeout}s)")

    # =======================================================================
    # 运动控制接口 —— 关节空间
    # =======================================================================

    def movej(self, joint_positions: Union[List[float], np.ndarray], move_time: float = 2.0):
        """关节空间运动（带插值） —— 移动到目标关节角度。

        MoveJ 是"点到点"运动模式，机器人控制器会自动规划平滑的轨迹。
        适合大范围移动（如初始化归位、大姿态调整），
        但不适合高频连续控制。

        Args:
            joint_positions: 14 维目标关节角度 [左臂7, 右臂7]
            move_time: 运动持续时间（秒）。越大运动越慢越平滑。

        Raises:
            CommandError: 参数维度错误或发送失败
        """
        if isinstance(joint_positions, np.ndarray):
            joint_positions = joint_positions.tolist()

        if len(joint_positions) != JointIndex.MOVEJ_DIM:
            raise CommandError(
                f"关节角度列表长度应为 {JointIndex.MOVEJ_DIM}, "
                f"实际 {len(joint_positions)}"
            )

        # 确保处于 MoveJ 模式
        if self.motion_mode != MotionMode.MOVEJ:
            self.set_movej_mode()

        data = {
            "joint": joint_positions,
            "time": move_time
        }

        if not self._send_request("request_movej", data):
            raise CommandError("MoveJ 命令发送失败")

        self.logger.debug(f"MoveJ 命令已发送: time={move_time}s")

    def servoj(
        self,
        joint_positions: Union[List[float], np.ndarray],
    ):
        """关节空间伺服控制（无插值，高频） —— 实时发送目标关节角度。

        ServoJ 是"实时跟踪"模式，直接发送目标位置到控制器。
        适合高频闭环控制（如策略推理循环的执行阶段），
        但不提供轨迹平滑（需要上层自己处理插值）。

        【与 MoveJ 的区别】
        MoveJ: 一次发送目标，机器人控制器自动插值，阻塞直到到达
        ServoJ: 持续发送目标，控制器实时跟踪，不阻塞

        Args:
            joint_positions: 16 维关节角度 [左臂7, 左夹爪1, 右臂7, 右夹爪1, 头2]

        Raises:
            CommandError: 参数维度错误或发送失败
        """
        if isinstance(joint_positions, np.ndarray):
            joint_positions = joint_positions.tolist()

        if len(joint_positions) != self.servoj_joint_num:
            raise CommandError(
                f"关节角度列表长度应为 {self.servoj_joint_num}, "
                f"实际 {len(joint_positions)}"
            )

        servo_data = {
            "q": joint_positions,          # 目标关节角度（16 维）
            "filter_ratio": 1.0             # 滤波比例（1.0 = 不滤波，最直接）
        }

        if not self._send_request("request_servoj", servo_data):
            raise CommandError("ServoJ 命令发送失败")

        # 【频率控制】
        # 每次发送完 ServoJ 指令后，限速到 100 Hz
        # 确保不会发送过快导致机器人控制器过载
        self.servoj_rate_limiter.sleep()

    # =======================================================================
    # 运动控制接口 —— 笛卡尔空间
    # =======================================================================

    def movep(self, pose_quat_list: Union[List[float], np.ndarray], move_time: float = 5.0):
        """笛卡尔空间运动（带插值） —— 移动末端到目标位姿。

        MoveP 模式下，机器人末端执行器沿直线轨迹运动到目标位姿。
        适合需要保持末端姿态的任务（如抓取、放置）。

        【数据格式】
        位姿用 14 维向量表示：
        [left_x, left_y, left_z, left_w, left_qx, left_qy, left_qz,
         right_x, right_y, right_z, right_w, right_qx, right_qy, right_qz]

        其中位置用 xyz 表示，姿态用四元数 (w, x, y, z) 表示。

        Args:
            pose_quat_list: 14 维目标位姿
            move_time: 运动时间（秒）

        Raises:
            CommandError: 参数维度错误或发送失败
        """
        if isinstance(pose_quat_list, np.ndarray):
            pose_quat_list = pose_quat_list.tolist()

        if len(pose_quat_list) != JointIndex.MOVEP_DIM:
            raise CommandError(
                f"位姿列表长度应为 {JointIndex.MOVEP_DIM}, "
                f"实际 {len(pose_quat_list)}"
            )

        if self.motion_mode != MotionMode.MOVEP:
            self.set_movep_mode()

        data = {
            "pos": pose_quat_list,
            "time": move_time
        }

        if not self._send_request("request_movep", data):
            raise CommandError("MoveP 命令发送失败")

        self.logger.debug(f"MoveP 命令已发送: time={move_time}s")

    def servop(
        self,
        left_pose: Union[List[float], np.ndarray],
        right_pose: Union[List[float], np.ndarray],
        move_time: float = 5.0
    ):
        """笛卡尔空间伺服控制（无插值） —— 实时发送目标位姿。

        ServoP 是笛卡尔空间的实时控制模式，适合需要连续调整末端位置的场景。
        与 ServoJ 类似，不提供轨迹平滑。

        Args:
            left_pose: 左臂末端位姿 [xyz(3), wxyz(4)] = 7 维
            right_pose: 右臂末端位姿 [xyz(3), wxyz(4)] = 7 维
            move_time: 运动时间（秒）

        Raises:
            CommandError: 参数维度错误或发送失败
        """
        if isinstance(left_pose, np.ndarray):
            left_pose = left_pose.tolist()
        if isinstance(right_pose, np.ndarray):
            right_pose = right_pose.tolist()

        if len(left_pose) != JointIndex.SERVOP_DIM[0]:
            raise CommandError(
                f"左臂位姿长度应为 {JointIndex.SERVOP_DIM[0]}, "
                f"实际 {len(left_pose)}"
            )
        if len(right_pose) != JointIndex.SERVOP_DIM[1]:
            raise CommandError(
                f"右臂位姿长度应为 {JointIndex.SERVOP_DIM[1]}, "
                f"实际 {len(right_pose)}"
            )

        if self.motion_mode != MotionMode.SERVOP:
            self.set_servop_mode()

        data = {
            "left_pos": left_pose,
            "right_pos": right_pose,
            "time": move_time
        }

        if not self._send_request("request_servop", data):
            raise CommandError("ServoP 命令发送失败")

        # 限速控制
        self.servop_rate_limiter.sleep()

    # =======================================================================
    # 夹爪和头部控制
    # =======================================================================

    def set_gripper(
        self,
        left_opening: float = 0.0,
        right_opening: float = 0.0,
        left_speed: float = 100.0,
        left_force: float = 50.0,
        right_speed: float = 100.0,
        right_force: float = 50.0
    ):
        """设置夹爪开度、速度和力度。

        【夹爪控制参数说明】
        夹爪控制器有三个可调参数：
        - opening: 开口度 [0-100]，0=完全闭合，100=完全张开
        - speed: 运动速度 [0-100]，越大夹爪动作越快
        - force: 夹持力度 [0-100]，越大夹爪抓得越紧

        【夹爪控制 vs 关节控制】
        夹爪有独立的控制器和通信命令，不通过 ServoJ/MoveJ 控制。
        这是物理上独立的硬件（夹爪模块）和手臂关节模块分开控制。

        Args:
            left_opening: 左夹爪开口度 [0-100]
            left_speed: 左夹爪速度 [0-100]
            left_force: 左夹爪力度 [0-100]
            right_opening: 右夹爪开口度 [0-100]
            right_speed: 右夹爪速度 [0-100]
            right_force: 右夹爪力度 [0-100]
        """
        data = {
            "left_opening": int(np.clip(left_opening, 0, 100)),
            "left_speed": int(np.clip(left_speed, 0, 100)),
            "left_force": int(np.clip(left_force, 0, 100)),
            "right_opening": int(np.clip(right_opening, 0, 100)),
            "right_speed": int(np.clip(right_speed, 0, 100)),
            "right_force": int(np.clip(right_force, 0, 100))
        }

        self._send_request("request_set_limx_2fclaw_cmd", data)

    def move_head(self, head_joint: Union[List[float], np.ndarray], move_time: float = 5.0):
        """移动头部到指定俯仰/偏航角度。

        头部有两个自由度：
        - pitch（俯仰）：点头动作，正值抬头，负值低头
        - yaw（偏航）：摇头动作，正值向右转，负值向左转

        Args:
            head_joint: 头部关节角度 [pitch, yaw]，2 维
            move_time: 运动时间（秒）

        Raises:
            CommandError: 参数维度错误
        """
        if isinstance(head_joint, np.ndarray):
            head_joint = head_joint.tolist()

        if len(head_joint) != JointIndex.HEAD_DIM:
            raise CommandError(
                f"头部关节应为 {JointIndex.HEAD_DIM} 维, "
                f"实际 {len(head_joint)}"
            )

        if self.motion_mode != MotionMode.MOVEJ:
            self.set_movej_mode()

        data = {
            "joint": head_joint,
            "time": move_time
        }

        self._send_request("request_moveh", data)
        self.logger.debug(f"MoveHead 命令已发送: {head_joint}")

    # =======================================================================
    # 模式切换
    # =======================================================================

    def set_movej_mode(self):
        """切换到 MoveJ 模式（关节空间带插值）。

        发送 mode=0 到机器人控制器，切换到点到点运动模式。
        """
        self._send_request("request_set_servo_mode", {"mode": 0})
        self.motion_mode = MotionMode.MOVEJ
        self.logger.info("已切换到 MoveJ 模式")

    def set_servoj_mode(self):
        """切换到 ServoJ 模式（关节空间实时伺服）。

        发送 mode=1 到机器人控制器，切换到实时跟踪模式。
        切换后重置频率限制器，重新建立时间基准。
        """
        self._send_request("request_set_servo_mode", {"mode": 1})
        self.motion_mode = MotionMode.SERVOJ
        self.servoj_rate_limiter.reset()  # 重置频率时钟
        self.logger.info("已切换到 ServoJ 模式")

    def set_movep_mode(self):
        """切换到 MoveP 模式（笛卡尔空间带插值）。

        注意：这里发送的 mode=0 和 MoveJ 一样（底层控制字相同），
        但通过后续的 MoveP 命令来区分。
        """
        self._send_request("request_set_servo_mode", {"mode": 0})
        self.motion_mode = MotionMode.MOVEP
        self.logger.info("已切换到 MoveP 模式")

    def set_servop_mode(self):
        """切换到 ServoP 模式（笛卡尔空间实时伺服）。

        使用 request_set_servop_mode 命令（与关节模式不同）。
        切换后重置频率限制器。
        """
        self._send_request("request_set_servop_mode")
        self.motion_mode = MotionMode.SERVOP
        self.servop_rate_limiter.reset()  # 重置频率时钟
        self.logger.info("已切换到 ServoP 模式")

    # =======================================================================
    # 辅助功能
    # =======================================================================

    def _move_to_init_pose(self):
        """移动到配置中指定的初始位置。

        执行顺序：
        1. 移动机械臂到初始关节角度（使用 MoveJ，带插值）
        2. 移动头部到初始俯仰/偏航角度（使用 MoveHead）
        3. 打开夹爪到 100%（完全张开）

        【为什么先移动臂再移动头？】
        头部通常比较轻，移动不会造成安全隐患。
        但先移动头部可能会遮挡摄像头视角，影响后续操作的可视化观察。
        所以先移动臂到安全位置，再调整头部视角。
        """
        self.logger.info("正在移动到初始位置...")

        # 第一步：移动机械臂
        if self.config.init_joints is not None:
            self.movej(self.config.init_joints, move_time=2.0)
            time.sleep(3.0)  # 等待 MoveJ 完成

        # 第二步：移动头部
        if self.config.init_head is not None:
            self.move_head(self.config.init_head, move_time=2.0)
            time.sleep(3.0)  # 等待 MoveHead 完成

        # 第三步：打开夹爪
        self.set_gripper(left_opening=100, right_opening=100)

        self.logger.info("初始化完成")

    def wait_until_reached(
        self,
        target_joints: Union[List[float], np.ndarray],
        tolerance: float = 0.05,
        timeout: float = 10.0
    ) -> bool:
        """等待机器人到达目标关节位置。

        【为什么需要这个函数？】
        MoveJ 命令发送后，机器人需要一段时间才能到达目标位置。
        move_time 只是一个"建议时间"，实际到达时间受负载、速度限制等影响。
        这个函数通过持续查询关节状态，确认机器人是否真的到达了目标位置。

        Args:
            target_joints: 目标关节角度（14 维：左臂7 + 右臂7）
            tolerance: 容差（弧度）。默认 0.05 弧度 ≈ 2.86 度。
                       L2 范数误差小于此值认为已到达。
            timeout: 超时时间（秒）。超时后返回 False。

        Returns:
            bool: 是否成功到达目标位置
        """
        if isinstance(target_joints, np.ndarray):
            target_joints = target_joints.tolist()

        target_array = np.array(target_joints)
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                states = self.get_joint_states(timeout=1.0)
                current_joints = states['states']

                # 提取机械臂关节（跳过夹爪和头部）
                arm_states = np.array(
                    current_joints[JointIndex.LEFT_ARM] +
                    current_joints[JointIndex.RIGHT_ARM]
                )

                # 计算 L2 范数误差
                # 使用 L2 范数（欧几里得距离）而不是最大单关节误差，
                # 因为 L2 范数更能反映整体位置的接近程度
                error = np.linalg.norm(arm_states - target_array)

                if error < tolerance:
                    self.logger.info(f"已到达目标位置 (error={error:.4f})")
                    return True

                self.logger.debug(f"当前误差: {error:.4f}, 目标: {tolerance}")
                time.sleep(0.1)  # 100 ms 检查一次

            except StateError:
                self.logger.warning("获取状态失败，重试中...")
                continue

        self.logger.warning(f"等待超时 ({timeout}s)")
        return False

    def emergency_stop(self):
        """紧急停止 —— 立即停止所有运动。

        这是一个安全关键函数，在发生异常情况时调用。
        发送 "request_emgy_stop" 命令到机器人控制器，
        控制器会立即切断电机动力。

        注意：这是一个"best effort"操作，不保证 100% 成功。
        """
        self.logger.warning("触发紧急停止!")
        self._send_request("request_emgy_stop", {})

    def set_light_effect(self, effect: int = 1):
        """设置机器人灯光效果。

        用于视觉反馈，比如：
        - 控制中：蓝色呼吸灯
        - 暂停：黄色常亮
        - 异常：红色闪烁

        Args:
            effect: 灯光效果编号
        """
        self._send_request("request_light_effect", {"effect": effect})

    def is_connected(self) -> bool:
        """检查 WebSocket 连接状态。

        Returns:
            bool: True 表示已连接
        """
        return self.connected

    def disconnect(self):
        """断开连接并清理所有资源。

        安全关闭流程：
        1. 设置 should_exit = True（通知轮询线程退出）
        2. 关闭 WebSocket 连接
        3. 等待 WebSocket 线程结束（最多 2 秒）
        4. 更新连接状态

        注意：应该在程序退出时或不再需要机器人控制时调用。
        """
        self.logger.info("正在断开连接...")
        self.should_exit = True  # 通知轮询线程退出

        if self.ws_client:
            self.ws_client.close()  # 关闭 WebSocket

        if self.ws_thread and self.ws_thread.is_alive():
            self.ws_thread.join(timeout=2.0)  # 等待线程结束

        self.connected = False
        self.logger.info("连接已断开")

    # ---- 上下文管理器支持 ----

    def __enter__(self):
        """上下文管理器入口。"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出时自动断开连接。

        Args:
            exc_type: 异常类型（如果发生异常）
            exc_val: 异常值
            exc_tb: 异常回溯
        """
        self.disconnect()

    def __del__(self):
        """析构函数 —— 对象被垃圾回收时自动断开连接。

        这是一个"安全网"，防止用户忘记显式调用 disconnect()。
        但不应依赖它来替代显式的资源管理。
        """
        if hasattr(self, 'connected') and self.connected:
            self.disconnect()


# ============================================================================
# 示例用法
# ============================================================================

if __name__ == "__main__":
    """独立运行此文件的测试脚本。

    测试内容：
    1. 连接机器人
    2. 等待到达初始位置
    3. 获取末端位姿
    4. 测试 ServoJ（关节伺服控制）
    5. 测试 MoveP（笛卡尔运动）
    6. 测试 ServoP（笛卡尔伺服）
    7. 测试夹爪控制

    注意：运行此脚本需要连接到真实的 Tron2 机器人硬件。
    """
    ee_pose_que = None

    # ---- 配置 ----
    init_joints = [
        0.026899, 0.2612, -0.02709991, -1.5477003, 0.265, 0.0180999, -0.0614999,
        0.008999, -0.269, 0.02069998, -1.5567001, -0.254, -0.02309972, 0.06469989
    ]

    seconde_joints = [
        0.0351, 0.2513, 1.373999, -1.5292, 0.2628, 0.021, -0.0642002,
        0.0202999, -0.2594, -1.369001, -1.5399, -0.256, -0.0129998, 0.0618
    ]
    init_head = [1.0467, -0.0139998]

    config = Tron2Config(
        robot_ip="10.192.1.2",
        init_joints=init_joints,
        init_head=init_head
    )

    # ---- 使用上下文管理器连接机器人 ----
    with Tron2(config) as robot:
        ee_pose_que = robot.ee_pose_queue

        if robot.is_connected():
            print("✅ 机器人连接成功，开始测试...")

            # ============================================================
            # 测试 1: 等待到达初始位置
            # ============================================================
            if robot.wait_until_reached(init_joints, tolerance=0.05):
                print("✅ 已到达初始位置")

            # ============================================================
            # 测试 2: 获取末端位姿
            # ============================================================
            ee_pose_start = robot.get_ee_poses()
            print("ee_pose_start:", ee_pose_start)

            # ============================================================
            # 测试 3: ServoJ（关节空间伺服控制）
            # ============================================================
            print("\n测试 ServoJ...")
            servoj_joint = init_joints + init_head  # 14 + 2 = 16 维
            last_servoj_joint = servoj_joint.copy()
            delta_j = 0.01

            # 下面是被注释掉的 ServoJ 正弦运动测试代码
            # 用于测试连续运动的平滑性
            # amplitude = 0.02
            # freq = 0.5
            # t0 = time.time()
            # while True:
            #     t = time.time() - t0
            #     angle = amplitude * math.sin(2 * math.pi * freq * t)
            #     servoj_joint[5] += angle
            #     servoj_joint[11] += angle
            #     ...

            for i in range(50):
                servoj_joint[-3] -= delta_j  # 头部俯仰逐渐降低
                robot.servoj(servoj_joint)
            print("servoj_joint:\n", servoj_joint)
            print("✅ ServoJ 测试完成")

            # ============================================================
            # 测试 4: MoveP（笛卡尔空间运动）
            # ============================================================
            print("\n测试 MoveP...")
            for i in range(10):
                ee_pose = robot.get_ee_poses()
                left_pose = ee_pose['left_position'] + ee_pose['left_quat']   # 3+4 = 7 维
                right_pose = ee_pose['right_position'] + ee_pose['right_quat'] # 3+4 = 7 维
                print("right_pose:", right_pose)

            left_pose[2] -= 0.1  # 左臂末端下移 10 cm
            ee_pose_cmd = left_pose + right_pose  # 7+7 = 14 维

            robot.movep(ee_pose_cmd, move_time=2.0)
            time.sleep(2.5)

            print("✅ MoveP 测试完成")

            # ============================================================
            # 测试 5: ServoP（笛卡尔空间伺服）
            # ============================================================
            print("\n测试 ServoP...")
            for j in range(10):
                left_pose[2] += 0.01  # 左臂末端上移 1 cm × 10 = 10 cm
                robot.servop(left_pose=left_pose, right_pose=right_pose)
            time.sleep(1)
            print("✅ ServoP 测试完成")

            # ============================================================
            # 测试 6: 夹爪控制
            # ============================================================
            print("\n测试夹爪...")
            left_pos = 100
            right_pos = 100
            robot.set_gripper(left_opening=left_pos, right_opening=right_pos)

            delta_g = 1.0
            for i in range(100):
                left_pos -= delta_g    # 从 100 逐渐减小到 0
                right_pos -= delta_g
                robot.set_gripper(left_opening=left_pos, right_opening=right_pos)

            time.sleep(1.0)

            print("✅ 所有测试完成")

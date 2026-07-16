"""
Tron2 Robot Control Module



提供Tron2机器人的WebSocket控制接口，支持多种运动模式：
- MoveJ/ServoJ: 关节空间控制
- MoveP/ServoP: 笛卡尔空间控制
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
# Rate Limiter
# ============================================================================

class RateLimiter:
    """频率限制器 - 基于绝对时间戳的固定频率控制
    
    确保固定频率执行，不受上层调用时间影响。
    适用于伺服控制等需要精确频率的场景。
    
    Examples:
        >>> limiter = RateLimiter(rate_hz=100.0)
        >>> for i in range(100):
        ...     # 执行控制命令
        ...     send_command()
        ...     # 等待到下一个周期
        ...     limiter.sleep()
    """
    
    def __init__(self, rate_hz: float):
        """初始化频率限制器
        
        Args:
            rate_hz: 目标频率 (Hz)
        """
        self.rate_hz = rate_hz
        self.period = 1.0 / rate_hz
        self.next_tick = time.monotonic()
    
    def sleep(self):
        """等待到下一个周期
        
        基于绝对时间戳计算等待时间，确保固定频率。
        如果上次调用超时，自动跳过周期并重置时钟。
        """
        current_time = time.monotonic()
        sleep_time = self.next_tick - current_time
        
        if sleep_time > 0:
            time.sleep(sleep_time)
        
        # 更新下一个tick（基于绝对时间）
        self.next_tick += self.period
        
        # 防止累积延迟：如果错过了周期，重置时钟
        if self.next_tick < time.monotonic():
            self.next_tick = time.monotonic() + self.period
    
    def reset(self):
        """重置时钟到当前时间"""
        self.next_tick = time.monotonic()


# ============================================================================
# Joint Index Constants
# ============================================================================

class JointIndex:
    """关节索引常量 - 集中管理所有magic index
    
    避免硬编码索引，便于协议变更和添加新关节。
    状态数组布局：[左臂(7), 左夹爪(1), 右臂(7), 右夹爪(1), 头部(2)]
    """
    # 基础维度
    ARM_DIM = 7
    GRIPPER_DIM = 1
    HEAD_DIM = 2
    
    # 复合维度
    STATE_DIM = ARM_DIM + GRIPPER_DIM + ARM_DIM + GRIPPER_DIM + HEAD_DIM  # 18
    SERVOJ_DIM = ARM_DIM * 2 + HEAD_DIM  # 16
    MOVEJ_DIM = ARM_DIM * 2  # 14
    MOVEP_DIM = 14  # 7 (left) + 7 (right)
    SERVOP_DIM = (7, 7)

    # 左臂关节
    LEFT_ARM_START = 0
    LEFT_ARM_END = ARM_DIM
    LEFT_ARM = slice(LEFT_ARM_START, LEFT_ARM_END)
    
    # 左夹爪
    LEFT_GRIPPER = LEFT_ARM_END
    
    # 右臂关节
    RIGHT_ARM_START = LEFT_GRIPPER + GRIPPER_DIM
    RIGHT_ARM_END = RIGHT_ARM_START + ARM_DIM
    RIGHT_ARM = slice(RIGHT_ARM_START, RIGHT_ARM_END)
    
    # 右夹爪
    RIGHT_GRIPPER = RIGHT_ARM_END
    
    # 头部关节
    HEAD_START = RIGHT_GRIPPER + GRIPPER_DIM
    HEAD_END = HEAD_START + HEAD_DIM
    HEAD = slice(HEAD_START, HEAD_END)
    HEAD_PITCH = HEAD_START
    HEAD_YAW = HEAD_START + 1
    
    # 维度常量 (保留兼容性)
    STATE_DIM_WITHOUT_HEAD = STATE_DIM - HEAD_DIM
    STATE_DIM_WITH_HEAD = STATE_DIM
    ARM_JOINT_DIM = ARM_DIM
    TOTAL_ARM_DIM = ARM_DIM * 2


# ============================================================================
# Configuration Data Classes
# ============================================================================
@dataclass
class Tron2Config:
    """Tron2机器人配置"""
    robot_ip: str = "10.192.1.2"
    port: int = 5000
    
    # 初始关节位置 (14维: 两臂各7个关节)
    init_joints: Optional[List[float]] = None
    
    # 初始头部位置 (2维: pitch, yaw)
    init_head: Optional[List[float]] = None
    
    # 状态队列大小
    state_queue_maxlen: int = 7
    
    # 轮询频率 (Hz)
    polling_rate: float = 200.0
    
    # 连接超时 (秒)
    connection_timeout: float = 5.0
    
    # 是否包含头部状态
    include_head_state: bool = True
    
    # Servo模式参数
    servo_kp: List[float] = field(default_factory=lambda: [
        420, 420, 300, 300, 200, 200, 200,  # 左臂
        420, 420, 300, 300, 200, 200, 200,  # 右臂
        60, 60  # 头部
    ])
    
    servo_kd: List[float] = field(default_factory=lambda: [
        12, 12, 15, 15, 10, 10, 10,  # 左臂
        12, 12, 15, 15, 10, 10, 10,  # 右臂
        3, 3  # 头部
    ])
    
    def __post_init__(self):
        """验证配置参数"""
        if self.init_joints is not None and len(self.init_joints) != JointIndex.MOVEJ_DIM:
            raise ValueError(f"init_joints should have {JointIndex.MOVEJ_DIM} elements, got {len(self.init_joints)}")
        
        if self.init_head is not None and len(self.init_head) != JointIndex.HEAD_DIM:
            raise ValueError(f"init_head should have {JointIndex.HEAD_DIM} elements, got {len(self.init_head)}")
        


# ============================================================================
# Enums
# ============================================================================

class MotionMode(Enum):
    """运动控制模式"""
    MOVEJ = "movej"  # 关节空间带插值
    SERVOJ = "servoj"  # 关节空间无插值
    MOVEP = "movep"  # 笛卡尔空间带插值
    SERVOP = "servop"  # 笛卡尔空间无插值



# ============================================================================
# Exceptions
# ============================================================================

class Tron2Error(Exception):
    """Tron2基础异常"""
    pass


class ConnectionError(Tron2Error):
    """连接异常"""
    pass


class CommandError(Tron2Error):
    """命令执行异常"""
    pass


class StateError(Tron2Error):
    """状态获取异常"""
    pass


# ============================================================================
# Main Robot Controller
# ============================================================================

class Tron2:
    """Tron2机器人控制类
    
    Examples:
        >>> config = Tron2Config(robot_ip="10.192.1.2")
        >>> robot = Tron2(config)
        >>> robot.set_movej_mode()
        >>> robot.movej([0.0]*14, move_time=2.0)
        >>> states = robot.get_joint_states()
        >>> robot.disconnect()
    """
    
    def __init__(self, config: Optional[Tron2Config] = None):
        """初始化Tron2机器人控制器
        
        Args:
            config: 机器人配置对象，如果为None则使用默认配置
        """
        self.config = config or Tron2Config()
        self.time_recoder = time.time()
        # 设置日志
        self._setup_logger()
        
        # WebSocket连接相关
        self.accid: Optional[str] = None
        self.ws_client: Optional[websocket.WebSocketApp] = None
        self.ws_thread: Optional[threading.Thread] = None
        self.connected = False
        self.should_exit = False
        
        # 状态管理
        self.motion_mode: Optional[MotionMode] = None
        
        # 状态数据结构
        self._init_state_buffers()
        
        # ServoJ模式参数
        self.servoj_joint_num = JointIndex.SERVOJ_DIM
        self.servoj_args = self._init_servoj_args()
        
        # 伺服频率限制器
        self.servoj_rate_limiter = RateLimiter(rate_hz=100.0)
        self.servop_rate_limiter = RateLimiter(rate_hz=100.0)
        
        # 启动连接
        self._connect()
        time.sleep(1)
        
        # 启动状态轮询
        self._start_polling_threads()
        
        # 初始化到起始位置
        if self.config.init_joints is not None or self.config.init_head is not None:
            self._move_to_init_pose()
    
    def _setup_logger(self):
        """设置日志系统"""
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
    
    def _init_state_buffers(self):
        """初始化状态缓冲区"""
        
        self.joint_states = {
            'timestamp': -1,
            'states': [-1.0] * JointIndex.STATE_DIM_WITH_HEAD,
            'joint_updated': False,  # 标记关节数据是否更新
            'gripper_updated': False  # 标记夹爪数据是否更新
        }
        
        self.ee_pose_states = {
            'timestamp': -1,
            "left_position": [-1.0, -1.0, -1.0],
            "left_quat": [-1.0, -1.0, -1.0, -1.0],
            "right_position": [-1.0, -1.0, -1.0],
            "right_quat": [-1.0, -1.0, -1.0, -1.0]
        }
        
        self.states_mode = None
        
        # 线程安全的队列和状态锁
        self.joint_state_queue = deque(maxlen=self.config.state_queue_maxlen)
        self.ee_pose_queue = deque(maxlen=self.config.state_queue_maxlen)
        self._queue_lock = threading.Lock()
        self._state_lock = threading.Lock()  # 保护joint_states的原子性
    
    def _init_servoj_args(self) -> Dict:
        """初始化Servo模式参数"""
        return {
            "v": [0.0] * self.servoj_joint_num,
            "kp": self.config.servo_kp[:self.servoj_joint_num],
            "kd": self.config.servo_kd[:self.servoj_joint_num],
            "tau": [0.0] * self.servoj_joint_num,
            "mode": [0.0] * self.servoj_joint_num,
            "na": self.servoj_joint_num
        }
    
    # ========================================================================
    # WebSocket 连接管理
    # ========================================================================
    
    def _generate_guid(self) -> str:
        """生成动态GUID"""
        return str(uuid.uuid4())
    
    def _send_request(self, title: str, data: Optional[Dict] = None) -> bool:
        """发送WebSocket请求
        
        Args:
            title: 请求标题/命令
            data: 请求数据
            
        Returns:
            是否发送成功
        """
        if data is None:
            data = {}
        
        if not self.ws_client or not self.connected:
            self.logger.warning(f"WebSocket未连接，无法发送: {title}")
            return False
        
        try:
            message = {
                "accid": self.accid,
                "title": title,
                "timestamp": int(time.time() * 1000),
                "guid": self._generate_guid(),
                "data": data
            }
            
            message_str = json.dumps(message)
            self.ws_client.send(message_str)
            return True
            
        except Exception as e:
            self.logger.error(f"发送请求失败 ({title}): {e}")
            return False
    
    def _on_open(self, ws):
        """WebSocket连接打开回调"""
        self.logger.info(f"机器人连接成功: {self.config.robot_ip}:{self.config.port}")
        self.connected = True
    
    def _on_message(self, ws, message: str):
        """WebSocket接收消息回调"""
        try:
            root = json.loads(message)
            title = root.get("title", "")
            self.accid = root.get("accid", self.accid)
            
            # 分发消息处理
            if title == "response_get_joint_state":
                self._handle_joint_state(root)
            elif title == "response_get_limx_2fclaw_state":
                self._handle_gripper_state(root)
            elif title == "response_get_move_pose":
                self._handle_ee_pose(root)
            elif title not in ["notify_robot_info","response_servoj","response_set_limx_2fclaw_cmd"]:
                self.logger.debug(f"收到消息: {title}")
                
        except json.JSONDecodeError:
            self.logger.error(f"无法解析消息: {message}")
        except Exception as e:
            self.logger.error(f"处理消息异常: {e}")
    
    def _handle_joint_state(self, root: Dict):
        """处理关节状态消息（原子更新）"""
        self.states_mode = 'joint'
        states = root.get("data", {})
        joint_q = states.get("q", [])
        joint_timestamp = root.get("timestamp", -1)
        
        with self._state_lock:
            self.joint_states["timestamp"] = joint_timestamp
            self.joint_states["states"][JointIndex.LEFT_ARM] = joint_q[:JointIndex.ARM_JOINT_DIM]
            self.joint_states["states"][JointIndex.RIGHT_ARM] = joint_q[JointIndex.ARM_JOINT_DIM:JointIndex.TOTAL_ARM_DIM]
            self.joint_states["states"][JointIndex.HEAD_PITCH] = joint_q[14]
            self.joint_states["states"][JointIndex.HEAD_YAW] = joint_q[15]
            temp = self.joint_states["states"][JointIndex.RIGHT_ARM]
            # self.logger.debug(f"right arm state: {temp}")
            
            self.joint_states["joint_updated"] = True
            self._try_commit_state()
    
    def _handle_gripper_state(self, root: Dict):
        """处理夹爪状态消息（原子更新）"""
        self.states_mode = 'gripper'
        claw_data = root.get("data", {})
        
        with self._state_lock:
            # 夹爪开口度归一化到 0-1
            self.joint_states["states"][JointIndex.LEFT_GRIPPER] = claw_data.get("left_opening", -1) / 100.0
            self.joint_states["states"][JointIndex.RIGHT_GRIPPER] = claw_data.get("right_opening", -1) / 100.0
            
            self.joint_states["gripper_updated"] = True
            self._try_commit_state()
        
        self.states_mode = None
    
    def _try_commit_state(self):
        """尝试提交完整状态到队列（需在_state_lock保护下调用）
        
        只有当关节和夹爪数据都更新后才提交，确保状态一致性。
        """
        # 验证状态完整性：关节和夹爪都已更新
        if (self.joint_states["joint_updated"] and 
            self.joint_states["gripper_updated"] and
            self.joint_states["states"][JointIndex.LEFT_ARM_START] != -1 and 
            self.joint_states["timestamp"] != -1):
            
            with self._queue_lock:
                self.joint_state_queue.append(self.joint_states.copy())
            
            # 重置更新标志
            self.joint_states["joint_updated"] = False
            self.joint_states["gripper_updated"] = False
    
    def _handle_ee_pose(self, root: Dict):
        """处理末端位姿消息"""
        self.states_mode = 'ee_pose'
        ee_pose_data = root.get("data", {})
        ee_pose_timestamp = root.get("timestamp", -1)
        
        self.ee_pose_states["timestamp"] = ee_pose_timestamp
        self.ee_pose_states["left_position"] = ee_pose_data.get("left_position", [-1, -1, -1])
        self.ee_pose_states["left_quat"] = ee_pose_data.get("left_quat", [-1, -1, -1, -1])
        self.ee_pose_states["right_position"] = ee_pose_data.get("right_position", [-1, -1, -1])
        self.ee_pose_states["right_quat"] = ee_pose_data.get("right_quat", [-1, -1, -1, -1])
        temp = self.ee_pose_states["right_quat"]
        # self.logger.debug(f"right arm quat:{temp}")
        with self._queue_lock:
            self.ee_pose_queue.append(self.ee_pose_states.copy())
            

    
    def _on_close(self, ws, close_status_code, close_msg):
        """WebSocket连接关闭回调"""
        self.logger.warning(f"机器人连接已关闭: {close_status_code} - {close_msg}")
        self.connected = False
    
    def _on_error(self, ws, error):
        """WebSocket错误回调"""
        self.logger.error(f"WebSocket错误: {error}")
    
    def _connect(self):
        """建立WebSocket连接"""
        ws_url = f"ws://{self.config.robot_ip}:{self.config.port}"
        self.logger.info(f"正在连接机器人: {ws_url}")
        
        self.ws_client = websocket.WebSocketApp(
            ws_url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_close=self._on_close,
            on_error=self._on_error
        )
        
        # 后台线程启动WebSocket
        self.ws_thread = threading.Thread(target=self._run_websocket, daemon=True)
        self.ws_thread.start()
    
    def _run_websocket(self):
        """运行WebSocket客户端循环"""
        try:
            self.ws_client.run_forever()
        except Exception as e:
            self.logger.error(f"WebSocket运行异常: {e}")
    
    # ========================================================================
    # 状态轮询
    # ========================================================================
    
    def _start_polling_threads(self):
        """启动状态轮询线程"""
        self.joint_polling_thread = threading.Thread(
            target=self._poll_feedback, 
            daemon=True
        )
        self.joint_polling_thread.start()
        self.logger.info(f"状态轮询已启动 ({self.config.polling_rate} Hz)")
    
    def _poll_feedback(self):
        """轮询机器人状态"""
        sleep_time = 1.0 / self.config.polling_rate
        
        while not self.should_exit:
            start_time = time.time()
            
            self._send_request("request_get_joint_state")
            self._send_request("request_get_limx_2fclaw_state")
            self._send_request("request_get_move_pose")
            
            # 控制轮询频率
            elapsed = time.time() - start_time
            time.sleep(max(0, sleep_time - elapsed))
    
    # ========================================================================
    # 状态获取接口
    # ========================================================================
    
    def get_joint_states(self, timeout: float = 1.0) -> Dict:
        """获取当前关节状态
        
        Args:
            timeout: 超时时间(秒)
            
        Returns:
            包含关节状态的字典: {'timestamp': int, 'states': List[float]}
            
        Raises:
            StateError: 超时未获取到状态
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self._queue_lock:
                if self.joint_state_queue:
                    states = self.joint_state_queue.pop()# popleft()
                    
                    # 验证状态维度
                    expected_dim = JointIndex.STATE_DIM_WITH_HEAD
                    if len(states['states']) != expected_dim:
                        raise StateError(f"状态维度错误: 期望{expected_dim}, 实际{len(states['states'])}")
                    
                    return states
            
            time.sleep(0.001)
        
        raise StateError(f"获取关节状态超时 ({timeout}s)")
    
    def get_ee_poses(self, timeout: float = 1.0) -> Dict:
        """获取当前末端位姿
        
        Args:
            timeout: 超时时间(秒)
            
        Returns:
            包含末端位姿的字典
            
        Raises:
            StateError: 超时未获取到状态
        """
        if self.motion_mode != MotionMode.MOVEP:
            self.set_movep_mode()
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self._queue_lock:
                if self.ee_pose_queue:
                    return self.ee_pose_queue.popleft()
            
            time.sleep(0.001)
        
        raise StateError(f"获取末端位姿超时 ({timeout}s)")
    
    # ========================================================================
    # 运动控制接口 - 关节空间
    # ========================================================================
    
    def movej(self, joint_positions: Union[List[float], np.ndarray], move_time: float = 2.0):
        """关节空间运动(带插值)
        
        Args:
            joint_positions: 14维关节角度
            move_time: 运动时间(秒)
            
        Raises:
            CommandError: 参数错误 or 发送失败
        """
        if isinstance(joint_positions, np.ndarray):
            joint_positions = joint_positions.tolist()
        
        if len(joint_positions) != JointIndex.MOVEJ_DIM:
            raise CommandError(f"关节角度列表长度应为{JointIndex.MOVEJ_DIM}, 实际{len(joint_positions)}")
        
        # 确保处于MoveJ模式
        if self.motion_mode != MotionMode.MOVEJ:
            self.set_movej_mode()
        
        data = {
            "joint": joint_positions,
            "time": move_time
        }
        
        if not self._send_request("request_movej", data):
            raise CommandError("MoveJ命令发送失败")
        
        self.logger.debug(f"MoveJ命令已发送: time={move_time}s")
    
    def servoj(
        self, 
        joint_positions: Union[List[float], np.ndarray],
    ):
        """关节空间伺服控制(无插值，高频)
        
        Args:
            joint_positions: 关节角度 (必须为16维: 14臂关节 + 2头关节)
        Raises:
            CommandError: 参数错误或发送失败
        """
        if isinstance(joint_positions, np.ndarray):
            joint_positions = joint_positions.tolist()
        
        if len(joint_positions) != self.servoj_joint_num:
            raise CommandError(
                f"关节角度列表长度应为{self.servoj_joint_num}, 实际{len(joint_positions)}"
            )
        
        servo_data = {
            "q": joint_positions, # todo
            "filter_ratio": 1.0
        }
        
        if not self._send_request("request_servoj", servo_data):
            raise CommandError("ServoJ命令发送失败")
        self.servoj_rate_limiter.sleep()  # 固定频率控制
    
    # ========================================================================
    # 运动控制接口 - 笛卡尔空间
    # ========================================================================
    
    def movep(self, pose_quat_list: Union[List[float], np.ndarray], move_time: float = 5.0):
        """笛卡尔空间运动(带插值)
        
        Args:
            pose_quat_list: 14维位姿 [left_xyz(3), left_wxyz(4), right_xyz(3), right_wxyz(4)]
            move_time: 运动时间(秒)
            
        Raises:
            CommandError: 参数错误 or 发送失败
        """
        if isinstance(pose_quat_list, np.ndarray):
            pose_quat_list = pose_quat_list.tolist()
        
        if len(pose_quat_list) != JointIndex.MOVEP_DIM:
            raise CommandError(f"位姿列表长度应为{JointIndex.MOVEP_DIM}, 实际{len(pose_quat_list)}")
        
        if self.motion_mode != MotionMode.MOVEP:
            self.set_movep_mode()
        
        data = {
            "pos": pose_quat_list,
            "time": move_time
        }
        
        if not self._send_request("request_movep", data):
            raise CommandError("MoveP命令发送失败")
        
        self.logger.debug(f"MoveP命令已发送: time={move_time}s")
    
    def servop(
        self, 
        left_pose: Union[List[float], np.ndarray],
        right_pose: Union[List[float], np.ndarray],
        move_time: float = 5.0
    ):
        """笛卡尔空间伺服控制(无插值)
        
        Args:
            left_pose: 左臂末端位姿 [xyz(3), wxyz(4)]
            right_pose: 右臂末端位姿 [xyz(3), wxyz(4)]
            move_time: 运动时间(秒)
            
        Raises:
            CommandError: 参数错误或发送失败
        """
        if isinstance(left_pose, np.ndarray):
            left_pose = left_pose.tolist()
        if isinstance(right_pose, np.ndarray):
            right_pose = right_pose.tolist()
        
        if len(left_pose) != JointIndex.SERVOP_DIM[0]:
            raise CommandError(f"左臂位姿长度应为{JointIndex.SERVOP_DIM[0]}, 实际{len(left_pose)}")
        if len(right_pose) != JointIndex.SERVOP_DIM[1]:
            raise CommandError(f"右臂位姿长度应为{JointIndex.SERVOP_DIM[1]}, 实际{len(right_pose)}")
        
        if self.motion_mode != MotionMode.SERVOP:
            self.set_servop_mode()
        
        data = {
            "left_pos": left_pose,
            "right_pos": right_pose,
            "time": move_time
        }
        
        if not self._send_request("request_servop", data):
            raise CommandError("ServoP命令发送失败")
        
        self.servop_rate_limiter.sleep()  # 固定频率控制
    
    # ========================================================================
    # 夹爪和头部控制
    # ========================================================================
    
    def set_gripper(
        self,
        left_opening: float = 0.0,
        right_opening: float = 0.0,
        left_speed: float = 100.0,
        left_force: float = 50.0,
        right_speed: float = 100.0,
        right_force: float = 50.0
    ):
        """设置夹爪参数
        
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
        """移动头部到指定位置
        
        Args:
            head_joint: 头部关节角度 [pitch, yaw]
            move_time: 运动时间(秒)
        """
        if isinstance(head_joint, np.ndarray):
            head_joint = head_joint.tolist()
        
        if len(head_joint) != JointIndex.HEAD_DIM:
            raise CommandError(f"头部关节应为{JointIndex.HEAD_DIM}维, 实际{len(head_joint)}")
        
        if self.motion_mode != MotionMode.MOVEJ:
            self.set_movej_mode()
        
        data = {
            "joint": head_joint,
            "time": move_time
        }
        
        self._send_request("request_moveh", data)
        self.logger.debug(f"MoveHead命令已发送: {head_joint}")
    
    # ========================================================================
    # 模式切换
    # ========================================================================
    
    def set_movej_mode(self):
        """切换到MoveJ模式"""
        self._send_request("request_set_servo_mode", {"mode": 0})
        self.motion_mode = MotionMode.MOVEJ
        self.logger.info("已切换到MoveJ模式")
    
    def set_servoj_mode(self):
        """切换到ServoJ模式"""
        self._send_request("request_set_servo_mode", {"mode": 1})
        self.motion_mode = MotionMode.SERVOJ
        self.servoj_rate_limiter.reset()  # 重置时钟
        self.logger.info("已切换到ServoJ模式")
    
    def set_movep_mode(self):
        """切换到MoveP模式"""
        self._send_request("request_set_servo_mode", {"mode": 0})
        self.motion_mode = MotionMode.MOVEP
        self.logger.info("已切换到MoveP模式")
    
    def set_servop_mode(self):
        """切换到ServoP模式"""
        self._send_request("request_set_servop_mode")
        self.motion_mode = MotionMode.SERVOP
        self.servop_rate_limiter.reset()  # 重置时钟
        self.logger.info("已切换到ServoP模式")
    
    # ========================================================================
    # 辅助功能
    # ========================================================================
    
    def _move_to_init_pose(self):
        """移动到初始位置"""
        self.logger.info("正在移动到初始位置...")
        # 2. 移动机械臂
        if self.config.init_joints is not None:
            self.movej(self.config.init_joints, move_time=2.0)
            # self.movej(self.config.init_joints, move_time=2.0)
            time.sleep(3.0)
        # self.logger.debug(f"self.config.init_joints:,{self.config.init_joints}")
        # 1. 先移动头部
        if self.config.init_head is not None:
            self.move_head(self.config.init_head, move_time=2.0)
            time.sleep(3.0)
        self.set_gripper(left_opening=100, right_opening=100)

        
        self.logger.info("初始化完成")
    
    def wait_until_reached(
        self, 
        target_joints: Union[List[float], np.ndarray],
        tolerance: float = 0.05,
        timeout: float = 10.0
    ) -> bool:
        """等待机器人到达目标位置
        
        Args:
            target_joints: 目标关节角度 (14维)
            tolerance: 容差(弧度)
            timeout: 超时时间(秒)
            
        Returns:
            是否成功到达
        """
        if isinstance(target_joints, np.ndarray):
            target_joints = target_joints.tolist()
        
        target_array = np.array(target_joints)
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                states = self.get_joint_states(timeout=1.0)
                current_joints = states['states']
                
                    # 提取机械臂关节 (跳过夹爪)
                arm_states = np.array(
                    current_joints[JointIndex.LEFT_ARM] + 
                    current_joints[JointIndex.RIGHT_ARM]
                )
                
                error = np.linalg.norm(arm_states - target_array)
                
                if error < tolerance:
                    self.logger.info(f"已到达目标位置 (error={error:.4f})")
                    return True
                
                self.logger.debug(f"当前误差: {error:.4f}, 目标: {tolerance}")
                time.sleep(0.1)
                
            except StateError:
                self.logger.warning("获取状态失败，重试中...")
                continue
        
        self.logger.warning(f"等待超时 ({timeout}s)")
        return False
    
    def emergency_stop(self):
        """紧急停止"""
        self.logger.warning("触发紧急停止!")
        self._send_request("request_emgy_stop", {})
    
    def set_light_effect(self, effect: int = 1):
        """设置灯光效果
        
        Args:
            effect: 效果编号
        """
        self._send_request("request_light_effect", {"effect": effect})
    
    def is_connected(self) -> bool:
        """检查连接状态"""
        return self.connected
    
    def disconnect(self):
        """断开连接并清理资源"""
        self.logger.info("正在断开连接...")
        self.should_exit = True
        
        if self.ws_client:
            self.ws_client.close()
        
        if self.ws_thread and self.ws_thread.is_alive():
            self.ws_thread.join(timeout=2.0)
        
        self.connected = False
        self.logger.info("连接已断开")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        # time.sleep(3)
        self.disconnect()
    
    def __del__(self):
        """析构函数"""
        if hasattr(self, 'connected') and self.connected:
            self.disconnect()


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    ee_pose_que = None
    # 配置机器人参数
    init_joints = [
        0.026899, 0.2612, -0.02709991, -1.5477003, 0.265, 0.0180999, -0.0614999,
        0.008999, -0.269, 0.02069998, -1.5567001, -0.254, -0.02309972, 0.06469989
    ]
    
    seconde_joints = [0.0351,0.2513,1.373999,-1.5292,0.2628,0.021,-0.0642002,
                    0.0202999,-0.2594,-1.369001,-1.5399,-0.256,-0.0129998,0.0618]
    init_head = [1.0467, -0.0139998]
    
    config = Tron2Config(
        robot_ip="10.192.1.2",
        init_joints=init_joints,
        init_head=init_head
    )
    
    # 使用上下文管理器
    with Tron2(config) as robot:
        ee_pose_que = robot.ee_pose_queue
        if robot.is_connected():
            print("✅ 机器人连接成功，开始测试...")
            
            # 等待到达初始位置
            if robot.wait_until_reached(init_joints, tolerance=0.05):
                print("✅ 已到达初始位置")
            
            ee_pose_start = robot.get_ee_poses()
            print("ee_pose_start:", ee_pose_start)

            # 测试ServoJ
            print("\n测试ServoJ...")
            servoj_joint = init_joints + init_head
            last_servoj_joint = servoj_joint.copy()
            delta_j = 0.01

            # # robot.set_gripper(left_opening=0,right_opening=0)
            # gripper_state = "open"
            # # 正弦运动参数
            # amplitude = 0.02      # 振幅（弧度），可根据需要调整
            # freq = 0.5           # 频率（Hz）
            # t0 = time.time()
            # while True:
            #     t = time.time() - t0
            #     angle = amplitude * math.sin(2 * math.pi * freq * t)
                
            #     # 左臂 (0-6)
            #     # for i in range(7):
            #     servoj_joint[5] += angle
            #     # 右臂 (7-13)
            #     # for i in range(7, 14):
            #     servoj_joint[11] += angle
            #     # 头 (14-15)
            #     # servoj_joint[14] += angle
            #     # servoj_joint[15] += angle * 0.5
            #     error = np.abs(np.array(servoj_joint) - np.array(last_servoj_joint))
            #     id = np.argmax(error)
            #     max_diff = error[id]
            #     print(f"joint {id} 's error is {max_diff}")
            #     # print(servoj_joint)
            #     if gripper_state == "open":
            #         gripper_state = "close"
            #         robot.set_gripper(left_opening=0,right_opening=0)
            #     else:
            #         gripper_state = 'open'
            #         robot.set_gripper(left_opening=100,right_opening=100)
            #     robot.servoj(servoj_joint)
            #     last_servoj_joint = servoj_joint.copy()
            #     # time.sleep(0.01)

            for i in range(50):
                servoj_joint[-3] -= delta_j
                robot.servoj(servoj_joint)
            print("servoj_joint:\n",servoj_joint)
            print("✅ ServoJ测试完成")

            # ee_pose = robot.get_ee_poses()
            # robot.set_movep_mode()
            # time.sleep(2)
            # robot.move_head([0.0,0.0])
            # # robot.movej(init_joints)
            # time.sleep(5)
            
            # 测试MoveP
            print("\n测试MoveP...")
            for i in range(10):
                ee_pose = robot.get_ee_poses()
                left_pose = ee_pose['left_position'] + ee_pose['left_quat']
                right_pose = ee_pose['right_position'] + ee_pose['right_quat']
                print("right_pose:",right_pose)
            left_pose[2] -= 0.1  # 下移10cm
            ee_pose_cmd = left_pose + right_pose
            
            robot.movep(ee_pose_cmd, move_time=2.0)
            time.sleep(2.5)
            
            print("✅ MoveP测试完成")
            
            # 测试ServoP
            print("\n测试ServoP...")
            for j in range(10):
                left_pose[2] += 0.01  # 上移
                robot.servop(left_pose=left_pose, right_pose=right_pose)
            time.sleep(1)
            print("✅ ServoP测试完成")
            
            # 测试夹爪
            print("\n测试夹爪...")
            left_pos = 100
            right_pos = 100
            robot.set_gripper(left_opening=left_pos, right_opening=right_pos)
            
            delta_g = 1.0
            for i in range(100):
                left_pos -= delta_g
                right_pos -= delta_g
                robot.set_gripper(left_opening=left_pos, right_opening=right_pos)
                # robot.set_gripper(left_opening=50,right_opening=50)

            time.sleep(1.0)
            
            print("✅ 所有测试完成")

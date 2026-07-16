"""
Tron2 Real Robot Environment

提供Tron2机器人的环境封装，包括：
- 机器人控制
- 多相机图像采集
- 观测与动作的时间同步
- 轨迹插值
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
# Configuration
# ============================================================================

@dataclass
class CameraConfig:
    """相机配置"""
    # 相机名称映射
    camera_names: List[str] = field(default_factory=lambda: [
        "head_camera_image",
        "left_wrist_image", 
        "right_wrist_image"
    ])
    
    # 观测输出的相机名称
    obs_camera_names: List[str] = field(default_factory=lambda: [
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist"
    ])
    
    # 相机分辨率 (H, W, C)
    resolution: Tuple[int, int, int] = (480, 640, 3)
    
    # 最大队列大小
    max_queue_size: int = 10
    
    # 是否保存调试图像
    save_debug_images: bool = True
    debug_image_dir: str = "./debug_images"

    


@dataclass
class EnvConfig:
    """环境配置"""
    # 机器人配置
    robot_config: Tron2Config = field(default_factory=Tron2Config)
    
    # 相机配置
    camera_config: CameraConfig = field(default_factory=CameraConfig)
    
    # 轨迹插值点数
    interp_points: int = 8
    
    # 时间同步容差 (秒)
    time_sync_tolerance: float = 0.01
    time_sync_max_retries: int = 3
    
    # 夹爪初始化开口度 (0-1)
    init_gripper_opening: float = 0.9

    # 原始配置字典（用于透传给其他组件）
    raw_config: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Tron2 Environment
# ============================================================================

class Tron2Env:
    """Tron2机器人环境
    
    Examples:
        >>> config = EnvConfig(robot_config=Tron2Config(robot_ip="10.192.1.2"))
        >>> env = Tron2Env(config)
        >>> obs = env.reset()
        >>> action = np.zeros(16)  # 16维动作
        >>> env.step(action)
    """
    
    def __init__(self, config: Optional[EnvConfig] = None):
        """初始化环境
        
        Args:
            config: 环境配置，如果为None则使用默认配置
        """
        self.config = config or EnvConfig()
        
        # 设置日志
        self._setup_logger()
        
        # 初始化机器人
        self.logger.info("正在初始化机器人控制器...")
        self.robot = Tron2(self.config.robot_config)
        
        # 初始化相机
        self.logger.info("正在初始化相机...")
        self.camera_manager = self._init_camera()
        
        # 状态管理
        self.last_action: Optional[np.ndarray] = None
        self.init_joints = self.config.robot_config.init_joints
        
        # 创建调试图像目录
        if self.config.camera_config.save_debug_images:
            Path(self.config.camera_config.debug_image_dir).mkdir(parents=True, exist_ok=True)
        
        self.logger.info("环境初始化完成")
    
    def _setup_logger(self):
        """设置日志系统"""
        self.logger = logging.getLogger("Tron2Env")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(asctime)s.%(msecs)03d] [%(name)s] [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _init_camera(self):
        """初始化相机管理器"""
        try:
            from realsense_image import MultiCameraManager
        except ImportError:
            self.logger.error("无法导入 MultiCameraManager，请确保 realsense_image 模块已安装")
            raise
        
        # 尝试从 YAML 加载（如果存在配置字典）
        if hasattr(self.config, 'raw_config'):
            camera_manager = MultiCameraManager.from_config(self.config.raw_config)
        else:
            camera_manager = MultiCameraManager(
                max_queue_size=self.config.camera_config.max_queue_size
            )
        
        camera_manager.start_capture()
        
        # 等待相机预热
        self.logger.info("相机预热中...")
        time.sleep(3.0)
        
        return camera_manager
    
    # ========================================================================
    # Environment Interface
    # ========================================================================
    
    def reset(self) -> Dict:
        """重置环境到初始状态
        
        Returns:
            初始观测
        """
        self.logger.info("重置环境...")
        
        # 获取当前观测
        obs = self.get_obs()
        
        # 验证图像尺寸
        expected_shape = self.config.camera_config.resolution
        for cam_name in self.config.camera_config.obs_camera_names:
            actual_shape = obs['images'][cam_name].shape
            if actual_shape != expected_shape:
                self.logger.warning(
                    f"{cam_name} 分辨率不匹配: 期望{expected_shape}, 实际{actual_shape}"
                )
        
        # 验证机器人位置
        if self.init_joints is not None:
            current_state = obs['state']
            arm_states = np.concatenate([
                current_state[JointIndex.LEFT_ARM], 
                current_state[JointIndex.RIGHT_ARM]
            ])
            init_arm = np.array(self.init_joints)
            
            error = np.abs(arm_states - init_arm).max()
            if error > 0.05:
                self.logger.warning(f"机器人未在初始位置，最大误差: {error:.4f}")
                self.robot.wait_until_reached(self.init_joints,tolerance=0.05)
        
        # 初始化夹爪
        test_action = obs['state'].copy()
        test_action[JointIndex.LEFT_GRIPPER] = self.config.init_gripper_opening
        test_action[JointIndex.RIGHT_GRIPPER] = self.config.init_gripper_opening
        self.step(test_action)
        
        self.logger.info("环境重置完成")
        return obs

    def step(self, action: Union[List[float], np.ndarray]):
        """执行动作
        
        Args:
            action: 14/16/18维动作向量
                   - 14维: [7关节(左), 7关节(右)]
                   - 16维: [7关节+1夹爪(左), 7关节+1夹爪(右)]
                   - 18维: [7关节+1夹爪(左), 7关节+1夹爪(右), 2头部]
        """
        # 输入验证
        if isinstance(action, list):
            action = np.array(action)
        
        if len(action) not in [JointIndex.MOVEJ_DIM, JointIndex.SERVOJ_DIM, JointIndex.STATE_DIM]:
            raise ValueError(f"动作维度应为{JointIndex.MOVEJ_DIM}/{JointIndex.SERVOJ_DIM}/{JointIndex.STATE_DIM}, 实际{len(action)}")
        
        # 提取机械臂动作 (前14维)
        arm_action = np.concatenate([
            action[JointIndex.LEFT_ARM], 
            action[JointIndex.RIGHT_ARM]
        ])
        
        # 提取头部动作 (2维)
        if len(action) >= JointIndex.STATE_DIM:
            head_action = action[JointIndex.HEAD]
        else:
            # 如果action不包含头部，则使用当前头部位置
            with self.robot._state_lock:
                curr_states = self.robot.joint_states['states']
                if len(curr_states) >= JointIndex.STATE_DIM_WITH_HEAD:
                    head_action = np.array(curr_states[JointIndex.HEAD])
                else:
                    head_action = np.array([0.0, 0.0])
        
        # 组合为16维伺服动作 (14臂 + 2头)
        full_servo_action = np.concatenate([arm_action, head_action])
        
        # 提取夹爪动作 (归一化到0-100)
        gripper_action = np.array([
            action[JointIndex.LEFT_GRIPPER], 
            action[JointIndex.RIGHT_GRIPPER]
        ]) * 100.0
        gripper_action = np.clip(gripper_action, 0, 100)
        
        # 首次调用处理
        if self.last_action is None:
            self.robot.servoj(full_servo_action)
            self.robot.set_gripper(
                left_opening=gripper_action[0], 
                right_opening=gripper_action[1]
            )
            self.last_action = full_servo_action
            return
        
        # 设置夹爪
        # gripper_action[0] = gripper_action[0] if gripper_action[0] > 10 else 0
        # gripper_action[1] = gripper_action[1] if gripper_action[1] > 10 else 0

        self.robot.set_gripper(
            left_opening=gripper_action[0], 
            right_opening=gripper_action[1]
        )
        
        # 轨迹插值 (16维)
        interpolated_traj = self._interpolate_trajectory(
            start=self.last_action,
            end=full_servo_action,
            num_points=self.config.interp_points
        )
        
        # 执行轨迹 (跳过起点)
        for i in range(1, len(interpolated_traj)):
            time_servoj = time.time()
            self.robot.servoj(interpolated_traj[i])
            time_servoj2 = time.time()
            dt = time_servoj2-time_servoj
            # self.logger.info(f"time1 : {time_servoj}, time2 : {time_servoj2},control rate: {1/dt}")
        # 更新历史动作
        self.last_action = full_servo_action
    
    def get_obs(self) -> Dict:
        """获取当前观测
        
        Returns:
            观测字典: {
                'state': np.ndarray,  # 关节状态 (16/18维)
                'images': Dict[str, np.ndarray]  # 图像字典
            }
        """
        # 1. 获取图像 (时间戳较早)
        rgb_images = self._get_images()
        img_timestamp = rgb_images['head_camera_image_timestamp']
        
        # 保存调试图像
        if self.config.camera_config.save_debug_images:
            self._save_debug_images(rgb_images)
        
        # 2. 获取关节状态
        qpos_dict = self._get_qpos()
        joint_timestamp = qpos_dict['timestamp'] / 1000.0
        
        self.logger.debug(
            f"时间戳 - 关节: {joint_timestamp:.3f}s, 图像: {img_timestamp:.3f}s, "
            f"差值: {joint_timestamp - img_timestamp:.3f}s"
        )
        
        # 3. 时间同步
        synced_qpos = self._sync_observation(
            img_timestamp=img_timestamp,
            initial_qpos=qpos_dict,
            using_sync = True,
        )
        
        # 4. 构建观测
        obs = {
            "state": np.array(synced_qpos['states'][:16]),
            "images": {
                self.config.camera_config.obs_camera_names[0]: rgb_images['head_camera_image'],
                self.config.camera_config.obs_camera_names[1]: rgb_images['left_wrist_image'],
                self.config.camera_config.obs_camera_names[2]: rgb_images['right_wrist_image']
            }
        }
        
        return obs
    
    # ========================================================================
    # Private Methods
    # ========================================================================
    
    def _interpolate_trajectory(
        self, 
        start: np.ndarray, 
        end: np.ndarray, 
        num_points: int
    ) -> np.ndarray:
        """线性插值生成轨迹
        
        Args:
            start: 起始位置 (16维: 14臂 + 2头)
            end: 结束位置 (16维: 14臂 + 2头)
            num_points: 插值点数
            
        Returns:
            插值轨迹 (num_points, 16)
        """
        t = np.linspace(0, 1, num_points)
        interpolated = np.zeros((num_points, JointIndex.SERVOJ_DIM))
        
        for i in range(JointIndex.SERVOJ_DIM):
            interpolated[:, i] = np.interp(t, [0, 1], [start[i], end[i]])
        
        return interpolated
    
    def _get_qpos(self) -> Dict:
        """获取关节状态"""
        return self.robot.get_joint_states(timeout=0.5)
    
    def _get_images(self) -> Dict:
        """获取相机图像
        
        Returns:
            图像字典: {
                'head_camera_image': np.ndarray,
                'head_camera_image_timestamp': float,
                ...
            }
        """
        all_frames = self.camera_manager.get_all_latest_frames()
        image_dict = {}
        
        for camera_name, frame_data in all_frames.items():
            if frame_data is not None:
                # BGR转RGB
                image_dict[camera_name] = frame_data['color'][:, :, ::-1]
                image_dict[f'{camera_name}_timestamp'] = frame_data['timestamp']
        
        return image_dict
    
    def _sync_observation(
        self, 
        img_timestamp: float, 
        initial_qpos: Dict,
        using_sync: bool = False
    ) -> Dict:
        """同步观测时间戳
        
        策略: 以图像时间戳为基准，尝试获取时间戳接近的关节状态
        
        Args:
            img_timestamp: 图像时间戳 (秒)
            initial_qpos: 初始关节状态
            
        Returns:
            同步后的关节状态
        """
        joint_timestamp = initial_qpos['timestamp'] / 1000.0
        time_diff = abs(joint_timestamp - img_timestamp)
        time_dif = joint_timestamp - img_timestamp
        
        if not using_sync:
            self.logger.info(f"(diff={time_dif:.4f}s)")
            return initial_qpos
        # 如果时间差在容差范围内，直接返回,or 
        if time_diff <= self.config.time_sync_tolerance:
            self.logger.debug(f"✅ 时间同步成功 (diff={time_dif:.4f}s)")
            return initial_qpos
        if joint_timestamp > img_timestamp:
            self.logger.warning(f"⚠️ joint later than img (diff={time_dif:.4f}s)")
            return initial_qpos
        # 尝试重新获取
        self.logger.debug(f"⚠️ 时间差过大 ({time_dif:.4f}s), 尝试重新同步...")
        
        for retry in range(self.config.time_sync_max_retries):
            time.sleep(0.005)
            
            qpos_dict = self._get_qpos()
            joint_timestamp = qpos_dict['timestamp'] / 1000.0
            time_diff = abs(joint_timestamp - img_timestamp)
            
            if time_diff <= self.config.time_sync_tolerance:
                self.logger.debug(
                    f"✅ 时间同步成功 (尝试{retry+1}次, diff={time_dif:.4f}s)"
                )
                return qpos_dict
        
        # 重试失败，使用最新数据
        self.logger.warning(
            f"⚠️ 时间同步失败 ({self.config.time_sync_max_retries}次重试后), "
            f"使用最新数据 joint{joint_timestamp}-img{img_timestamp}(diff={time_dif:.4f}s)"
        )
        return qpos_dict
    
    def _save_debug_images(self, rgb_images: Dict):
        """保存调试图像"""
        debug_dir = Path(self.config.camera_config.debug_image_dir)
        timestamp = time.time()
        
        for key in ['head_camera_image', 'left_wrist_image', 'right_wrist_image']:
            if key in rgb_images:
                img = Image.fromarray(rgb_images[key])
                save_path = debug_dir / f"{key}.jpg"
                img.save(save_path)
    
    def close(self):
        """关闭环境并释放资源"""
        self.logger.info("关闭环境...")
        
        if hasattr(self, 'robot'):
            self.robot.disconnect()
        
        if hasattr(self, 'camera_manager'):
            self.camera_manager.stop_capture()
        
        self.logger.info("环境已关闭")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.close()


# ============================================================================
# Policy Wrapper (Example)
# ============================================================================

class PolicyWrapper:
    """策略包装器基类"""
    
    def get_action(self, observation: Dict) -> np.ndarray:
        """获取动作
        
        Args:
            observation: 观测字典
            
        Returns:
            动作数组
        """
        raise NotImplementedError


class WebsocketPolicyWrapper(PolicyWrapper):
    """基于WebSocket的策略客户端"""
    
    def __init__(self, host: str = "localhost", port: int = 8000):
        """初始化WebSocket策略客户端
        
        Args:
            host: 服务器地址
            port: 服务器端口
        """
        try:
            from openpi_client import websocket_client_policy, image_tools
            self.ws_client = websocket_client_policy.WebsocketClientPolicy(
                host=host, 
                port=port
            )
            self.image_tools = image_tools
        except ImportError as e:
            raise ImportError(f"无法导入 openpi_client: {e}")
        
        self.logger = logging.getLogger("WebsocketPolicy")
    
    def get_action(self, observation: Dict) -> np.ndarray:
        """通过WebSocket获取动作
        
        Args:
            observation: 观测字典
            
        Returns:
            动作序列 (action_horizon, action_dim)
        """
        import einops
        
        # 预处理图像
        obs = observation.copy()
        for cam_name in obs["images"]:
            img = self.image_tools.convert_to_uint8(
                self.image_tools.resize_with_pad(obs["images"][cam_name], 224, 224)
            )
            obs["images"][cam_name] = einops.rearrange(img, "h w c -> c h w")
        
        # 推理
        result = self.ws_client.infer(obs)
        actions = np.stack(result['actions'], axis=0)
        
        return actions


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # 配置环境
    init_joints = [
        0.026899, 0.2612, -0.02709991, -1.5477003, 0.265, 0.0180999, -0.0614999,
        0.008999, -0.269, 0.02069998, -1.5567001, -0.254, -0.02309972, 0.06469989
    ]
    
    init_head = [1.0567, -0.0139998]
    
    robot_config = Tron2Config(
        robot_ip="10.192.1.2",
        init_joints=init_joints,
        init_head=init_head
    )
    serial_to_name = {'serial_to_name':{
            "245022302696": 'head_camera_image',
            "409122274385": 'left_wrist_image',
            "230322276915": 'right_wrist_image'
        }}
    env_config = EnvConfig(
        robot_config=robot_config,
        interp_points=6,
        time_sync_tolerance=0.01,
        raw_config = {'camera':serial_to_name}
    )
    
    # 使用上下文管理器
    with Tron2Env(env_config) as env:
        # 重置环境
        obs = env.reset()
        print(f"✅ 环境重置完成")
        print(f"   状态维度: {obs['state'].shape}")
        print(f"   图像数量: {len(obs['images'])}")
        
        # 初始化策略
        try:
            policy = WebsocketPolicyWrapper(host='0.0.0.0', port=8000)
            print(f"✅ 策略加载完成")
        except ImportError:
            print("⚠️ 无法加载策略，使用随机动作")
            policy = None
        
        # 运行循环
        max_steps = 100
        for step in range(max_steps):
            print(f"\n{'='*50}")
            print(f"Step {step+1}/{max_steps}")
            print(f"{'='*50}")
            
            # 获取观测
            obs = env.get_obs()
            print(f"✅ 获取观测: state={obs['state'][:4]}...")
            
            # 获取动作
            if policy is not None:
                try:
                    actions = policy.get_action(obs)
                    print(f"✅ 策略推理完成: {actions.shape}")
                    
                    # 执行动作序列
                    for action in actions:
                        env.step(action)
                        
                except Exception as e:
                    print(f"⚠️ 策略推理失败: {e}")
                    break
            else:
                # 随机动作测试
                action = obs['state'].copy()
                action[:JointIndex.MOVEJ_DIM] += np.random.randn(JointIndex.MOVEJ_DIM) * 0.01  # 小扰动
                env.step(action)
            
            time.sleep(0.1)
        
        print(f"\n✅ 测试完成")
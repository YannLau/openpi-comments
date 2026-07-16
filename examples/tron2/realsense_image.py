import logging
import pyrealsense2 as rs
import numpy as np
import cv2
from threading import Thread, Lock
import time
from collections import deque
from typing import Dict, List, Optional, Tuple, Any


# ============================================================================
# Multi-Camera Manager
# ============================================================================

class MultiCameraManager:
    """RealSense 多相机管理器
    
    支持多相机同步采集、自动重连和线程安全的状态获取。
    """
    
    def __init__(
        self, 
        max_queue_size: int = 10,
        serial_to_name: Optional[Dict[str, str]] = None,
        camera_configs: Optional[Dict[str, Dict[str, int]]] = None
    ):
        """初始化多相机管理器
        
        Args:
            max_queue_size: 每个相机的最大队列长度
            serial_to_name: 序列号到名称的映射
            camera_configs: 相机配置（分辨率、FPS）
        """
        self._setup_logger()
        
        self.max_queue_size = max_queue_size
        self.running = False
        self.lock = Lock()
        
        # 30
        # self.serial_to_name = serial_to_name or {
        #     "245022302696": 'head_camera_image',
        #     "409122274385": 'left_wrist_image',
        #     "230322276915": 'right_wrist_image'
        # }
        # 31
        self.serial_to_name = serial_to_name or {
            "245022302696": 'head_camera_image',
            "409122274385": 'left_wrist_image',
            "230322276915": 'right_wrist_image'
        }
        # 默认相机配置
        if camera_configs:
            self.camera_configs = camera_configs
        else:
            self.camera_configs = {
                'left_wrist_image': {'color_width': 640, 'color_height': 480, 'fps': 30},
                'right_wrist_image': {'color_width': 640, 'color_height': 480, 'fps': 30},
                'head_camera_image': {'color_width': 640, 'color_height': 480, 'fps': 30}
            }
        
        self.pipeline_dict = {}
        self.frame_queues = {name: deque(maxlen=max_queue_size) for name in self.camera_configs}
        self.time_stamps = {name: deque(maxlen=100) for name in self.camera_configs}
        
        self.capture_thread: Optional[Thread] = None

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]):
        """从配置字典创建管理器实例"""
        camera_cfg = config_dict.get('camera', {})
        
        # 提取序列号映射（如果存在）
        serial_to_name = camera_cfg.get('serial_to_name')
        
        # 构造相机配置
        res = camera_cfg.get('resolution', [640, 480])
        fps = camera_cfg.get('fps', 30)
        
        camera_configs = {}
        for name in camera_cfg.get('camera_names', []):
            camera_configs[name] = {
                'color_width': res[1], # W
                'color_height': res[0], # H
                'fps': fps
            }
            
        return cls(
            max_queue_size=camera_cfg.get('max_queue_size', 10),
            serial_to_name=serial_to_name,
            camera_configs=camera_configs if camera_configs else None
        )

    def _setup_logger(self):
        self.logger = logging.getLogger("CameraManager")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def detect_cameras(self) -> List[str]:
        """检测连接的 RealSense 相机序列号"""
        ctx = rs.context()
        devices = ctx.query_devices()
        serials = [dev.get_info(rs.camera_info.serial_number) for dev in devices]
        return [s for s in serials if 'Asic' not in s]

    def setup_pipelines(self):
        """配置所有相机的采集管道"""
        serial_numbers = self.detect_cameras()
        self.logger.info(f"检测到 {len(serial_numbers)} 个相机: {serial_numbers}")
        
        if not serial_numbers:
            raise RuntimeError("未检测到 RealSense 相机")
        
        for serial in serial_numbers:
            if serial not in self.serial_to_name:
                self.logger.warning(f"序列号 {serial} 未定义映射名称，跳过")
                continue
                
            camera_name = self.serial_to_name[serial]
            cam_cfg = self.camera_configs.get(camera_name, {'color_width': 640, 'color_height': 480, 'fps': 30})
            
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)
            
            # 配置颜色流
            config.enable_stream(
                rs.stream.color, 
                cam_cfg['color_width'], cam_cfg['color_height'], 
                rs.format.bgr8, cam_cfg['fps']
            )
            
            # 配置深度流
            config.enable_stream(
                rs.stream.depth, 
                cam_cfg['color_width'], cam_cfg['color_height'], 
                rs.format.z16, cam_cfg['fps']
            )
            
            self.pipeline_dict[camera_name] = {
                'pipeline': pipeline,
                'config': config,
                'serial': serial
            }

    def start_capture(self):
        """开始采集"""
        if self.running:
            return
            
        if not self.pipeline_dict:
            self.setup_pipelines()
        
        for name, info in self.pipeline_dict.items():
            try:
                info['pipeline'].start(info['config'])
                self.logger.info(f"相机 {name} ({info['serial']}) 已启动")
            except Exception as e:
                self.logger.error(f"无法启动相机 {name}: {e}")
        
        self.running = True
        self.capture_thread = Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def _capture_loop(self):
        """内部采集循环"""
        while self.running:
            for name, info in self.pipeline_dict.items():
                try:
                    # 降低超时时间，避免单个相机卡顿阻塞全局
                    frames = info['pipeline'].wait_for_frames(timeout_ms=200)
                    
                    color_frame = frames.get_color_frame()
                    depth_frame = frames.get_depth_frame()
                    
                    if not color_frame or not depth_frame:
                        continue
                        
                    timestamp = time.time()
                    frame_data = {
                        'color': np.asanyarray(color_frame.get_data()),
                        'depth': np.asanyarray(depth_frame.get_data()),
                        'timestamp': timestamp,
                        'frame_number': color_frame.get_frame_number(),
                        'device_time': color_frame.timestamp
                    }
                    
                    # 线程安全更新队列
                    self.frame_queues[name].append(frame_data)
                    
                    with self.lock:
                        self.time_stamps[name].append({
                            'frame_number': color_frame.get_frame_number(),
                            'timestamp': timestamp,
                            'device_time': color_frame.timestamp
                        })
                        
                except Exception as e:
                    self.logger.debug(f"相机 {name} 获取帧失败: {e}")
                    continue
            
            # 避免空转占用过多 CPU
            time.sleep(0.001)

    def get_latest_frame(self, camera_name: str) -> Optional[Dict[str, Any]]:
        """获取最新一帧（不移除）"""
        queue = self.frame_queues.get(camera_name)
        if queue:
            try:
                return queue.pop()# [-1] # 获取最新但不弹出
            except IndexError:
                return None
        return None

    def get_all_latest_frames(self) -> Dict[str, Optional[Dict[str, Any]]]:
        """获取所有相机的最新帧"""
        return {name: self.get_latest_frame(name) for name in self.frame_queues}

    def get_timestamp_history(self, camera_name: str) -> List[Dict[str, Any]]:
        """获取时间戳历史"""
        history = self.time_stamps.get(camera_name)
        if history:
            with self.lock:
                return list(history)
        return []

    def stop_capture(self):
        """停止采集并释放资源"""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
            
        for name, info in self.pipeline_dict.items():
            try:
                info['pipeline'].stop()
                self.logger.info(f"相机 {name} 已停止")
            except Exception as e:
                self.logger.error(f"停止相机 {name} 失败: {e}")
        
        self.pipeline_dict.clear()

    def __enter__(self):
        self.start_capture()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_capture()

    def __del__(self):
        self.stop_capture()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    serial_to_name = {
            "245022302588": 'head_camera_image',
            "427622273979": 'left_wrist_image',
            "427622273394": 'right_wrist_image'
        }
    with MultiCameraManager() as cm:
        time.sleep(2)
        for _ in range(10):
            frames = cm.get_all_latest_frames()
            for name, data in frames.items():
                if data:
                    from PIL import Image
                    img = Image.fromarray(data['color'][:, :, ::-1])
                    img.save(f"examples/tron2/{name}.png")
                    print(f"{name}: #{data['frame_number']} @ {data['timestamp']:.3f}")
            time.sleep(0.01) 

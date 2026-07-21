"""
Tron2 多相机管理模块 —— RealSense 相机采集与管理

【模块定位】
提供多台 Intel RealSense 相机的并发采集、帧缓存和线程安全获取功能。
被 real_env.py 中的 Tron2Env 使用，作为机器人的"视觉传感器系统"。

【系统整体架构】
┌─────────────────────────────────────────────────────────────────┐
│                        Tron2Env                                  │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │   robot_utils.py (机器人控制)                             │   │
│  └───────────────────────────────────────────────────────────┘   │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │   realsense_image.py (多相机管理) ←── 当前模块             │   │
│  │                                                           │   │
│  │   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐ │   │
│  │   │ 头部摄像头    │   │ 左腕摄像头   │   │ 右腕摄像头   │ │   │
│  │   │(cam_high)    │   │(cam_left_wrist)│ │(cam_right_wrist)│ │
│  │   └──────────────┘   └──────────────┘   └──────────────┘ │   │
│  └───────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘

【相机硬件】
使用 Intel RealSense 深度相机（如 D415/D435），同时提供：
- RGB 彩色图像（640×480，30 FPS）
- 深度图像（640×480，30 FPS）

【多相机并发架构】
主线程（Tron2Env）               采集线程（_capture_loop）          RealSense 相机
    │                                  │                              │
    │───── get_all_latest_frames() ───→│                              │
    │                                  │─── wait_for_frames() ─────→│ 相机 1
    │                                  │←── color + depth ─────────│
    │                                  │─── wait_for_frames() ─────→│ 相机 2
    │                                  │←── color + depth ─────────│
    │                                  │─── wait_for_frames() ─────→│ 相机 3
    │                                  │←── color + depth ─────────│
    │←── 返回帧字典 ─────────────────│                              │
"""

import logging
import pyrealsense2 as rs
import numpy as np
import cv2
from threading import Thread, Lock
import time
from collections import deque
from typing import Dict, List, Optional, Tuple, Any


# ============================================================================
# 多相机管理器
# ============================================================================

class MultiCameraManager:
    """RealSense 多相机管理器 —— 同时管理多个 RealSense 相机的采集。

    【设计目标】
    1. 并发采集：用一个后台线程轮询所有相机，而不是每个相机一个线程
    2. 帧缓存：每个相机维护一个帧队列，防止偶尔的采集延迟导致数据丢失
    3. 线程安全：通过锁保护帧队列的读写，确保主线程和采集线程不冲突
    4. 名称抽象：通过序列号→名称的映射，用"逻辑名称"而不是序列号访问相机

    【工作流程】
    初始化 → detect_cameras() 检测已连接的相机
           → setup_pipelines() 为每个相机创建采集管道
           → start_capture() 启动后台采集线程
           → _capture_loop() 持续轮询每个相机的最新帧
           → frame_queues[name] 存储最近 N 帧
           → get_latest_frame() / get_all_latest_frames() 供外部读取

    【序列号→名称映射的重要性】
    RealSense 相机通过 USB 序列号唯一标识，每次物理连接不变。
    但序列号很难记忆（如 "245022302696"），所以映射为有意义的名称。
    这个映射需要根据实际硬件修改。

    Examples:
        >>> cm = MultiCameraManager()
        >>> cm.start_capture()
        >>> frames = cm.get_all_latest_frames()
        >>> for name, data in frames.items():
        ...     if data:
        ...         rgb_image = data['color'][:, :, ::-1]  # BGR→RGB
        >>> cm.stop_capture()
    """

    def __init__(
        self,
        max_queue_size: int = 10,
        serial_to_name: Optional[Dict[str, str]] = None,
        camera_configs: Optional[Dict[str, Dict[str, int]]] = None
    ):
        """初始化多相机管理器。

        Args:
            max_queue_size: 每个相机的帧队列最大长度。
                            决定了最多缓存多少帧历史。值越大，能容忍的读取延迟越高，
                            但内存占用也越大。10 帧 × 3 相机 × 0.92MB/帧 ≈ 28MB。
            serial_to_name: 相机序列号到逻辑名称的映射字典。
                            例如 {"245022302696": "head_camera_image"}
            camera_configs: 每个相机的详细配置（分辨率、帧率）。
                            {名称: {'color_width': 640, 'color_height': 480, 'fps': 30}}
        """
        self._setup_logger()

        # ---- 基础状态 ----
        self.max_queue_size = max_queue_size
        self.running = False  # 是否正在采集
        self.lock = Lock()    # 保护 time_stamps 的线程锁

        # ---- 相机序列号到名称的映射 ----
        # 【重要】使用前必须根据实际硬件的序列号修改！
        # 可以通过运行 detect_cameras() 获取当前连接的相机序列号。
        #
        # 相机布局：
        #   头部相机：安装在机器人头部，提供全局视角（cam_high）
        #   左腕相机：安装在左臂腕部，观察左手操作区域
        #   右腕相机：安装在右臂腕部，观察右手操作区域
        self.serial_to_name = serial_to_name or {
            "245022302696": 'head_camera_image',    # 头部相机
            "409122274385": 'left_wrist_image',      # 左腕相机
            "230322276915": 'right_wrist_image'      # 右腕相机
        }

        # ---- 默认相机配置 ----
        if camera_configs:
            self.camera_configs = camera_configs
        else:
            self.camera_configs = {
                'head_camera_image': {'color_width': 640, 'color_height': 480, 'fps': 30},
                'left_wrist_image': {'color_width': 640, 'color_height': 480, 'fps': 30},
                'right_wrist_image': {'color_width': 640, 'color_height': 480, 'fps': 30}
            }

        # ---- 采集管道字典 ----
        # key: 相机名称 (str)
        # value: {'pipeline': rs.pipeline, 'config': rs.config, 'serial': str}
        self.pipeline_dict = {}

        # ---- 帧队列 ----
        # key: 相机名称, value: deque of frame_data dicts
        # deque(maxlen) 是一个环形缓冲区，满了自动覆盖最旧的帧
        self.frame_queues = {
            name: deque(maxlen=max_queue_size) for name in self.camera_configs
        }

        # ---- 时间戳历史 ----
        # 用于调试和性能分析，记录最近 100 帧的时间戳
        self.time_stamps = {
            name: deque(maxlen=100) for name in self.camera_configs
        }

        # ---- 后台采集线程 ----
        self.capture_thread: Optional[Thread] = None

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]):
        """从配置字典创建管理器实例（工厂方法）。

        提供从字典（通常从 YAML 或 JSON 文件解析而来）初始化管理器的能力。
        这样可以通过配置文件而不是硬编码来管理相机设置。

        Args:
            config_dict: 配置字典，格式：
                {
                    'camera': {
                        'serial_to_name': {"序列号": "名称"},
                        'resolution': [480, 640],  # [H, W]
                        'fps': 30,
                        'camera_names': ["head_camera_image", ...],
                        'max_queue_size': 10
                    }
                }

        Returns:
            MultiCameraManager: 配置好的管理器实例
        """
        camera_cfg = config_dict.get('camera', {})

        # 提取序列号映射
        serial_to_name = camera_cfg.get('serial_to_name')

        # 构造相机配置
        res = camera_cfg.get('resolution', [640, 480])  # [H, W]
        fps = camera_cfg.get('fps', 30)

        camera_configs = {}
        for name in camera_cfg.get('camera_names', []):
            camera_configs[name] = {
                'color_width': res[1],   # W（注意：config里resolution是[H,W]）
                'color_height': res[0],  # H
                'fps': fps
            }

        return cls(
            max_queue_size=camera_cfg.get('max_queue_size', 10),
            serial_to_name=serial_to_name,
            camera_configs=camera_configs if camera_configs else None
        )

    def _setup_logger(self):
        """设置日志系统。"""
        self.logger = logging.getLogger("CameraManager")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    # ====================================================================
    # 相机发现与管道配置
    # ====================================================================

    def detect_cameras(self) -> List[str]:
        """检测当前连接的 RealSense 相机序列号列表。

        使用 pyrealsense2 的 context API 查询所有已连接的 RealSense 设备。
        过滤掉 'Asic'（ASIC 虚拟设备，不是物理相机）。

        Returns:
            List[str]: 检测到的物理相机序列号列表。
                       例如 ["245022302696", "409122274385", "230322276915"]

        Raises:
            RuntimeError: 如果未检测到任何 RealSense 相机。
        """
        ctx = rs.context()
        devices = ctx.query_devices()
        serials = [dev.get_info(rs.camera_info.serial_number) for dev in devices]
        # 过滤掉非物理设备
        # 'Asic' 是 RealSense SDK 内部用于 ASIC 编程的虚拟设备，不是真实相机
        return [s for s in serials if 'Asic' not in s]

    def setup_pipelines(self):
        """为每个检测到的相机创建并配置 RealSense 采集管道。

        【管道（Pipeline）是 RealSense SDK 的核心概念】
        pipeline 代表一条完整的图像采集流水线：
        传感器 → 硬件处理 → 固件处理 → 输出帧

        每个相机需要：
        1. 一个独立的 rs.pipeline() 实例
        2. 一个 rs.config() 配置，指定设备（序列号）、流类型、分辨率、帧率

        【流的类型】
        - rs.stream.color: RGB 彩色流，格式 bgr8（8位BGR，24位/像素）
        - rs.stream.depth: 深度流，格式 z16（16位深度值，单位毫米）

        【为什么同时配置深度流？】
        RealSense 是深度相机，深度和彩色图像是同步采集的。
        即使当前只需要 RGB 图像（策略输入只需要 RGB），
        也需要配置深度流来保持传感器的正常工作模式。

        Raises:
            RuntimeError: 未检测到任何 RealSense 相机
        """
        serial_numbers = self.detect_cameras()
        self.logger.info(f"检测到 {len(serial_numbers)} 个相机: {serial_numbers}")

        if not serial_numbers:
            raise RuntimeError("未检测到 RealSense 相机")

        for serial in serial_numbers:
            # 检查序列号是否在映射表中
            if serial not in self.serial_to_name:
                self.logger.warning(f"序列号 {serial} 未定义映射名称，跳过")
                continue

            camera_name = self.serial_to_name[serial]
            cam_cfg = self.camera_configs.get(
                camera_name,
                {'color_width': 640, 'color_height': 480, 'fps': 30}
            )

            # 为这个相机创建独立的管道和配置
            pipeline = rs.pipeline()
            config = rs.config()

            # enable_device(serial) 精确指定使用哪个物理相机
            # 如果不指定序列号，多个同型号相机会冲突
            config.enable_device(serial)

            # 【彩色流配置】
            # 格式 bgr8：OpenCV 的标准格式，可以直接用于 cv2 操作
            config.enable_stream(
                rs.stream.color,
                cam_cfg['color_width'], cam_cfg['color_height'],
                rs.format.bgr8, cam_cfg['fps']
            )

            # 【深度流配置】
            # 格式 z16：16 位无符号整数，每个像素值表示到物体的距离（毫米）
            # 0 表示无效测量（距离太远或太近）
            config.enable_stream(
                rs.stream.depth,
                cam_cfg['color_width'], cam_cfg['color_height'],
                rs.format.z16, cam_cfg['fps']
            )

            # 保存配置好的管道
            self.pipeline_dict[camera_name] = {
                'pipeline': pipeline,
                'config': config,
                'serial': serial
            }

    # ====================================================================
    # 采集控制
    # ====================================================================

    def start_capture(self):
        """启动所有相机的采集并开启后台采集线程。

        初始化流程：
        1. 如果尚未配置管道，自动调用 setup_pipelines()
        2. 启动所有相机的 pipeline.start()（每个相机开始传输帧）
        3. 启动后台采集线程 _capture_loop

        【为什么用单线程采集所有相机，而不是每个相机一个线程？】
        1. 减少线程数（3个相机 vs 4+个线程，减少上下文切换开销）
        2. 避免多线程同时调用 RealSense API 的潜在冲突
        3. 简化同步逻辑（单线程内帧之间天然有序）
        4. 降低 CPU 占用（每个线程都有固定开销）

        【缺点】
        单个相机异常（如 USB 断开）会阻塞所有相机的采集。
        所以 wait_for_frames() 的超时时间设得较短（200ms）。
        """
        if self.running:
            return

        if not self.pipeline_dict:
            self.setup_pipelines()

        # 启动所有相机的管道
        for name, info in self.pipeline_dict.items():
            try:
                info['pipeline'].start(info['config'])
                self.logger.info(f"相机 {name} ({info['serial']}) 已启动")
            except Exception as e:
                self.logger.error(f"无法启动相机 {name}: {e}")

        self.running = True

        # 启动后台采集线程（daemon=True：主线程退出时自动结束）
        self.capture_thread = Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def _capture_loop(self):
        """后台采集循环 —— 持续从所有相机获取最新帧。

        这是整个模块的核心循环，在后台线程中运行。
        每次循环会依次查询每个相机的最新帧，更新帧队列。

        【性能优化考虑】
        - wait_for_frames(timeout_ms=200)：每个相机最多等待 200ms
        - 三个相机串行查询，最坏情况延迟 = 3 × 200ms = 600ms
        - 但正常情况下每个相机大约 33ms 一帧（30 FPS），
          所以通常 < 100ms 就能拿到新帧
        - time.sleep(0.001)：防止空转时 100% 占用 CPU

        【帧数据格式】
        frame_data = {
            'color': np.ndarray,      # [H, W, 3] BGR 格式，uint8
            'depth': np.ndarray,      # [H, W] 深度图，uint16（单位：毫米）
            'timestamp': float,       # Python time.time() 时间戳（秒）
            'frame_number': int,      # 帧序号（从 0 开始递增）
            'device_time': float,     # RealSense 硬件时间戳（毫秒）
        }
        """
        while self.running:
            for name, info in self.pipeline_dict.items():
                try:
                    # 等待新帧到达（阻塞，但设了超时防止永久等待）
                    # timeout_ms=200：如果 200ms 内没有新帧，跳过这个相机
                    frames = info['pipeline'].wait_for_frames(timeout_ms=200)

                    # 从帧集合中提取彩色帧和深度帧
                    color_frame = frames.get_color_frame()
                    depth_frame = frames.get_depth_frame()

                    # 如果任一帧无效，跳过这次采集
                    if not color_frame or not depth_frame:
                        continue

                    # 记录采集时间戳（Python 侧的时间）
                    timestamp = time.time()

                    # 【核心操作：将 RealSense 帧转换为 numpy 数组】
                    # get_data() 返回原始字节缓冲区，
                    # np.asanyarray() 将其包装为 numpy 数组（零拷贝）
                    # 这样后续的图像处理（缩放、裁剪等）可以直接使用 numpy 操作
                    frame_data = {
                        'color': np.asanyarray(color_frame.get_data()),     # [H,W,3] BGR uint8
                        'depth': np.asanyarray(depth_frame.get_data()),     # [H,W] uint16，单位毫米
                        'timestamp': timestamp,                             # Python 侧时间戳
                        'frame_number': color_frame.get_frame_number(),     # 帧序号
                        'device_time': color_frame.timestamp                # 硬件时间戳（毫秒）
                    }

                    # 将新帧推入队列（deque 满了自动丢弃最旧的）
                    self.frame_queues[name].append(frame_data)

                    # 记录时间戳历史（线程安全，用于调试）
                    with self.lock:
                        self.time_stamps[name].append({
                            'frame_number': color_frame.get_frame_number(),
                            'timestamp': timestamp,
                            'device_time': color_frame.timestamp
                        })

                except Exception as e:
                    # wait_for_frames() 超时是很常见的（特别是在启动阶段），
                    # 所以用 debug 级别记录，避免刷屏
                    self.logger.debug(f"相机 {name} 获取帧失败: {e}")
                    continue

            # 防止空转时 100% CPU 占用
            # 1ms 的 sleep 足够让出 CPU，同时保持低延迟
            time.sleep(0.001)

    # ====================================================================
    # 帧获取接口
    # ====================================================================

    def get_latest_frame(self, camera_name: str) -> Optional[Dict[str, Any]]:
        """获取指定相机的最新一帧（取出后从队列移除）。

        使用 pop() 而不是 [-1] 索引：
        - pop(): 取出并移除，确保下次调用返回更新的帧
        - [-1] 索引: 只读不移除，如果外部读取频率高于采集频率，
                     会反复读到同一帧（导致动作重复执行）

        Args:
            camera_name: 相机名称（如 "head_camera_image"）

        Returns:
            Optional[Dict]: 帧数据字典，如果队列为空返回 None。
                            {
                                'color': np.ndarray,    # [H,W,3] BGR
                                'depth': np.ndarray,    # [H,W] 深度
                                'timestamp': float,      # 采集时间
                                'frame_number': int,     # 帧序号
                                'device_time': float     # 硬件时间
                            }
        """
        queue = self.frame_queues.get(camera_name)
        if queue:
            try:
                return queue.pop()  # LIFO：取最新的一帧
            except IndexError:
                return None
        return None

    def get_all_latest_frames(self) -> Dict[str, Optional[Dict[str, Any]]]:
        """获取所有相机的最新帧。

        这是 Tron2Env.get_obs() 调用的主要接口，
        一次性获取机器人全部视角的当前图像。

        Returns:
            Dict[str, Optional[Dict]]: 相机名称→帧数据的字典。
                帧数据可能为 None（如果相机尚未准备好）。
        """
        return {name: self.get_latest_frame(name) for name in self.frame_queues}

    def get_timestamp_history(self, camera_name: str) -> List[Dict[str, Any]]:
        """获取指定相机的时间戳历史（用于调试和性能分析）。

        返回最近 100 帧的时间戳记录，可以用于：
        - 检查帧率是否稳定（应该接近配置的 FPS）
        - 检测帧间隔是否有异常抖动
        - 比较硬件时间戳和软件时间戳的延迟

        Args:
            camera_name: 相机名称

        Returns:
            List[Dict]: 时间戳历史列表，按时间顺序排列。
        """
        history = self.time_stamps.get(camera_name)
        if history:
            with self.lock:
                return list(history)
        return []

    # ====================================================================
    # 资源管理
    # ====================================================================

    def stop_capture(self):
        """停止采集并释放所有相机资源。

        安全关闭流程：
        1. 设置 running = False，通知采集线程退出循环
        2. 等待采集线程结束（最多 1 秒）
        3. 停止所有相机管道（释放 RealSense 驱动资源）
        4. 清空管道字典

        【为什么需要等待采集线程？】
        如果不等采集线程结束就直接关闭管道，
        采集线程可能还在调用 wait_for_frames()，
        导致管道关闭后出现异常。
        """
        self.running = False

        # 等待采集线程结束
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)

        # 停止所有相机管道
        # 注意：即使一个相机停止失败，也要继续尝试停止其他相机
        for name, info in self.pipeline_dict.items():
            try:
                info['pipeline'].stop()
                self.logger.info(f"相机 {name} 已停止")
            except Exception as e:
                self.logger.error(f"停止相机 {name} 失败: {e}")

        self.pipeline_dict.clear()

    # ---- 上下文管理器支持 ----
    # 使用 with 语句可以自动管理相机的启动和停止：
    #   with MultiCameraManager() as cm:
    #       frames = cm.get_all_latest_frames()
    # 退出 with 块时自动调用 stop_capture()

    def __enter__(self):
        """上下文管理器入口 —— 进入 with 块时启动采集。"""
        self.start_capture()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出 —— 离开 with 块时停止采集。"""
        self.stop_capture()

    def __del__(self):
        """析构函数 —— 对象被回收时自动停止采集。

        这是一个"安全网"，防止用户忘记显式调用 stop_capture()。
        但不应该依赖它来代替显式资源管理，
        因为 Python 的垃圾回收时机是不确定的。
        """
        self.stop_capture()


# ============================================================================
# 独立运行测试
# ============================================================================

if __name__ == "__main__":
    """独立运行此文件时，执行相机采集测试。

    测试流程：
    1. 使用上下文管理器连接所有相机
    2. 等待 2 秒让相机预热
    3. 采集 10 帧，每帧保存为 PNG 图像
    4. 打印帧序号和时间戳

    运行方式：
        python examples/tron2/realsense_image.py
    """
    logging.basicConfig(level=logging.INFO)

    # 【注意】测试时可能需要根据实际硬件修改序列号！
    serial_to_name = {
        "245022302588": 'head_camera_image',     # 头部相机
        "427622273979": 'left_wrist_image',       # 左腕相机
        "427622273394": 'right_wrist_image'       # 右腕相机
    }

    with MultiCameraManager(serial_to_name=serial_to_name) as cm:
        # 等待相机预热（自动曝光收敛）
        time.sleep(2)

        # 采集 10 帧并保存
        for _ in range(10):
            frames = cm.get_all_latest_frames()
            for name, data in frames.items():
                if data:
                    from PIL import Image
                    # RealSense 输出的是 BGR 格式，
                    # 保存前需要转换为 RGB（PIL/PNG 使用 RGB）
                    img = Image.fromarray(data['color'][:, :, ::-1])
                    img.save(f"examples/tron2/{name}.png")
                    print(f"{name}: #{data['frame_number']} @ {data['timestamp']:.3f}")

            time.sleep(0.01)

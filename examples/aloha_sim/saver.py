"""
ALOHA 仿真结果保存模块

本模块定义了 VideoSaver 类，用于在每个仿真 episode 结束时将过程录制为视频文件。
它作为 Runtime 框架的"订阅者"（Subscriber），在每一个时间步被回调，
收集渲染图像，并在 episode 结束时合成为 MP4 视频。

工作机制：
  1. Runtime 主循环每步调用 on_step() → 保存当前帧图像
  2. Runtime 在 episode 开始/结束时调用 on_episode_start()/on_episode_end()
  3. on_episode_end() 中将收集的所有帧合成为 MP4 视频文件

输出格式：
  - 视频文件: data/aloha_sim/videos/out_0.mp4, out_1.mp4, ...
  - 帧率: 50 FPS（与 ALOHA 仿真环境的默认控制频率一致）
  - 图像: RGB 彩色，由 224x224 的观测图像组成
"""

import logging  # 日志系统，用于输出保存进度信息
import pathlib  # 跨平台路径处理，用于管理输出目录和文件命名

import imageio  # 视频/图像读写库，用于将帧序列合成为 MP4 文件
import numpy as np  # 数值计算库，用于图像数据格式转换
from openpi_client.runtime import subscriber as _subscriber  # openpi 运行时框架的订阅者接口
from typing_extensions import override  # 类型提示：显式标记方法覆盖


class VideoSaver(_subscriber.Subscriber):
    """视频录制器：将每次推理运行的仿真过程保存为 MP4 视频文件。

    这个类实现了 Subscriber 接口，因此可以被 Runtime 主循环集成。
    Subscriber 接口定义了三个回调方法，分别在 episode 的三个阶段被调用：

      Runtime 主循环:
          ┌─────────────────────────────────────────────────┐
          │  on_episode_start() → 清空上一轮的帧缓存        │
          │                                                 │
          │  循环每一步:                                     │
          │    on_step() → 收集当前帧图像 → 追加到缓存列表  │
          │                                                 │
          │  on_episode_end() → 将帧缓存写入 MP4 文件       │
          └─────────────────────────────────────────────────┘

    关键设计点：
      - 只保存图像（cam_high 摄像头），不保存动作或状态数据
      - 通过 subsample 参数可以跳帧保存（减少视频文件大小）
      - 视频文件名自动递增（out_0.mp4, out_1.mp4, ...），避免覆盖
    """

    def __init__(self, out_dir: pathlib.Path, subsample: int = 1) -> None:
        """初始化视频保存器。

        Args:
            out_dir:    视频输出目录。如果目录不存在会自动创建。
            subsample:  帧采样间隔。每 subsample 帧保存一帧。
                        - 1（默认）: 保存所有帧（最完整，文件也最大）
                        - 2: 每隔一帧保存一帧（文件大小减半，帧率减半）
                        - 3: 每隔两帧保存一帧，以此类推
                        适用于：视频太大时减少文件大小，或仿真很长时生成加速回放。
        """
        # 确保输出目录存在（parents=True 会递归创建父目录）
        out_dir.mkdir(parents=True, exist_ok=True)
        self._out_dir = out_dir
        self._images: list[np.ndarray] = []  # 当前 episode 的帧缓存列表
        self._subsample = subsample  # 帧采样间隔

    @override
    def on_episode_start(self) -> None:
        """新 episode 开始时的回调。

        Runtime 在调用 environment.reset() 之后、进入主循环之前调用此方法。
        我们需要清空上一轮的帧缓存，为新的 episode 做好准备。

        注意：
          - 如果上一轮视频还没保存（理论上不会发生，因为 on_episode_end()
            负责保存），这些帧会被丢弃。
          - 在第一个 episode 开始前，_images 就是空的，所以这个调用是安全的。
        """
        self._images = []  # 清空帧缓存，准备录制新的 episode

    @override
    def on_step(self, observation: dict, action: dict) -> None:
        """每一步的回调：从观测中提取图像并缓存。

        Runtime 在每次 environment.apply_action() 之后调用此方法。
        我们接收当前观测，提取摄像头图像，保存到帧缓存列表。

        Args:
            observation: 当前观测字典。包含:
                         - "images": {"cam_high": 图像数组}
                           图像格式: [C, H, W]（通道优先），uint8，值范围 [0,255]
            action:      当前步执行的动作字典（在这个场景中我们不需要它，
                         但 Subscriber 接口要求接收这个参数）。
        """
        # ====================================================================
        # 提取图像
        #
        # observation["images"]["cam_high"] 是来自环境的观测图像。
        # 在 env.py 的 _convert_observation() 中，图像被转换为
        # [C, H, W] 格式（C=3 通道，H=224 高度，W=224 宽度）。
        #
        # 但是 imageio.mimwrite()（用于合成视频）期望的图像格式是
        # [H, W, C]（高度优先，即 NumPy/PIL 的标准格式）。
        # 所以这里需要做轴序转换。
        # ====================================================================

        # 从观测中取出顶部摄像头图像，形状为 [3, 224, 224]（[C, H, W]）
        im = observation["images"]["cam_high"]  # [C, H, W]

        # 转置为 imageio 期望的 [H, W, C] 格式
        # transpose(1, 2, 0) 的含义：
        #   原形状 [C, H, W] 的索引 0=C, 1=H, 2=W
        #   新形状 [H, W, C] 的索引 0=H(原1), 1=W(原2), 2=C(原0)
        # 所以 transpose(1, 2, 0) = 按 (H, W, C) 的顺序重排维度
        im = np.transpose(im, (1, 2, 0))  # [H, W, C]

        # 添加到当前 episode 的帧缓存列表
        # 注意：这里只是把引用加入列表，没有复制图像数据。
        # 因为 next step 的观测是全新的数组（env.py 中每次 step 都创建新 dict），
        # 所以不存在数据被覆盖的风险。
        self._images.append(im)

    @override
    def on_episode_end(self) -> None:
        """Episode 结束时的回调：将缓存的帧合成为 MP4 视频文件。

        Runtime 在 episode 完成（成功、失败或超时）之后调用此方法。
        我们负责将 on_step() 中收集的所有帧写入视频文件。

        文件命名规则：
          - 扫描输出目录中已有的 out_N.mp4 文件
          - 取最大的 N，加 1 作为新文件的编号
          - 例如已有 out_0.mp4, out_1.mp4，新文件为 out_2.mp4

        视频参数：
          - 编码器：imageio 默认的 MP4 编码器（通常是 libx264 或 ffmpeg）
          - 帧率：50 / subsample（因为跳帧后播放速度会变快，
            需要相应降低帧率来保持实际播放时长一致）
          - 图像尺寸：自动从第一帧形状确定（224x224）
        """
        # ====================================================================
        # 步骤 1：确定新文件的编号
        #
        # 扫描目录中所有符合 out_N.mp4 模式的文件，取最小编号 + 1。
        # 这样即使删除了一些旧视频，编号也会继续递增（不会复用已删除的编号）。
        #
        # 路径解析示例：
        #   p = pathlib.Path("data/aloha_sim/videos/out_3.mp4")
        #   p.stem  → "out_3"
        #   p.stem.split("_") → ["out", "3"]
        #   int("3") → 3
        # ====================================================================
        existing = list(self._out_dir.glob("out_[0-9]*.mp4"))
        next_idx = max([int(p.stem.split("_")[1]) for p in existing], default=-1) + 1
        out_path = self._out_dir / f"out_{next_idx}.mp4"

        # ====================================================================
        # 步骤 2：合成视频
        #
        # imageio.mimwrite() 是 imageio 库中"将多帧写入视频文件"的函数。
        # 参数说明：
        #   - out_path:    输出文件路径（MP4 格式）
        #   - pix:         帧列表，每个元素是 [H, W, C] 的 uint8 图像数组
        #   - fps:         视频帧率（frames per second）
        #
        # 帧率计算：
        #   仿真环境以 50Hz 运行（由 Runtime 的 max_hz=50 控制），
        #   所以默认帧率为 50 FPS。
        #   如果 subsample > 1，我们跳过了部分帧，
        #   相应地降低帧率以保持视频时长与实际仿真时长一致。
        #   例如 subsample=2：每隔一帧保存一帧，帧率 = 50/2 = 25 FPS
        # ====================================================================

        # 应用帧采样：只取 [::subsample] 索引的帧
        # 例如 subsample=2 → 取第 0, 2, 4, 6, ... 帧
        # 注意：这里对每个元素调用 np.asarray() 确保类型正确，
        # 虽然 _images 中的元素应该已经是 ndarray，但 imageio 要求显式转换。
        frames = [np.asarray(x) for x in self._images[:: self._subsample]]

        logging.info(f"Saving video to {out_path}")
        imageio.mimwrite(
            out_path,
            frames,
            fps=50 // max(1, self._subsample),  # 调整帧率保持播放时长一致
        )

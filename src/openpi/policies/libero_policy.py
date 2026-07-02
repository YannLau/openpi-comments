"""
========================================================================
  LIBERO 机器人平台策略（Policy）—— 输入/输出数据变换
========================================================================

什么是 LIBERO？
  LIBERO 是一个开源的机器人操作基准测试平台，包含一个带双机械臂和夹爪的
  桌面机器人平台，用于学习各种日常操作任务（如"打开抽屉"、"拿起红色方块"）。

本文件的作用：
  作为 LIBERO 平台与 π₀（通用机器人模型）之间的"适配器"。它负责：
    1. 输入变换（LiberoInputs）：将 LIBERO 数据集的原始数据格式
       （图像、状态、动作、语言指令）转换为 π₀ 模型期望的统一格式。
    2. 输出变换（LiberoOutputs）：将 π₀ 模型的输出（预测的动作）
       裁剪到 LIBERO 机器人实际的动作维度。

如果你要将 openpi 适配到自己的机器人平台，可以复制这个文件并将其中
  具体的键名和维度替换为你自己的数据集格式。

相关阅读：
  - aloha_policy.py：ALOHA 机器人平台的类似适配器（双机械臂夹爪）
  - droid_policy.py：DROID 机器人平台的类似适配器
  - transforms.py：通用的数据变换工具（归一化、重排、分词等）
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_libero_example() -> dict:
    """创建一个随机的 LIBERO 测试示例，用于验证或调试。

    这个函数模拟了一个 LIBERO 数据集的典型数据点，包含：
      - observation/state：机械臂状态（8 维，关节位置/速度等）
      - observation/image：主摄像头图像（224x224 彩色图）
      - observation/wrist_image：腕部摄像头图像（224x224 彩色图）
      - prompt：自然语言指令

    为什么会用到这个函数？
      在开发和调试策略时，有时需要一个"假数据"来测试数据流是否正常，
      而不需要加载真实数据集。这个函数就是用来生成这种测试数据的。

    Returns:
        一个字典，包含与真实 LIBERO 数据具有相同键和形状的随机数据。
        图像的像素值在 [0, 255] 范围内，数据类型为 uint8。
    """
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    """图像预处理工具函数：将各种格式的输入图像统一为 uint8 (H, W, C) 格式。

    为什么要处理不同格式？
      在不同的数据流程中，图像可能有不同的表示方式：
        1. 训练时（LeRobot 数据集）：图像用 float32 存储，形状为 (C, H, W)
           这是深度学习框架中常见的"通道优先"格式。
        2. 推理时（策略服务）：图像可能已经是 uint8，形状为 (H, W, C)
           这是常规图像文件的"通道最后"格式。
      这个函数将两种格式都统一到 (H, W, C) uint8，方便 π₀ 模型处理。

    Args:
        image: 输入图像，可以是：
              - uint8 格式，(H, W, C) 形状 → 直接返回
              - float32 格式，(C, H, W) 形状 → 乘以 255 转为 uint8，再调转轴

    Returns:
        统一为 uint8 类型、(H, W, C) 形状的图像数组。
    """
    image = np.asarray(image)

    # 情况1：如果图像是浮点类型（通常是 LeRobot 数据集的范围 [0, 1] float32）
    # 将其映射回 [0, 255] 的无符号整数范围
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)

    # 情况2：如果图像是"通道优先"格式 (C, H, W)
    # 使用 einops.rearrange 调整轴顺序为 (H, W, C)
    # einops 是一个简洁的张量操作库，rearrange 用于重排维度
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")

    return image


@dataclasses.dataclass(frozen=True)
class LiberoInputs(transforms.DataTransformFn):
    """LIBERO 输入变换：将 LIBERO 数据集格式转换为 π₀ 模型的统一输入格式。

    这个类实现了 DataTransformFn 协议（参考 transforms.py），可以像函数一样调用。

    它的工作就像"适配器模式"中的一个适配器：
      左侧（LIBERO 格式）：
        - observation/state:    (8,) float32
        - observation/image:    (H, W, 3) uint8
        - observation/wrist_image: (H, W, 3) uint8
        - actions:              (T, 7) float32  （训练时存在）
        - prompt:               str

      右侧（π₀ 模型统一格式）：
        - state:                (8,) float32
        - image:                {"base_0_rgb": ..., "left_wrist_0_rgb": ..., "right_wrist_0_rgb": ...}
        - image_mask:           {"base_0_rgb": True, ...}  （哪些图像是真实存在的）
        - actions:              (T, 7) float32  （仅训练时）
        - prompt:               str

    关键设计说明：
      π₀ 模型支持最多三路图像输入：
        1. base_0_rgb —— 主视角（第三人称视角）
        2. left_wrist_0_rgb —— 左腕部视角
        3. right_wrist_0_rgb —— 右腕部视角
      对于只有两个摄像头的 LIBERO，我们将右手腕图像用零填充，
      并通过 image_mask 告知模型"这是一个填充图像，不要使用"。
      这是为了使模型能兼容不同摄像头配置的机器人平台。

    如果你想适配自己的数据集：
      1. 复制这个类
      2. 修改数据读取的键名（如将 "observation/image" 改为你数据集的键名）
      3. 调整图像数量（如果多于/少于三个摄像头）
      4. 注意 image_mask 的处理方式（π₀ 和 π₀-FAST 模型不同）
    """

    # 决定使用的模型类型。这会影响 image_mask 的处理方式。
    # 例如：π₀-FAST 模型对填充图像的掩码处理与 π₀ 不同。
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        """执行输入变换。

        Args:
            data: LIBERO 数据集的一个数据点。字典结构如下（键名可能因具体数据集而异）：
                - "observation/state": 机器人状态向量 (8,)
                - "observation/image": 主摄像头图像 (H, W, 3) 或 (3, H, W)
                - "observation/wrist_image": 腕部图像 (H, W, 3) 或 (3, H, W)
                - "actions": (可选) 动作序列 (T, 7)，仅在训练时存在
                - "prompt": (可选) 自然语言指令字符串

        Returns:
            模型统一格式的字典，键名不可修改（这是模型期望的输入接口）。
        """
        # 步骤1：图像预处理
        # ---------------
        # 由于 LeRobot 数据集在存储时会将图像自动转为 float32 (C, H, W) 格式，
        # 但策略推理时跳过了这一步，所以我们需要手动处理。
        #
        # 如果你有自己的数据集：
        #   - 如果图像键名不同，修改下面两行的键名
        #   - 如果没有腕部图像，可以注释掉并将其替换为 base_image 的零数组
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # 步骤2：组装模型输入字典
        # ---------------
        # 注意：下面字典中的键名是 π₀ 模型的统一接口，不要修改！
        # 这些键名在模型的 forward 方法中被直接引用。
        inputs = {
            # 机器人状态（关节位置、速度等）
            "state": data["observation/state"],

            # 图像字典：包含所有摄像头图像
            "image": {
                # 主视角（第三人称摄像头）
                "base_0_rgb": base_image,

                # 左腕部摄像头
                "left_wrist_0_rgb": wrist_image,

                # 右腕部摄像头——LIBERO 没有右腕，用零填充
                # 这是为了保持模型输入的统一性（三个图像入口）
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },

            # 图像掩码字典：标记哪些图像是真实存在的（True）或填充的（False）
            # 模型会使用 mask 来忽略填充图像的计算
            "image_mask": {
                "base_0_rgb": np.True_,      # 主摄像头：真实存在
                "left_wrist_0_rgb": np.True_,  # 左腕：真实存在
                # 右腕：π₀-FAST 模型需要对填充图像设 True 掩码以防止信息泄漏，
                #       π₀ 模型则设 False 忽略它。
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # 步骤3：动作数据（仅训练时存在）
        # ---------------
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # 步骤4：语言指令
        # ---------------
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoOutputs(transforms.DataTransformFn):
    """LIBERO 输出变换：将 π₀ 模型的输出转换为 LIBERO 平台的格式。

    这个变换通常在推理阶段使用。

    为什么要裁剪维度？
      π₀ 模型在训练时会将动作填充到 model_action_dim（例如 32 维）。
      但 LIBERO 机器人只使用前 7 维来控制其 7 个自由度（或 8 维，含夹爪）。
      多余的维度只是填充的零值，没有实际意义，需要去除。

    例如：
      模型预测的动作 (32,) → 裁剪前 7 维 → (7,) LIBERO 实际动作
    """

    def __call__(self, data: dict) -> dict:
        """执行输出变换。

        Args:
            data: 模型的输出字典，包含 "actions" 键。
                  actions 的形状可能是 (T, model_action_dim) 或 (model_action_dim,)，
                  其中 model_action_dim 可能为 32 等。

        Returns:
            只包含实际动作维度的字典：
                {"actions": np.array 形状 (T, 7) 或 (7,)}

            对你自己的数据集：
              将 `7` 替换为你的机器人的实际动作维度。
        """
        # ... 表示所有前导维度（包括可能的批处理维度和时间步维度），
        # :7 表示只取前 7 个动作维度。
        # 对于 LIBERO 机器人，动作空间是 7 维的（6 维位姿 + 1 维夹爪）。
        return {"actions": np.asarray(data["actions"][..., :7])}

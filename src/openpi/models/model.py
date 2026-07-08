"""
openpi 模型基础模块。

本模块定义了所有模型架构共用的基础数据结构和抽象接口，包括：
- 观察数据格式（Observation）
- 模型配置基类（BaseModelConfig）
- 模型实现基类（BaseModel）
- 数据预处理函数（preprocess_observation）
- 检查点参数恢复函数（restore_params）
"""

import abc  # 抽象基类，用于定义抽象方法和接口
from collections.abc import Sequence
import dataclasses  # 数据类装饰器，用于定义简单的数据容器
import enum  # 枚举类型
import logging  # 日志模块
import pathlib
from typing import Generic, TypeVar

import augmax  # 数据增强库（图像随机裁剪、旋转、颜色抖动等）
from flax import nnx  # Flax NNX：JAX 的神经网络库（新版 API）
from flax import struct  # Flax 的 struct 装饰器，支持 PyTree 语义的数据类
from flax import traverse_util  # Flax 的工具函数，用于展平/还原嵌套字典
import jax  # JAX：Google 的高性能数值计算框架
import jax.numpy as jnp  # JAX 的 NumPy 接口
import numpy as np
import orbax.checkpoint as ocp  # Orbax：JAX 的检查点管理库
import safetensors  # 安全的张量序列化格式
import torch  # PyTorch：用于加载 PyTorch 格式的检查点

from openpi.models_pytorch import pi0_pytorch  # PyTorch 版本的 π₀ 模型
from openpi.shared import image_tools  # 图像工具（resize 等）
import openpi.shared.array_typing as at  # 数组类型注解工具

logger = logging.getLogger("openpi")

# ---------------------------------------------------------------------------
# 类型变量
# ---------------------------------------------------------------------------
# TypeVar 用于泛型，这里的 ArrayT 可以是 JAX 数组、PyTorch 张量或 NumPy 数组中的任意一种。
# 这使得 Observation 和 Actions 等数据结构可以同时支持不同的后端。
ArrayT = TypeVar("ArrayT", bound=jax.Array | torch.Tensor | np.ndarray)


# ---------------------------------------------------------------------------
# 模型类型枚举
# ---------------------------------------------------------------------------
class ModelType(enum.Enum):
    """支持的模型类型枚举。

    目前支持三种模型架构：
    - PI0: π₀，基于流匹配（flow matching）的 VLA 模型
    - PI0_FAST: π₀-FAST，自回归版本的 π₀（生成速度更快）
    - PI05: π₀.₅，升级版流匹配模型，加入了"知识隔离"（knowledge insulation）机制
    """

    PI0 = "pi0"
    PI0_FAST = "pi0_fast"
    PI05 = "pi05"
    DUMMY = "dummy"  # 新增-简单的MLP测试模型


# ---------------------------------------------------------------------------
# 模型输入常量
# ---------------------------------------------------------------------------

# 模型始终期望接收以下三个视角的摄像头图像：
# - base_0_rgb: 基础视角（通常是机器人本体上的某个固定视角）
# - left_wrist_0_rgb: 左腕部摄像头
# - right_wrist_0_rgb: 右腕部摄像头
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)

# 图像输入分辨率。所有输入图像都会被缩放到此尺寸。
# 如果未来发布小模型，这个值可能需要调整。
IMAGE_RESOLUTION = (224, 224)


# ---------------------------------------------------------------------------
# 数据格式说明
# ---------------------------------------------------------------------------
#
# 数据流水线（data pipeline）处理过程：
# 1. 数据变换（transforms）将原始数据集中的样本转换成嵌套字典格式
# 2. 嵌套字典随后被转换为结构化的 Observation 和 Actions 对象
#
# 嵌套字典的格式如下：
# {
#     # ---- 观测数据 ----
#     "image": {
#         "base_0_rgb": (float32|uint8)[*b, h, w, 3],  # RGB 图像，像素值在 [-1, 1] 或 [0, 255]
#         ...  # 其他摄像头视角
#     },
#     "image_mask": {
#         "base_0_rgb": bool[*b],  # 图像是否有效（True = 有效）
#         ...  # 其他视角的掩码
#     },
#     "state": float32[*b, s],  # 低维机器人状态（如关节角度、末端执行器位姿等）
#     "tokenized_prompt": int32[*b, l],  # （可选）经过标记化的语言指令
#     "tokenized_prompt_mask": bool[*b, l],  # （可选）标记化指令的掩码
#     "token_ar_mask": int32[*b, l],  # （可选，仅 FAST 模型使用）自回归掩码
#     "token_loss_mask": bool[*b, l],  # （可选，仅 FAST 模型使用）损失计算掩码
#
#     # ---- 动作数据 ----
#     "actions": float32[*b ah ad]  # 动作序列
# }
#
# 其中：
#   *b = 批次维度（batch dimensions）   *b 中的星号 * 表示这个维度是可选的或可变的（比如训练时是 2，推理时是 1，不影响校验）。
#   h, w = 图像高度/宽度
#   s = 状态向量维度
#   l = 序列长度（token 数量）
#   ah = 动作时间步数（action horizon）
#   ad = 动作空间维度（action dimension）
#


# ---------------------------------------------------------------------------
# Observation — 观测数据结构
# ---------------------------------------------------------------------------
@at.typecheck  # 运行时类型检查装饰器
@struct.dataclass  # Flax 的 struct.dataclass，使类具有 PyTree 语义（可被 JAX 的 tree 操作函数处理）
class Observation(Generic[ArrayT]):
    """观测数据 —— 模型的输入。

    包含视觉信息（多视角图像）、机器人状态以及可选的文本指令。
    数据变换流水线的输出最终会被转换为这个格式。

    使用泛型参数 ArrayT 来支持不同的数组后端（jax.Array / torch.Tensor / np.ndarray）。
    """

    # images 是一个字典。这个字典的键是字符串（摄像头名字），值是一个浮点数张量。
    # 这个张量的形状是 [批次大小, 高度, 宽度, 通道数]，并且这个张量使用的数组类型（JAX/Torch/Numpy）由外层的 ArrayT 决定。
    # *b 中的星号 * 表示这个维度是可选的或可变的（比如训练时是 2，推理时是 1，不影响校验）。

    # 多视角图像，像素值范围为 [-1, 1]，float32 类型。
    # 键名对应于摄像头视角名称（如 "base_0_rgb"），值是一个批次的图像张量。
    images: dict[str, at.Float[ArrayT, "*b h w c"]]

    # 图像掩码，与 images 字典的键一一对应。
    # True 表示该图像有效；False 表示该图像丢失或无效。
    image_masks: dict[str, at.Bool[ArrayT, "*b"]]

    # 低维机器人状态（如关节角度、末端执行器位置等连续值）。
    state: at.Float[ArrayT, "*b s"]

    # --- 语言指令（可选）---
    # 经过 tokenizer 处理后的文本指令（整数 token ID 序列）。
    tokenized_prompt: at.Int[ArrayT, "*b l"] | None = None
    # 对应的 attention mask，标记哪些 token 是有效内容、哪些是 padding。
    tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None

    # --- π₀-FAST 模型专用字段 ---
    # Fast 模型是自回归架构，需要以下额外信息：

    # 自回归掩码：控制哪些位置参与自回归预测。
    token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    # 损失掩码：控制哪些位置的预测参与损失计算。
    token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None

    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "Observation[ArrayT]":
        """从嵌套字典创建 Observation 对象。

        这个方法是数据流水线的关键接口，它定义了非结构化的嵌套字典到结构化 Observation 的映射规则。
        数据变换的输出先表现为嵌套字典，然后通过此方法转换为 Observation。

        主要处理逻辑：
        1. 验证 tokenized_prompt 和 tokenized_prompt_mask 同时存在或同时不存在
        2. 如果输入图像是 uint8 类型（常见于原始数据），自动转换为 [-1, 1] 范围的 float32
        3. 处理 PyTorch 张量的特殊维度排列（PyTorch 使用 NCHW 格式）
        """

        """
        1. 字母分别代表什么？
        在深度学习张量中，图像数据的维度通常用这四个字母表示：

        N (Batch)：批次大小（几张图一起算）。

        C (Channel)：通道数（RGB 彩色图为 3，灰度图为 1）。

        H (Height)：图像高度（像素行数）。

        W (Width)：图像宽度（像素列数）。

        所谓“格式”，就是这四个维度排列的先后顺序。

        2. 主要的图像数据格式对比
        除了 PyTorch 默认的 NCHW，业内还有以下几种主流格式：

        格式	排列顺序	典型框架/硬件	特点
        NCHW	[N, C, H, W]	PyTorch、MXNet、Caffe	通道优先。同一通道的像素在内存中连续存储。在早期的 cuDNN 库上，卷积运算速度通常更快。
        NHWC	[N, H, W, C]	JAX (Flax)、TensorFlow (默认)、Keras	通道最后。同一位置的像素（RGB值）在内存中连续存储。对 CPU 更友好，且更适配 NVIDIA Tensor Core 的混合精度训练。
        CHWN	[C, H, W, N]	极少数高性能推理后端	将通道放在最前，批次放在最后，用于特定内存对齐优化。

        """

        # 校验：tokenized_prompt 和 tokenized_prompt_mask 必须成对出现
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")

        # 处理图像数据：如果图像是 uint8 格式，转换为 [-1, 1] 的 float32
        for key in data["image"]:
            if data["image"][key].dtype == np.uint8:
                # NumPy/JAX 格式：像素值从 [0, 255] → [0, 1] → [-1, 1]
                data["image"][key] = data["image"][key].astype(np.float32) / 255.0 * 2.0 - 1.0
            elif hasattr(data["image"][key], "dtype") and data["image"][key].dtype == torch.uint8:
                # PyTorch 格式：除了数值转换外，还需要将维度从 BHWC 置换为 BCHW
                # 因为 PyTorch 模型（如 siglip）期望 NCHW 格式
                data["image"][key] = data["image"][key].to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0

        return cls(
            images=data["image"],
            image_masks=data["image_mask"],
            state=data["state"],
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
        )

    def to_dict(self) -> at.PyTree[ArrayT]:
        """将 Observation 转换为嵌套字典格式。

        这是 from_dict 的逆操作，用于将结构化的 Observation 转回嵌套字典，
        方便进行序列化或与其他 JAX 函数交互。
        """
        result = dataclasses.asdict(self)
        # 注意这里将 images/masks 重新命名为 image/image_mask，
        # 以匹配数据流水线中使用的字典键名约定。
        result["image"] = result.pop("images")
        result["image_mask"] = result.pop("image_masks")
        return result


"""
在 Python 中，`cls` 是 **类方法（`@classmethod`）** 的第一个参数的**约定名称**（类似于实例方法中用 `self`）。

针对你这段代码，最直接的回答是：

**`cls` 指代的是 `Observation` 这个类本身（而不是该类的某个实例对象）。**

为了帮你彻底理解，我们把它拆成三点来看：

### 1. 谁把 `cls` 传进去的？
        因为方法上面有 `@classmethod` 装饰器，Python 解释器在调用 `Observation.from_dict(...)` 时，**不会**传入实例（因为还没创建），而是**自动把 `Observation` 这个类作为第一个参数传给 `cls`**。

### 2. 它和 `self` 有什么区别？
        - **`self`**：指向一个**已经创建好的具体对象**。你可以通过 `self.images` 访问该对象的具体数据。
        - **`cls`**：指向**类本身**。你可以通过 `cls` 来调用类的构造函数，或者修改类级别的属性。

### 3. 在这段代码中，`cls` 具体用来做什么？
        在 `from_dict` 方法的最后，有这样一行代码：
        ```python
        return cls(
            images=data["image"],
            image_masks=data["image_mask"],
            state=data["state"],
            ...
        )
        ```
        这里的 `cls(...)` 实际上就是在**调用 `Observation` 类的 `__init__` 构造函数**，创建一个新的 `Observation` 实例并返回。

        ---

### 为什么要用 `cls` 而不是直接写 `Observation`？
        这是为了**支持继承**（多态）。

        假设你以后写了一个子类：
        ```python
        class AdvancedObservation(Observation):
            pass
        ```
        当你调用 `AdvancedObservation.from_dict(data)` 时：
        - 如果写死了 `return Observation(...)`，你会得到父类的实例，这不符合预期。
        - 使用 `return cls(...)`，由于传入的 `cls` 此时是 `AdvancedObservation`，所以它会自动返回 `AdvancedObservation` 的实例。

        **简单总结**：你可以把这里的 `cls` 理解为“**一个能帮你造出当前类对象的工具**”，它是 Python 类方法的标准写法，目的就是为了优雅地实现“工厂方法”模式。
"""

# ---------------------------------------------------------------------------
# Actions — 动作类型别名
# ---------------------------------------------------------------------------
# 动作数据的格式定义。包含在数据变换输出的 "actions" 字段中。
# 维度含义：
#   *b = 批次维度
#   ah = action_horizon，即动作序列的长度（预测未来多少个时间步的动作）
#   ad = action_dim，即动作空间的维度数
Actions = at.Float[ArrayT, "*b ah ad"]


# ---------------------------------------------------------------------------
# preprocess_observation — 观测数据预处理
# ---------------------------------------------------------------------------
def preprocess_observation(
    rng: at.KeyArrayLike | None,
    observation: Observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
) -> Observation:
    """预处理观测数据。

    这是输入到模型之前的最后一步处理，包含以下操作：
    1. 验证所需的图像键都存在
    2. 如果图像尺寸不符合要求，进行缩放（使用填充保持宽高比）
    3. 如果处于训练模式，应用数据增强（随机裁剪、旋转、颜色抖动等）
    4. 补全缺失的图像掩码（默认为全有效）

    Args:
        rng: JAX 随机数生成器 key，用于数据增强中的随机操作。训练模式下必须提供。
        observation: 原始观测数据。
        train: 是否为训练模式。训练模式下会应用数据增强。
        image_keys: 需要处理的图像键名列表，默认使用 IMAGE_KEYS 常量。
        image_resolution: 目标图像尺寸 (高, 宽)。

    Returns:
        预处理后的 Observation 对象。
    """
    # 验证所有必需的图像键都存在
    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    batch_shape = observation.state.shape[:-1]  # 获取批次形状（去掉状态维度）

    out_images = {}
    for key in image_keys:
        image = observation.images[key]

        # 如果图像尺寸与目标不一致，进行缩放
        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            image = image_tools.resize_with_pad(image, *image_resolution)

        if train:
            # ---- 训练时的数据增强 ----
            # augmax 库期望输入在 [0, 1] 范围内，因此先转换，原值域范围为[-1,+1]
            image = image / 2.0 + 0.5

            transforms = []
            # 对非腕部摄像头图像进行空间变换（随机裁剪、缩放、旋转）
            # 腕部图像通常包含关键的操作细节，不做空间变换以免影响任务
            if "wrist" not in key:
                height, width = image.shape[1:3]
                transforms += [
                    augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),  # 随机裁剪 95%
                    augmax.Resize(width, height),  # 缩放回原尺寸
                    augmax.Rotate((-5, 5)),  # 随机旋转 ±5 度
                ]
            # 颜色抖动（对所有图像包括腕部视图都应用）
            transforms += [
                augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
            ]

            # 为批次中的每张图像分配独立的随机种子，实现每个样本不同的增强
            sub_rngs = jax.random.split(rng, image.shape[0])
            # 使用 vmap 对批次应用增强链，实现向量化处理
            image = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)

            # 转换回模型需要的 [-1, 1] 范围
            image = image * 2.0 - 1.0

        out_images[key] = image

    # --- 处理图像掩码 ---
    """
    这段代码的作用是 为每个图像的掩码兜底：确保每个图像视角都有对应的掩码。

  具体来说：
  1. 遍历所有处理后的图像（out_images 中的每个键）
  2. 检查原始数据中是否提供了该视角的掩码
  3. 如果没提供 → 生成一个全 True 的默认掩码（"我默认所有图像都有效"）
  4. 如果提供了 → 将原始掩码转换为 JAX 数组使用

  掩码的效果

  掩码是一个布尔张量，形状为 [*b]（批次维度，不含空间维度），每个样本对应一个 True 或 False：

  - True：这张图像是有效的，模型应该正常处理它
  - False：这张图像是无效/缺失的，模型应忽略它

  在模型内部的实际效果

  掩码主要影响两个地方：

  1. ViT（视觉 Transformer）的处理

  当 image_mask[样本i] = False 时，模型（pi0.py / pi0_fast.py）会将该样本的图像特征替换为全零向量或特殊占位 token，使其不参与后续 attention 计算。

  2. 损失计算

  在 compute_loss 中，被掩码的样本对应的损失会被置零，即该样本不贡献梯度更新。

  实际场景举例

  想象一个机器人数据集，收集过程中某个摄像头偶尔断线：

  batch sample  | base_0_rgb | left_wrist_0_rgb | mask(base) | mask(wrist)
  --------------|------------|------------------|------------|-------------
  样本 1（正常）  |  正常图像   |    正常图像       |   True     |   True
  样本 2（断线）  |  正常图像   |  全黑/全灰占位    |   True     |   False
  样本 3（正常）  |  正常图像   |    正常图像       |   True     |   True

  - 样本 2 的左腕部图像是占位数据，不能用来训练 —— 掩码 False 让模型直接跳过它
  - 其他样本正常参与训练

  这样做的意义是：不需要扔掉整个样本 —— 一个摄像头坏了，其他视角的监督信号仍然能用来训练。
    """
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # 如果某个视角没有提供掩码，默认全部有效（不屏蔽任何图像）
            out_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool)
        else:
            out_masks[key] = jnp.asarray(observation.image_masks[key])
    """
    好问题。核心原因有两个：

### 1. 掩码的形状只关心「批次维度」，不关心空间维度

图像掩码是**逐样本**（per-sample）的布尔值，形状是 `[*b]`，不是 `[*b, h, w, c]`。它回答的是"这个样本的这张图像有没有效"，不是"这张图像的哪些像素有效"。

所以不管是用 `state` 还是 `image` 取批次维度，本质上都是在取 `*b`：

```python
state.shape       # → [*b, s]    如 [4, 8] → batch_shape = [4]
image.shape       # → [*b, h, w, c] 如 [4, 224, 224, 3] → batch_shape = [4]
```

两种写法都能得到 `[*b]`，区别只在于从哪个张量上切。

### 2. 用 `state` 更健壮（主要理由）

用 `state.shape[:-1]` 而非 `image.shape[:-3]` 的考虑：

|              | 用 `state`                                                                                                          | 用 `image`                                                                 |
| ------------ | ------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| **保证存在** | `state` 是 `Observation` 的**必需字段**，一定有                                                                     | 图像以 `dict[str, ...]` 存储，不能保证某个 key 一定存在                    |
| **语义清晰** | `state` 是低维的，`shape[:-1]` 就是"切掉最后一维"；而 `image.shape[:-3]` 读起来容易混淆（"为什么减去3？"）          |                                                                            |
| **不变性**   | 所有字段共享同一组批次维度 —— `state` 的 `*b` 就是 `image` 的 `*b`，用状态明确声明"这是整个 observation 的批次形状" |                                                                            |
| **避免冗余** | 只在循环前定义一次 `batch_shape`，循环中复用                                                                        | 如果某个 key 的 `image` 是空的或形状异常，可能出现问题（虽然实践中不太会） |

另外注意第 326 行：

```python
batch_shape = observation.state.shape[:-1]  # 去掉状态维度 s
```

`batch_shape` 已经定义好了，后面在 `out_masks` 的生成和可能的其他逻辑中共用——一次提取，多处使用，符合 DRY 原则。

### 一句话总结

形状上两者等价，但 `state.shape[:-1]` 更稳妥、更清晰、只切一个维度 —— 而 `image.shape[:-3]` 会让人多读两秒去想"为什么是3"。代码选择了更不容易出错的那条路。
    """

    # 构建并返回新的 Observation 对象
    return Observation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
    )


# ---------------------------------------------------------------------------
# BaseModelConfig — 模型配置基类
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)  # frozen=True 使配置不可变，保证安全性
class BaseModelConfig(abc.ABC):
    """模型配置的抽象基类。

    所有具体模型（π₀、π₀-FAST、π₀.₅）的配置都应继承此类。
    配置对象定义了模型的超参数，并负责创建对应的模型实例。

    配置对象在 openpi 的整个流程中处于核心位置：
    训练配置（TrainConfig）包含模型配置和数据配置，
    通过模型配置可以创建和加载模型。
    """

    # 动作空间维度 —— 模型输出的动作向量有多少个分量
    # 例如：6 自由度机械臂 + 夹爪 = 7 维动作空间
    action_dim: int

    # 动作时间步数（action horizon）—— 模型一次性预测未来多少个时间步的动作
    # 更大的 horizon 意味着模型规划更远的未来，但预测精度可能下降
    action_horizon: int

    # 标记化文本指令的最大长度
    # 超过此长度的指令会被截断，不足的会进行 padding
    max_token_len: int

    @property
    @abc.abstractmethod
    def model_type(self) -> ModelType:
        """返回模型类型枚举值。

        子类必须实现此属性以标识自己的模型架构类型。
            PI0 = "pi0"
            PI0_FAST = "pi0_fast"
            PI05 = "pi05"
        """
        ...

    @abc.abstractmethod
    def create(self, rng: at.KeyArrayLike) -> "BaseModel":
        """创建一个新的模型实例，并初始化参数。

        Args:
            rng: JAX 随机数生成器 key，用于参数初始化。

        Returns:
            初始化后的模型实例。
        """
        ...

    def load(self, params: at.Params, *, remove_extra_params: bool = True) -> "BaseModel":
        """从给定的参数字典创建模型。

        这是加载预训练模型的主要方法。它首先创建一个"空壳"模型（使用 nnx.eval_shape 避免实际分配内存），
        然后将传入的参数加载到模型中。

        关键步骤：
        1. 使用 nnx.eval_shape 创建模型的结构（不包含实际参数值）
        2. 使用 nnx.split 将模型拆分为图结构（graphdef）和参数状态（state）
        3. 可选地移除参数字典中多余的项（intersect_trees）
        4. 验证参数结构与模型结构匹配
        5. 将参数状态替换为加载的参数
        6. 合并图结构和参数状态，得到完整的模型

        Args:
            params: 要加载的模型参数字典。
            remove_extra_params: 是否移除 params 中模型不需要的键（默认为 True）。
                                 这在加载预训练检查点时很有用，因为检查点可能包含
                                 训练过程中保存的额外状态。

        Returns:
            加载了给定参数的模型实例。
        """
        model = nnx.eval_shape(self.create, jax.random.key(0))
        graphdef, state = nnx.split(model)
        if remove_extra_params:
            params = ocp.transform_utils.intersect_trees(state.to_pure_dict(), params)
        # 校验参数结构的兼容性（仅检查形状，不检查数据类型）
        at.check_pytree_equality(expected=state.to_pure_dict(), got=params, check_shapes=True, check_dtypes=False)
        state.replace_by_pure_dict(params)
        return nnx.merge(graphdef, state)

    """
    我好奇的是，这里并不需要pi0或者pi0.5的模型代码吗？只根据params就可以将神经网络重构起来吗？如果JAX是如此，那么pytorch呢？  
    
好问题，这触及了 JAX 和 PyTorch 在模型加载上的本质区别。

## JAX：架构来自 `self.create`，而非 params

先看仔细：`load` 方法里其实**调用了 `self.create`**：

```python
model = nnx.eval_shape(self.create, jax.random.key(0))  # ← 这里的 create 就是架构代码！
graphdef, state = nnx.split(model)
```

`self.create` 是谁？它是每个具体配置类（如 `Pi0Config`）中实现的抽象方法，里面**就是模型架构代码**：

```python
# 假设在 pi0_config.py 中
class Pi0Config(BaseModelConfig):
    def create(self, rng):
        return Pi0Model(
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            gemma=GemmaModel(...),      # ← 架构在这里定义
            siglip=SiglipModel(...),
            ...
        )
```

`nnx.eval_shape` 会**完整执行 `self.create` 的代码路径**，只是不分配实际内存。所以：

| 组件                                           | 来源                         |
| ---------------------------------------------- | ---------------------------- |
| **架构拓扑**（层结构、连接方式、forward 逻辑） | `self.create()` → `graphdef` |
| **参数值**（具体权重数字）                     | 检查点文件 → `params`        |

两者缺一不可。`graphdef` 就是"神经网络的设计图纸"，`params` 是"材料"。

## 一个比喻

```
graphdef = 建筑蓝图（"这里有一面墙，一个窗户"）
params   = 具体建材（"砖头 500 块，水泥 3 吨"）

graphdef + params = 完整的建筑
```

**参数本身不包含架构信息**。`params` 里面只有一堆形状为 `(784, 256)`、`(256, 128)` 的数值张量，没有"这些张量应该组成一个两层 MLP"的信息。

## JAX 的独特之处：eval_shape + split

JAX 之所以给你"只用 params 就重构了模型"的错觉，是因为：

**1. `eval_shape` 零成本执行架构代码**

PyTorch 中 `model = Pi0Pytorch(config)` 会初始化所有参数（分配内存、可能随机初始化）。而 JAX 的 `eval_shape` 只追踪形状，**不分配数组、不执行随机初始化**，远快于实际创建。

**2. `split` 把架构和参数彻底分离**

```python
graphdef, state = nnx.split(model)
# graphdef = 蓝图（代码结构）
# state    = 参数数值（但现在是空的占位符）

state.replace_by_pure_dict(params)  # 从检查点填入真实值
nnx.merge(graphdef, state)           # 组装
```

这让你可以**把架构代码（graphdef）和参数值（state）分开序列化/传输/复用**。

## PyTorch 的做法

再看 `load_pytorch`：

```python
def load_pytorch(self, train_config, weight_path: str):
    model = pi0_pytorch.PI0Pytorch(config=train_config.model)  # ← 显式创建架构
    safetensors.torch.load_model(model, weight_path)            # ← 填入权重
    return model
```

PyTorch 没有 `eval_shape` 和 `split` 这类分离机制，所以流程是：

```
1. 创建完整模型（含随机初始化的参数） ← 架构 + 参数
2. 用 load_state_dict / safetensors 覆盖参数
3. 得到完整模型
```

PyTorch 中架构和参数是**天生绑定的**——模型对象既包含"有哪些层"也包含"这些层的权重"。加载时两步必须一起做，先建模型再灌权重。

## 核心区别总结

|                    | **JAX (Flax NNX)**                            | **PyTorch**                                     |
| ------------------ | --------------------------------------------- | ----------------------------------------------- |
| **架构定义**       | `create()` 方法，通过 `eval_shape` 零成本追踪 | `PI0Pytorch(config)` 显式实例化（含参数初始化） |
| **架构与参数**     | 可分离（`split` → `merge`）                   | 绑定（一个对象包含两者）                        |
| **加载实质**       | 从 `graphdef` 恢复蓝图 → 从 `params` 填数值   | 创建模型对象 → `load_state_dict` 覆盖权重       |
| **需要架构代码吗** | ✅ 需要（`self.create` 中）                   | ✅ 需要（`PI0Pytorch(config)` 中）              |

## 回到你的问题

> 这里并不需要 pi0 或者 pi0.5 的模型代码吗？

**需要。** 只是它的调用被封装在了 `self.create()` 里，而 `nnx.eval_shape` 让这个调用看起来像没有执行完整构架代码一样——但实际上它完整执行了，只是跳过了内存分配。

> 只根据 params 就可以将神经网络重构起来吗？

**不行。** `params` 只是一堆数值字典。没有 `graphdef`（来自架构代码），你甚至不知道哪些参数对应哪些层。检查点文件 + 架构代码 = 完整模型，缺一不可。
    """

    def load_pytorch(self, train_config, weight_path: str):
        """从 PyTorch 权重文件加载模型。

        用于加载 PyTorch 格式的预训练权重。这在以下场景中特别有用：
        - 使用 PyTorch 进行训练（而非 JAX）
        - 加载从 JAX 转换而来的 PyTorch 权重（使用 convert_jax_model_to_pytorch.py）

        Args:
            train_config: 训练配置对象，包含模型配置等信息。
            weight_path: PyTorch 权重文件路径（通常是 .safetensors 格式）。

        Returns:
            PyTorch 模型实例。
        """
        logger.info(f"train_config: {train_config}")
        model = pi0_pytorch.PI0Pytorch(config=train_config.model)
        safetensors.torch.load_model(model, weight_path)
        return model

    @abc.abstractmethod
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[Observation, Actions]:
        """返回模型的输入规范。

        用于获取模型期望的输入形状和数据类型。返回值中的每个元素都是
        jax.ShapeDtypeStruct（仅包含 shape 和 dtype 信息，不包含实际数据）。
        这在以下场景中很实用：
        - 构建数据加载器
        - 模型编译（JIT）
        - 调试输入输出尺寸

        Args:
            batch_size: 批次大小。

        Returns:
            (Observation 规范, Actions 规范) 的元组。
        """
        ...

    def fake_obs(self, batch_size: int = 1) -> Observation:
        """生成模拟的观测数据（全 1 填充）。

        主要用于调试和测试：创建一个具有正确形状和数据类型的虚拟观测，
        可用于快速验证模型的前向传播是否正常。

        Args:
            batch_size: 批次大小。

        Returns:
            模拟的 Observation 对象。
        """
        observation_spec, _ = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), observation_spec)

    def fake_act(self, batch_size: int = 1) -> Actions:
        """生成模拟的动作数据（全 1 填充）。

        与 fake_obs 类似，用于调试和测试。

        Args:
            batch_size: 批次大小。

        Returns:
            模拟的 Actions。
        """
        _, action_spec = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), action_spec)


# ---------------------------------------------------------------------------
# BaseModel — 模型实现基类
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class BaseModel(nnx.Module, abc.ABC):
    """所有模型实现的抽象基类。

    继承自 nnx.Module（提供神经网络模块的基础功能）和 ABC（提供抽象接口）。

    子类必须实现：
    - compute_loss(): 计算训练损失
    - sample_actions(): 根据观测生成动作预测

    共享属性（通过 super().__init__() 初始化）：
    - action_dim: 动作空间维度
    - action_horizon: 动作预测时间步数
    - max_token_len: 指令最大 token 长度
    """

    action_dim: int  # 动作空间维度
    action_horizon: int  # 动作预测长度（时间步数）
    max_token_len: int  # 文本指令最大长度

    @abc.abstractmethod
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: Observation,
        actions: Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        """计算训练损失。

        这是训练过程中最核心的方法。不同类型的模型（流匹配、自回归）会使用不同的损失函数。

        Args:
            rng: JAX 随机数生成器 key。
            observation: 观测数据（模型输入）。
            actions: 真实动作数据（训练目标/标签）。
            train: 是否为训练模式（影响某些操作如 dropout）。

        Returns:
            每个样本每个时间步的损失值，形状为 [*b, ah]。
            注意：返回的是逐样本逐时间步的损失（而非标量），
            调用方需要自行聚合（取平均/求和）。
        """
        ...

    @abc.abstractmethod
    def sample_actions(self, rng: at.KeyArrayLike, observation: Observation, **kwargs) -> Actions:
        """根据观测生成动作预测（推理阶段）。

        这是推理（部署）时调用的核心方法。对于流匹配模型，这涉及从噪声向动作的逐步去噪过程；
        对于自回归模型，这涉及逐个 token 地自回归生成。

        Args:
            rng: JAX 随机数生成器 key，用于采样中的随机操作。
            observation: 观测数据（模型输入）。
            **kwargs: 额外的生成参数（如采样温度、去噪步数等），具体取决于模型实现。

        Returns:
            预测的动作序列。
        """
        ...


# ---------------------------------------------------------------------------
# restore_params — 检查点参数恢复函数
# ---------------------------------------------------------------------------
def restore_params(
    params_path: pathlib.Path | str,
    *,
    restore_type: type[np.ndarray] | type[jax.Array] = jax.Array,
    dtype: jnp.dtype | None = None,
    sharding: jax.sharding.Sharding | None = None,
) -> at.Params:
    """从检查点恢复模型参数。

    此函数兼容 openpi 训练过程中保存的检查点（通过 `save_state` 保存的 orbax 格式）
    以及 openpi 官方发布的预训练检查点。

    恢复过程的几个关键步骤：
    1. 解析检查点路径（支持本地路径和 GCS 云存储路径）
    2. 配置分片策略（如果使用 JAX 数组）
    3. 使用 orbax 的 PyTreeCheckpointer 恢复参数
    4. 处理 NNX 特有的参数结构（移除 "value" 后缀）

    Args:
        params_path: 检查点目录的路径。可以是本地路径或 GCS 路径（"gs://..."）。
        restore_type: 恢复参数的类型。默认为 `jax.Array`（JAX 数组）。
                     设置为 `np.ndarray` 可以以 NumPy 数组形式加载参数。
        dtype: 将所有参数转换为指定的数据类型。默认保持检查点中的原始类型。
        sharding: 参数的分片策略（用于多设备分布）。如果不提供，
                  默认会将参数复制到所有可用设备上。
                  注意：仅在 restore_type=jax.Array 时有效。

    Returns:
        恢复的模型参数字典（NNX 中称为 "pure dict" 的格式）。
    """
    # 解析路径：如果是 GCS 路径，直接使用；否则解析为绝对路径
    params_path = pathlib.Path(params_path).resolve() if not str(params_path).startswith("gs://") else params_path

    # 如果未指定分片策略且恢复类型是 JAX 数组，创建默认的分片配置
    # 默认策略：将所有参数复制到所有可用设备上（fully replicated）
    if restore_type is jax.Array and sharding is None:
        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # 使用 orbax 的 PyTreeCheckpointer 恢复参数
    # orbax 是 JAX 生态中标准的检查点管理库
    with ocp.PyTreeCheckpointer() as ckptr:
        # 先读取元数据，了解检查点包含哪些内容
        metadata = ckptr.metadata(params_path)
        item = {"params": metadata["params"]}

        # 恢复参数树
        params = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=restore_type, dtype=dtype),
                    item,
                ),
            ),
        )["params"]

    # ---- 处理 NNX 的 "value" 后缀 ----
    # 如果在 openpi 训练中使用 `save_state` 保存检查点，NNX 会自动在每个参数键路径
    # 的末尾添加 "value" 层级。例如：
    #   原始键: ("params", "dense", "kernel")
    #   NNX 保存后: ("params", "dense", "kernel", "value")
    # 为了方便后续使用，这里将 "value" 后缀移除，还原为标准的纯字典格式。
    flat_params = traverse_util.flatten_dict(params)
    if all(kp[-1] == "value" for kp in flat_params):
        flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
    return traverse_util.unflatten_dict(flat_params)

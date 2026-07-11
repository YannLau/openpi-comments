"""
============================================================
  aloha_policy.py — ALOHA 机器人策略的数据变换

  本文件定义了 ALOHA（A Low-cost Open-source Hardware A-arm）
  机器人平台与 openpi 模型之间的数据格式转换。

  简单来说，这个文件解决了"坐标系对齐"的问题：
    机器人传感器数据 → 重命名 + 格式转换 + 关节坐标系对齐 → 模型输入
    模型预测的动作 → 坐标系反向转换 → 机器人可执行的指令

  核心概念：
    - ALOHA 是一个双臂机器人平台，每只手臂有 6 个关节 + 1 个夹爪
      → 所以状态/动作维度是 (6+1 + 6+1) = 14 维
    - ALOHA 有 4 个摄像头：高视角、低视角、左腕、右腕
    - π₀ 基础模型在"内部格式"（pi internal format）上预训练，
      ALOHA 的原始数据需要转换到这个格式才能正确推理

  坐标系对齐（adapt_to_pi）是这里的核心难点：
    ALOHA 和 π₀ 对同一个关节角度的定义方向可能相反（符号不同），
    夹爪的开合值也存在线性/角度的差异，需要通过一系列数学变换来对齐。
============================================================
"""

"""
 use_delta_joint_actions 这个属性为什么只有Aloha平台有？另外，他为何能选择这个是否使用增量动作，这不应该是完全看基座模型pi0或者pi0.5训练时选择输出的是绝对动作还是增量动作吗？  
 
 ## `use_delta_joint_actions` 的两个误区

你的问题很敏锐，恰好抓住了两个容易混淆的点。让我逐一拆解。

---

### 1. 为什么只有 Aloha 有这个属性？

**因为 ALOHA 是唯一一个** ***原始数据集存储的是绝对关节角度*** **的平台。**

| 平台       | 原始数据中的动作格式                        | 是否需要 Delta 转换？                                         |
| ---------- | ------------------------------------------- | ------------------------------------------------------------- |
| **ALOHA**  | **绝对关节角度** (absolute joint angles)    | **是** — 通过 `use_delta_joint_actions`                       |
| **LIBERO** | 本身就是**增量动作** (delta actions 已存储) | 不需要，但有 `extra_delta_transform` 作为"二次增量"的兼容选项 |
| **DROID**  | 混合（末端执行器位姿 / 关节位置）           | 根据 `action_space` 类型条件判断，走的是 inline 逻辑          |

所以 `use_delta_joint_actions` 是 `LeRobotAlohaDataConfig` **独有**的，因为只有 ALOHA 的标准 LeRobot 数据集里存的是绝对角度。

对比看代码结构就清楚了：

**ALOHA** (`config.py:616-668`):
```python
@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    use_delta_joint_actions: bool = True  # 先转 delta，再喂模型
```

**LIBERO** (`config.py:690-701`):
```python
class LeRobotLiberoDataConfig(DataConfigFactory):
    extra_delta_transform: bool = False  # 数据已经是 delta 了，默认不再做转换
```

参数名特意叫 `extra_` delta，就是暗示"你的数据已经是 delta 了，这是额外再做一次"。

---

### 2. 更核心的问题：为什么能「选」是否用增量动作？这不应该是模型决定的吗？

这是最大的误区来源。你的直觉是：

> 模型（pi0 / pi0.5）在训练时就固定了输出绝对还是增量 → 所以下游使用者没有选择权

**但实际情况是：模型学到的始终是"输入变换之后的东西"，它根本不关心原始数据是绝对还是增量。**

```
训练时数据流：

原始 ALOHA 数据 (绝对关节角 0.5 rad)
     │
     ▼
DeltaActions(输入变换)  ←──── use_delta_joint_actions=True
     │  actions = actions - state
     ▼
[模型学习预测的是这张量：actions - state]     ← 模型看到的是增量
```

```
推理时数据流：

实时观测 → 模型预测增量动作
     │
     ▼
AbsoluteActions(输出变换)   ← 对应输出变换
     │  actions = actions + state
     ▼
执行绝对关节角
```

如果 `use_delta_joint_actions=False`，变换就不插入，模型直接看到原始绝对角度，也就直接学绝对角度。

**选择权在数据预处理层，不在模型层。** 模型就是个函数逼近器 —— 它无条件地拟合训练时看到的 target，至于 target 是增量还是绝对，对它来说只是数值分布不同。

---

### 3. 那为什么 ALOHA 默认要用增量？为什么仿真又关了？

**用增量**（默认  `True`）的原因：

- **压缩输出范围**：增量动作通常是 `[-small, +small]` 的小值，而绝对关节角覆盖 `[-π, π]` 全范围。模型预测小范围数值更容易、更稳定。
- **语义更合理**：对于精细操作（插笔帽、叠毛巾），"当前位置向目标方向移动 0.02 rad" 比"移动到 0.77 rad"更直接。
- **夹爪单独排除**（通过 `make_bool_mask(6, -1, 6, -1)` 让夹爪维度为 False）：夹爪只有开/合，对增量做累加毫无意义。

**仿真关了**（`pi0_aloha_sim` 设 `False`）的原因：

```python
TrainConfig(
    name="pi0_aloha_sim",
    data=LeRobotAlohaDataConfig(
        repo_id="lerobot/aloha_sim_transfer_cube_human",
        use_delta_joint_actions=False,  # 仿真中不使用 delta action
    ),
    ...
)
```
仿真环境的动作接口可能直接吃绝对关节角，或者仿真数据集的动作本身就分布在绝对空间中，做 delta 转换反而引入了不必要的环节。这不是模型决定的，而是**数据 + 下游执行器接口**决定的。

---

### 一句话总结

`use_delta_joint_actions` 是**数据预处理开关**，不是**模型架构参数**。pi0/pi0.5 本身并不绑定绝对/增量 —— 它学到的是 transformers 流水线最终喂给它的东西。
所以你可以在不同场景下灵活选择要不要做 delta 转换，只需保证训练和推理时 input 和 output 变换互为逆操作即可。

这种设计是合理的解耦：**模型管拟合，数据管表示。**
"""


import dataclasses  # 数据类装饰器，用于定义不可变的变换配置
from typing import ClassVar  # 类变量类型注解（ClassVar 表示这个是类级别的常量，不是实例属性）

import einops  # 强大的张量操作库，提供人类可读的重排操作（如 rearrange）
import numpy as np  # 科学计算库

from openpi import transforms  # 数据变换模块的基础类（DataTransformFn）


# ============================================================================
# 函数：make_aloha_example（生成示例数据）
#
# 这是一个辅助函数，用于生成一个随机的 ALOHA 输入样例。
# 它的主要用途是测试和调试——你可以用这个函数生成的数据
# 来验证变换流水线是否能正常工作，而不用连接真实的机器人。
#
# 返回值中各字段的含义：
#   - state: 机器人当前状态（关节角度 + 夹爪开合）
#            ALOHA 双臂 = 左臂 6 个关节 + 1 个夹爪 + 右臂 6 个关节 + 1 个夹爪 = 14 维
#   - images: 4 个摄像头的图像
#   - prompt: 语言指令，告诉机器人要做什么
# ============================================================================
def make_aloha_example() -> dict:
    """创建一个 ALOHA 策略的随机输入样例（用于测试和调试）。"""
    return {
        # 状态向量：14 维，全 1 数组
        # 为什么是 14？
        #   左臂：6 个关节角 + 1 个夹爪 = 7 维
        #   右臂：6 个关节角 + 1 个夹爪 = 7 维
        #   总计：14 维
        "state": np.ones((14,)),

        # 图像数据：[channel, height, width] 格式，即 (3, 224, 224)
        #   3 个通道：RGB（红绿蓝）
        #   224 x 224：图像分辨率
        #   uint8：每个像素 0-255
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_low": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


# ============================================================================
# 数据类：AlohaInputs（输入变换）
#
# 这个类的作用是将 ALOHA 机器人的原始观测数据转换为模型可接受的格式。
# DataTransformFn 是变换函数的基类，实现了 __call__ 的特殊方法，
# 使得 AlohaInputs 的实例可以像函数一样被调用。
#
# @dataclass(frozen=True) 表示这是一个不可变的数据类，创建后不能修改，
# 这有助于避免意外修改配置。
#
# 输入变换的核心工作：
#   1. 解码原始数据（_decode_aloha）：调整图像维度顺序，转换关节坐标系
#   2. 重命名摄像头字段：将 ALOHA 的摄像头名映射到 π₀ 模型期望的命名
#   3. 处理缺失摄像头：用黑色图像填充，并标记 image_mask=False
#   4. 如果训练数据中有动作（actions），也进行坐标系转换
#   5. 如果输入中有文本提示（prompt），直接传递
#
# 摄像头命名映射关系：
#   ALOHA 原始名称 → π₀ 模型内部名称
#   "cam_high"      → "base_0_rgb"        （基础/主摄像头，必须存在）
#   "cam_left_wrist" → "left_wrist_0_rgb"  （左腕摄像头，可选）
#   "cam_right_wrist" → "right_wrist_0_rgb"（右腕摄像头，可选）
#   "cam_low"        → 不映射（π₀ 模型不使用低视角摄像头）
#
# 为什么要重命名？
#   ALOHA 使用人类可读的名称（cam_high, cam_low...），
#   而 π₀ 模型使用结构化的命名（base_0_rgb, left_wrist_0_rgb...），
#   需要在这两套"命名空间"之间做转换。
# ============================================================================
@dataclasses.dataclass(frozen=True)
class AlohaInputs(transforms.DataTransformFn):
    """ALOHA 策略的输入变换。

    期望接收的输入格式：
    - images: dict[name, img] 其中 img 是 [channel, height, width] 格式。
              name 必须包含 "cam_high"（主摄像头），其他摄像头可选。
    - state: [14] 维数组（6+1+6+1）。
    - actions: [action_horizon, 14] 维数组（仅在训练时提供）。
    """

    # 是否适配到 π₀ 内部格式。
    # True  → 将 ALOHA 的关节和夹爪值转换到 π₀ 内部空间
    #          （这个内部空间是 π₀ 基础模型预训练时使用的）
    # False → 保持原始值不变（仅当使用未经过预训练的模型时需要）
    adapt_to_pi: bool = True

    # 期望的摄像头名称列表（类变量，所有实例共享）。
    # ClassVar 告诉类型检查器：这是类级别的常量，不是实例字段。
    # 所有输入摄像头必须在这些名称中，多出来的摄像头会报错。
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = (
        "cam_high",
        "cam_low",
        "cam_left_wrist",
        "cam_right_wrist",
    )

    def __call__(self, data: dict) -> dict:
        """执行输入变换：ALOHA 原始格式 → 模型统一格式。

        Args:
            data: ALOHA 原始输入字典，包含 images, state, actions（可选）, prompt（可选）。

        Returns:
            变换后的字典，可供 π₀ 模型使用。
        """
        # ========== 第 1 步：解码原始数据 ==========
        # _decode_aloha 会做两件事：
        #   1. 图像：从 [C, H, W] 转换为 [H, W, C]（模型期望的格式）
        #   2. 状态：如果 adapt_to_pi=True，转换关节坐标系
        data = _decode_aloha(data, adapt_to_pi=self.adapt_to_pi)

        # ========== 第 2 步：验证输入摄像头 ==========
        in_images = data["images"]
        # 检查是否有不在 EXPECTED_CAMERAS 中的摄像头
        # set 差集运算：in_images 中有而 EXPECTED_CAMERAS 中没有的名称
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(
                f"Expected images to contain {self.EXPECTED_CAMERAS}, "
                f"got {tuple(in_images)}"
            )

        # ========== 第 3 步：处理主摄像头 ==========
        # 主摄像头（cam_high）是必须存在的，否则会抛出 KeyError
        base_image = in_images["cam_high"]

        # π₀ 模型期望的图像字段名是 "base_0_rgb"
        images = {
            "base_0_rgb": base_image,
        }
        # image_mask 告诉模型哪些摄像头真正有数据
        # np.True_ 表示这个摄像头有真实数据
        image_masks = {
            "base_0_rgb": np.True_,
        }

        # ========== 第 4 步：处理可选摄像头（腕部摄像头） ==========
        # 腕部摄像头是可选的，可能存在也可能不存在。
        # 如果不存在，用全黑图像填充，并将 image_mask 设为 False。
        extra_image_names = {
            # 目标名称（π₀ 模型使用） → 源名称（ALOHA 原始数据）
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                # 摄像头存在 → 正常映射
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                # 摄像头不存在 → 用与主摄像头同形状的黑色图像填充
                # np.zeros_like(base_image) 生成与 base_image 形状相同
                # 但全部为 0 的数组（黑色图像）
                images[dest] = np.zeros_like(base_image)
                # 标记为 False，告诉模型"这个摄像头没有真实数据"
                image_masks[dest] = np.False_

        # ========== 第 5 步：组装最终输出 ==========
        inputs = {
            "image": images,  # π₀ 模型的图像输入
            "image_mask": image_masks,  # 图像掩码，指示哪些是真实图像
            "state": data["state"],  # 转换后的状态向量
        }

        # 训练时才有的字段
        if "actions" in data:
            # 将动作也转换到 π₀ 坐标系
            actions = np.asarray(data["actions"])
            actions = _encode_actions_inv(actions, adapt_to_pi=self.adapt_to_pi)
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


# ============================================================================
# 数据类：AlohaOutputs（输出变换）
#
# 这个类负责将模型预测的动作从 π₀ 内部格式转换回 ALOHA 机器人可执行的格式。
#
# 输出变换是输入变换的"逆过程"——但也只是部分逆过程。
# 因为输出变换只涉及"动作"的反向转换，不涉及图像等输入字段。
# ============================================================================
@dataclasses.dataclass(frozen=True)
class AlohaOutputs(transforms.DataTransformFn):
    """ALOHA 策略的输出变换。

    将 π₀ 模型预测的动作转换回 ALOHA 机器人可执行的格式。
    """

    # 是否从 π₀ 内部格式转换回来
    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        """执行输出变换：模型动作 → ALOHA 可执行动作。

        Args:
            data: 模型输出的字典，包含 "actions" 字段。

        Returns:
            转换后的动作字典。
        """
        # 只取前 14 维（ALOHA 双臂共 14 个自由度）
        # 为什么需要这步？
        #   模型可能预测了更多维度的动作（例如包含速度、加速度等），
        #   但 ALOHA 机器人只需要关节位置/角度。
        actions = np.asarray(data["actions"][:, :14])

        # _encode_actions 将 π₀ 内部格式转回 ALOHA 格式
        return {"actions": _encode_actions(actions, adapt_to_pi=self.adapt_to_pi)}


# ============================================================================
# 以下是一系列辅助函数，负责 ALOHA 和 π₀ 之间的坐标系转换。
#
# 为什么需要这些转换？
#   机器人的关节坐标系没有"通用标准"。
#   同样是"左臂的第一关节"，ALOHA 和 π₀ 可能定义的方向完全相反。
#   就像"向前"这个方向，不同坐标系下可能是 +x、-x 或 +z。
#
# 具体差异有两点：
#   1. 关节方向符号翻转：某些关节在 ALOHA 和 π₀ 中的正方向定义相反
#   2. 夹爪值的量纲差异：ALOHA 用线性位移（米）表示夹爪开合，
#      π₀ 用角度（弧度）表示
# ============================================================================


# ============================================================================
# 函数：_joint_flip_mask
#
# 返回一个"翻转掩码"向量，用于在 ALOHA 和 π₀ 之间转换关节符号。
#
# 掩码中的每个元素对应一个关节：
#   1  = 方向一致，不需要翻转
#   -1 = 方向相反，需要取反
#
# 为什么是 [1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1]？
#   这 14 个值对应双臂的 14 个自由度：
#     左臂 6 个关节： [1, -1, -1, 1, 1, 1]  ← 第 2、3 关节方向相反
#     左夹爪：        [1]                     ← 方向一致
#     右臂 6 个关节： [1, -1, -1, 1, 1, 1]  ← 第 2、3 关节方向相反
#     右夹爪：        [1]                     ← 方向一致
#   也就是说，ALOHA 和 π₀ 在双臂的第 2、3 关节上定义的方向是相反的。
#   这些数值是通过人工标定得出的，不能随意修改。
# ============================================================================
def _joint_flip_mask() -> np.ndarray:
    """用于在 ALOHA 和 π₀ 之间转换关节角度的符号翻转掩码。"""
    return np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])


# ============================================================================
# 归一化 / 反归一化工具函数
#
# _normalize:   将 [min_val, max_val] 范围映射到 [0, 1]
# _unnormalize: 将 [0, 1] 范围映射回 [min_val, max_val]
#
# 公式：
#   normalized = (x - min_val) / (max_val - min_val)
#   unnormalized = x * (max_val - min_val) + min_val
#
# 这两个函数常用于处理夹爪位置/角度在不同量纲之间的转换。
# ============================================================================
def _normalize(x, min_val, max_val):
    """将 x 从 [min_val, max_val] 线性映射到 [0, 1]。"""
    return (x - min_val) / (max_val - min_val)


def _unnormalize(x, min_val, max_val):
    """将 x 从 [0, 1] 线性映射回 [min_val, max_val]。"""
    return x * (max_val - min_val) + min_val


# ============================================================================
# 函数：_gripper_to_angular
#
# 将 ALOHA 的夹爪开合值（线性位移，单位：米）转换为
# π₀ 使用的夹爪开合值（角度，单位：弧度）。
#
# 为什么需要这个转换？
#   ALOHA 底层硬件（Interbotix 机器人臂）用电机控制夹爪，
#   电机旋转角度和夹爪实际开口宽度之间存在非线性关系。
#   ALOHA 将此关系简化为线性处理。
#
#   而 π₀ 基础模型在预训练时使用的是"角度"表示，
#   所以我们需要将 ALOHA 的线性数值"还原"为角度值。
#
# 转换步骤：
#   1. 反归一化：将 [0, 1] 的归一化值映射回 ALOHA 的原始线性值
#   2. 线性 → 弧度：使用 Interbotix 电机运动学公式，
#      将线性位移转换为电机轴旋转角度
#   3. 重新归一化：将弧度值映射到 π₀ 夹爪数据的 [0, 1] 范围
#
# 部分常数值来源：
#   - PUPPET_GRIPPER_POSITION_OPEN / CLOSED: ALOHA 代码中定义的夹爪开合极限值
#   - arm_length, horn_radius: Interbotix 电机硬件参数
#   - 2405, 3110, 4096, 2048: π₀ 夹爪编码器的计数值
# ============================================================================
def _gripper_to_angular(value):
    """将 ALOHA 的夹爪线性位移值转换为 π₀ 的夹爪角度（弧度）值。

    ALOHA 将夹爪位置编码为线性空间，但 π₀ 在预训练时使用的是角度空间。
    这个函数反转了 ALOHA 的线性化处理，得到电机轴的实际旋转角度。
    """
    # 第 1 步：反归一化到 ALOHA 的原始线性值（单位：米）
    # 这些常量来自 ALOHA 代码：
    #   PUPPET_GRIPPER_POSITION_OPEN   = 0.05800 （夹爪完全张开时的线性位移）
    #   PUPPET_GRIPPER_POSITION_CLOSED = 0.01844 （夹爪完全闭合时的线性位移）
    value = _unnormalize(value, min_val=0.01844, max_val=0.05800)

    # 第 2 步：线性位移 → 角度（弧度）
    # 这是 Interbotix 舵机控制库中的运动学逆变换。
    #
    # 想象一个简单的曲柄-滑块机构：
    #   旋转臂（horn）通过连杆（arm）驱动夹爪做直线运动。
    #   已知线性位移，求旋转角度的公式如下。
    def linear_to_radian(linear_position, arm_length, horn_radius):
        # 余弦定理：cos(θ) = (b² + c² - a²) / (2bc)
        # 其中：
        #   a = arm_length（连杆长度）
        #   b = horn_radius（旋转臂长度）
        #   c = linear_position（线性位移，即夹爪开口大小）
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (
            2 * horn_radius * linear_position
        )
        # arcsin 得到角度值（弧度）
        # np.clip 确保数值在 arcsin 的定义域 [-1, 1] 内
        return np.arcsin(np.clip(value, -1.0, 1.0))

    # Interbotix 硬件参数（单位：米）：
    #   arm_length  = 0.036 （连杆长度，约 3.6 厘米）
    #   horn_radius = 0.022 （旋转臂长度，约 2.2 厘米）
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # 第 3 步：将弧度值归一化到 π₀ 的 [0, 1] 范围
    #
    # π₀ 的夹爪数据范围：
    #   编码器计数值从 2405（完全张开）到 3110（完全闭合）
    #   总编码器范围：4096（12 位编码器）
    #   ALOHA 使用 2048 作为零点（编码器行程的中点）
    #   转换为弧度后，范围在 (0.5476, 1.6296) 之间
    return _normalize(value, min_val=0.5476, max_val=1.6296)


# ============================================================================
# 函数：_gripper_from_angular
#
# 将 π₀ 的夹爪角度值转换为 ALOHA 的夹爪角度值。
# 这是输出变换的一部分（模型输出 → ALOHA 格式）。
#
# 特别注意：这里的转换是在"角度"域中进行的。
# π₀ 输出的也是角度，但两者的"零位"和"范围"定义不同。
# 所以需要做的只是一个线性偏移 + 重新归一化。
# ============================================================================
def _gripper_from_angular(value):
    """将 π₀ 的夹爪角度转换为 ALOHA 的夹爪角度。

    π₀ 和 ALOHA 都用角度表示夹爪，但范围不同：
      - π₀: [0.5476, 1.6296] 弧度（归一化到 [0, 1]）
      - ALOHA: [-0.6213, 1.4910] 弧度（归一化到 [0, 1]）

    所以这个函数做的事情是：将 π₀ 范围的弧度值映射到 ALOHA 范围。
    """
    # 第 1 步：先做偏移（对齐零位）
    # π₀ 的 0 对应编码器值 2048（中点），弧度值为 0.5476，
    # 所以需要先加 0.5476 将零位对齐。
    value = value + 0.5476

    # 第 2 步：重新归一化到 ALOHA 的弧度范围
    # ALOHA 代码中定义的夹爪角度范围：
    #   PUPPET_GRIPPER_JOINT_OPEN  = 1.4910  （完全张开）
    #   PUPPET_GRIPPER_JOINT_CLOSE = -0.6213 （完全闭合）
    return _normalize(value, min_val=-0.6213, max_val=1.4910)


# ============================================================================
# 函数：_gripper_from_angular_inv
#
# _gripper_from_angular 的逆函数。
# 在训练时，需要将 ALOHA 格式的动作标签转换到 π₀ 格式。
# 这是 _gripper_from_angular 的直接数学逆运算：
#   1. 反归一化 ALOHA 的角度范围
#   2. 减去偏移量，回到 π₀ 的范围
# ============================================================================
def _gripper_from_angular_inv(value):
    """_gripper_from_angular 的逆函数（ALOHA → π₀ 格式）。"""
    # 第 1 步：从 ALOHA 的归一化角度范围还原
    value = _unnormalize(value, min_val=-0.6213, max_val=1.4910)
    # 第 2 步：减去偏移量
    return value - 0.5476


# ============================================================================
# 函数：_decode_aloha（解码 ALOHA 原始数据）
#
# 这是输入变换的第一个步骤——将 ALOHA 的原始传感器数据
# 转换为更易处理的形式。
#
# 主要操作：
#   1. 状态向量：应用关节翻转和夹爪转换（_decode_state）
#   2. 图像：调整通道顺序和数据类型
# ============================================================================
def _decode_aloha(data: dict, *, adapt_to_pi: bool = False) -> dict:
    """解码 ALOHA 的原始输入数据。

    将 ALOHA 格式的原始传感器数据转换为统一的中间格式。

    Args:
        data: ALOHA 原始输入字典。
        adapt_to_pi: 是否转换到 π₀ 格式。

    Returns:
        解码后的字典，state 和 images 已被修改。
    """
    # state 的结构：[左臂_6关节, 左夹爪, 右臂_6关节, 右夹爪]
    # 维度：[6, 1, 6, 1] = 14
    state = np.asarray(data["state"])
    state = _decode_state(state, adapt_to_pi=adapt_to_pi)

    # 定义图像转换函数
    def convert_image(img):
        img = np.asarray(img)
        # 如果图像是浮点数（0~1 范围），转换为 uint8（0~255）
        # np.issubdtype(img.dtype, np.floating) 检查是否为浮点数类型
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)

        # 重排维度顺序：
        #   输入：[C, H, W]  →  (Channel, Height, Width)
        #   输出：[H, W, C]  →  (Height, Width, Channel)
        #
        # 为什么需要这个转换？
        #   - ALOHA 使用的格式：[C, H, W]（通道在前）
        #   - openpi/π₀ 模型使用的格式：[H, W, C]（通道在后）
        #   - einops.rearrange 用描述性的字符串来表达维度变换，
        #     比 transpose 或 permute 更容易理解和维护。
        return einops.rearrange(img, "c h w -> h w c")

    images = data["images"]
    images_dict = {name: convert_image(img) for name, img in images.items()}

    data["images"] = images_dict
    data["state"] = state
    return data


# ============================================================================
# 函数：_decode_state（解码状态向量）
#
# 对 ALOHA 的状态向量应用坐标系转换。
#
# ALOHA 的原始状态向量格式：
#   [左臂关节1, 左臂关节2, ..., 左臂关节6, 左夹爪,
#    右臂关节1, 右臂关节2, ..., 右臂关节6, 右夹爪]
#
# 当 adapt_to_pi=True 时，做两件事：
#   1. 关节翻转：用 _joint_flip_mask() 乘以各个关节，
#      将方向与 π₀ 对齐
#   2. 夹爪转换：将夹爪从线性位移转换为角度值（弧度）
# ============================================================================
def _decode_state(state: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    """解码 ALOHA 的状态向量，可选地转换到 π₀ 坐标系。

    Args:
        state: ALOHA 原始状态向量 (14,)。
        adapt_to_pi: 是否转换到 π₀ 内部格式。

    Returns:
        转换后的状态向量。
    """
    if adapt_to_pi:
        # 第 1 步：翻转关节方向
        state = _joint_flip_mask() * state

        # 第 2 步：转换夹爪值（线性位移 → 角度）
        # 索引 [6, 13] 分别是左夹爪和右夹爪在向量中的位置：
        #   0~5:   左臂 6 个关节
        #   6:     左夹爪
        #   7~12:  右臂 6 个关节
        #   13:    右夹爪
        state[[6, 13]] = _gripper_to_angular(state[[6, 13]])
    return state


# ============================================================================
# 函数：_encode_actions（编码输出动作）
#
# 将 π₀ 预测的动作转换回 ALOHA 格式（输出变换）。
#
# 这是 _decode_state 的"逆过程"（但只针对动作，不包括图像）：
#   1. 关节符号翻转（乘以 flip mask）
#   2. 夹爪角度从 π₀ 范围映射到 ALOHA 范围
# ============================================================================
def _encode_actions(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    """将 π₀ 预测的动作转换回 ALOHA 可执行格式。

    Args:
        actions: π₀ 模型预测的动作数组 [..., 14]。
        adapt_to_pi: 是否需要从 π₀ 格式转换。

    Returns:
        ALOHA 格式的动作数组。
    """
    if adapt_to_pi:
        # 翻转关节方向
        actions = _joint_flip_mask() * actions
        # 转换夹爪（π₀ 角度范围 → ALOHA 角度范围）
        actions[:, [6, 13]] = _gripper_from_angular(actions[:, [6, 13]])
    return actions


# ============================================================================
# 函数：_encode_actions_inv（编码输入动作）
#
# 将 ALOHA 格式的训练动作标签转换为 π₀ 格式（输入变换的一部分）。
#
# 这是 _encode_actions 的逆函数，用于训练数据预处理：
#   1. 关节符号翻转
#   2. 夹爪角度从 ALOHA 范围转换到 π₀ 范围
# ============================================================================
def _encode_actions_inv(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    """将 ALOHA 格式的训练动作标签转换为 π₀ 格式。

    这是 _encode_actions 的逆函数，用于训练数据预处理。

    Args:
        actions: ALOHA 格式的动作数组 [..., 14]。
        adapt_to_pi: 是否需要转换到 π₀ 格式。

    Returns:
        π₀ 格式的动作数组。
    """
    if adapt_to_pi:
        actions = _joint_flip_mask() * actions
        actions[:, [6, 13]] = _gripper_from_angular_inv(actions[:, [6, 13]])
    return actions

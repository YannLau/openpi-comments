import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


# ============================================================================
# 辅助函数：构建随机样本（用于测试/调试）
# ============================================================================

def make_aloha_example() -> dict:
    """创建一个随机的 Aloha 策略输入样例，用于测试和调试目的。

    返回的字典结构模拟了真实机器人控制环路的输入格式：
    - state: 14 维关节状态向量（左右各 7 个关节角 + 夹爪位置）
    - images: 4 个摄像头视角（高清、低清、左右腕部）
    - prompt: 语言指令

    Returns:
        dict: 包含随机生成的 state、images 和 prompt 的字典
    """
    return {
        "state": np.ones((14,)),  # 14维关节状态（ALOHA 机器人：7+1+7+1 关节空间）
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),         # 高清全局摄像头
            "cam_low": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),          # 低清全局摄像头
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),   # 左腕部摄像头
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),  # 右腕部摄像头
        },
        "prompt": "do something",  # 自然语言指令
    }


# ============================================================================
# Tron2 输入变换类
# ============================================================================

@dataclasses.dataclass(frozen=True)
class Tron2Inputs(transforms.DataTransformFn):
    """Tron2 策略的输入变换。

    责任链：将机器人底层（Aloha/Tron2）的原始观测数据，转换为模型推理所需的
    标准化输入格式。这是策略推理流水线的第一步。

    【工作流程】
    1. 接收机器人控制环路的原始字典数据
    2. 解码图像（通道重排、类型转换）和关节状态
    3. 将多摄像头图像映射到模型期望的命名空间
    4. 构建包含 images、image_mask、state 的标准输入字典

    【数据流示例】
    robot_raw_data ──→ _decode_tron2() ──→ 重映射图像键名 ──→ 返回标准输入
         │                    │                    │
         ▼                    ▼                    ▼
    {state:[16],        {state:[16],          {image:{...},
     images:{...},       images: HWC格式,      image_mask:{...},
     prompt:...}          prompt:...}           state:[16]}

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [16]
    - actions: [action_horizon, 16]
    """

    # 注意：这个适配开关用于处理 ALOHA 和 π 内部运行时之间的关节空间差异。
    # Tron2 的关节空间是未知的，所以设为 False 表示不做适配。
    # 如果设置为 True，会将 ALOHA 标准关节角度转换为 π 内部运行时的关节空间。
    adapt_to_pi: bool = False

    # 期望的摄像头名称集合。所有输入的摄像头名称必须在这个集合中。
    # 缺失的摄像头会被替换为黑色图像，并且对应的 image_mask 会被设为 False。
    # 注意：cam_low（低清摄像头）在这个配置中没有被使用。
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = (
        "cam_high",          # 高清全局摄像头（必选，用作基础图像）
        "cam_low",           # 低清全局摄像头（可选）
        "cam_left_wrist",   # 左腕部摄像头（可选）
        "cam_right_wrist",  # 右腕部摄像头（可选）
    )

    def __call__(self, data: dict) -> dict:
        """执行输入变换主逻辑。

        Args:
            data: 原始输入字典，包含以下字段:
                - images: dict[str, np.ndarray] 图像字典，键为摄像头名称，值为 [C, H, W] 格式
                - state: np.ndarray 关节状态向量 [16]
                - actions (optional): np.ndarray 动作标签 [action_horizon, 16]（仅训练时有）
                - prompt (optional): str 自然语言指令

        Returns:
            dict: 标准化后的输入字典:
                - image: dict[str, np.ndarray] 图像字典 [H, W, C] 格式
                - image_mask: dict[str, np.bool_] 图像有效掩码（True=有效, False=缺失）
                - state: np.ndarray 关节状态向量
                - actions (optional): 编码后的动作
                - prompt (optional): 语言指令

        Raises:
            ValueError: 如果输入摄像头不在 EXPECTED_CAMERAS 集合中
        """
        # ================================================================
        # 第 1 步：解码原始数据
        # - 图像：从 [C, H, W] 重排为 [H, W, C]（模型期望的格式）
        # - 状态：根据 adapt_to_pi 标志进行关节空间转换
        # - 如果 adapt_to_pi 为 False（Tron2 默认），则不进行任何转换
        # ================================================================
        data = _decode_tron2(data, adapt_to_pi=self.adapt_to_pi)

        # ================================================================
        # 第 2 步：验证摄像头名称
        # ================================================================
        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(
                f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}"
            )

        # ================================================================
        # 第 3 步：构建标准图像命名空间
        #
        # 模型期望的图像键名是抽象的（base_0_rgb, left_wrist_0_rgb），
        # 而机器人平台提供的键名是具体的（cam_high, cam_left_wrist）。
        # 这一步负责"命名空间映射"。
        #
        # 图像命名规则：
        # - base_0_rgb: 基础/全局视角（必选），原始数据键名 "cam_high"
        # - left_wrist_0_rgb: 左腕部视角（可选），原始数据键名 "cam_left_wrist"
        # - right_wrist_0_rgb: 右腕部视角（可选），原始数据键名 "cam_right_wrist"
        # ================================================================

        # 基础图像（必选）—— 作为所有缺失图像的形状模板
        base_image = in_images["cam_high"]

        images = {
            "base_0_rgb": base_image,  # 基础/全局RGB图像
        }
        image_masks = {
            "base_0_rgb": np.True_,  # 基础图像始终有效
        }

        # 附加图像（可选）—— 如果缺失，用与基础图像同尺寸的黑色图像填充
        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",    # 模型键名 → 原始键名
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                # 摄像头存在 → 使用真实图像
                images[dest] = in_images[source]
                image_masks[dest] = np.True_  # 标记为有效
            else:
                # 摄像头缺失 → 用零填充（黑色图像），让模型知道没有这个视角
                images[dest] = np.zeros_like(base_image)  # 使用 base_image 的形状
                image_masks[dest] = np.False_  # 标记为无效

        # ================================================================
        # 第 4 步：组装标准输入字典
        # ================================================================
        inputs = {
            "image": images,          # dict[str, np.ndarray] 所有图像
            "image_mask": image_masks, # dict[str, np.bool_]   图像存在性掩码
            "state": data["state"],    # [16] 关节状态向量
        }

        # 训练时才有动作标签 —— 对动作进行编码后传递给模型
        if "actions" in data:
            actions = np.asarray(data["actions"])
            actions = _encode_actions_inv(actions, adapt_to_pi=self.adapt_to_pi)
            inputs["actions"] = actions  # [action_horizon, 16]

        # 语言指令（可选）
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


# ============================================================================
# Tron2 输出变换类
# ============================================================================

@dataclasses.dataclass(frozen=True)
class Tron2Outputs(transforms.DataTransformFn):
    """Tron2 策略的输出变换。

    责任链：将模型的原始输出（动作预测）转换为机器人可以执行的命令格式。
    这是策略推理流水线的最后一步。

    【与 Tron2Inputs 的区别】
    Tron2Inputs: 原始数据 → 模型输入（数据进入模型之前的处理）
    Tron2Outputs: 模型输出 → 可执行动作（数据离开模型之后的处理）

    当前只做了一个操作：截取动作的前 16 维（去除可能多余的后缀维度），
    然后根据 adapt_to_pi 标志进行关节空间编码。
    """

    # 与 Tron2Inputs 中的 adapt_to_pi 含义相同，控制动作编码时是否
    # 从 π 内部运行时的关节空间转换回 ALOHA 标准关节空间。
    adapt_to_pi: bool = False

    def __call__(self, data: dict) -> dict:
        """执行输出变换。

        Args:
            data: 模型输出的字典，包含 "actions" 字段:
                - actions: np.ndarray 形状为 [action_horizon, N]，其中 N >= 16

        Returns:
            dict: 变换后的输出字典:
                - actions: np.ndarray 形状为 [action_horizon, 16]
        """
        # 仅保留前 16 维动作（裁剪掉模型可能输出的多余维度）
        # 为什么是 16 维？Tron2 的关节空间是 7+1+7+1 结构：
        # [左臂7关节, 左夹爪1, 右臂7关节, 右夹爪1] = 16
        actions = np.asarray(data["actions"][:, :16])

        # 根据 adapt_to_pi 标志进行编码（关节取反 + 夹爪空间转换）
        return {"actions": _encode_actions(actions, adapt_to_pi=self.adapt_to_pi)}


# ============================================================================
# 关节空间变换工具函数
# ============================================================================
#
# 【背景知识：为什么需要关节空间转换？】
#
# ALOHA 机器人和 π 内部运行时（pi internal runtime）使用不同的关节空间约定。
# 具体差异有两点：
#
# 1. 关节方向符号（Joint Flip）
#    ALOHA 和 π 的关节角正负方向定义不同，某些关节需要乘以 -1 来进行对齐。
#    这不会改变关节的物理位置，只是改变表示方式。
#    例如：关节1在 ALOHA 中正方向是顺时针，在 π 中是逆时针 → 需要取反。
#
# 2. 夹爪空间（Gripper Space）
#    ALOHA 的夹爪位置是线性空间（单位：米），通过电位计读取的电压值转换而来。
#    π 内部使用的夹爪位置是角度空间（单位：弧度），通过编码器计数转换而来。
#    两者之间需要经过一个复杂的数学变换（线性→角度，再归一化）。
#
# 【数据流概览】
#
#    ALOHA 原始数据 (线性夹爪空间)
#         │
#         ▼
#    _decode_state() ──→ adapt_to_pi=True ──→ 乘以关节翻转掩码 + 夹爪线性→角度
#         │                              ↓
#         │               π 内部运行时空间 (角度夹爪空间)
#         │                              ↓
#         │               _encode_actions() ← 推理时转换回 ALOHA 空间
#         │                              ↓
#         │               _encode_actions_inv() ← 训练时保持在 π 空间
#         │
#         ▼
#    adapt_to_pi=False → 不进行转换（Tron2 默认）
#


def _joint_flip_mask() -> np.ndarray:
    """返回关节翻转掩码，用于在 ALOHA 和 π 关节空间之间转换。

    【什么是关节翻转？】
    ALOHA 机器人和 π 内部运行时对某些关节的正方向定义是相反的。
    如果一个关节在 ALOHA 中正转对应 π 中的反转，就需要乘以 -1。

    掩码形状：[14]（对应 14 维 ALOHA 关节空间）
    其中 1 表示方向一致，-1 表示方向相反。

    结构：[左臂7 + 左夹爪 + 右臂7 + 右夹爪] = 14 维
    具体：
    - 左臂： [1, -1, -1, 1, 1, 1, 1]  → 第 2、3 个关节需要翻转
    - 左夹爪：[1]                      → 不翻转
    - 右臂： [-1, -1, 1, 1, 1, 1]     → 第 1、2 个关节需要翻转
    - 右夹爪：[1]                      → 不翻转

    Returns:
        np.ndarray: [14] 形状的数组，值为 1 或 -1
    """
    return np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])


def _normalize(x, min_val, max_val):
    """线性归一化：将值从 [min_val, max_val] 映射到 [0, 1]

    公式: (x - min_val) / (max_val - min_val)

    Args:
        x: 输入值
        min_val: 最小值
        max_val: 最大值

    Returns:
        归一化到 [0, 1] 区间的值
    """
    return (x - min_val) / (max_val - min_val)


def _unnormalize(x, min_val, max_val):
    """反归一化：将值从 [0, 1] 映射回 [min_val, max_val]

    公式: x * (max_val - min_val) + min_val

    这是 _normalize 的逆运算。

    Args:
        x: [0, 1] 区间的归一化值
        min_val: 目标最小值
        max_val: 目标最大值

    Returns:
        映射回原始区间的值
    """
    return x * (max_val - min_val) + min_val


def _gripper_to_angular(value):
    """将 ALOHA 的夹爪线性位置转换为 π 内部的夹爪角度位置。

    【为什么需要这个转换？】
    ALOHA 机器人的夹爪使用电位计测量位置，输出的是线性位移（米）。
    π 内部运行时使用的是关节角度（弧度）。两者通过 Interbotix 机械臂的
    运动学模型相互转换。

    【转换流程（三步）】
    1. 反归一化: [0,1] → [0.01844, 0.05800]（ALOHA 夹爪的物理线性位移范围，单位：米）
    2. 线性→弧度: 通过 Interbotix 运动学逆解，将线性位移转换为关节角度（弧度）
       - 运动学公式: θ = arcsin((r² + L² - a²) / (2 * r * L))
         - r = horn_radius（舵机摇臂半径）= 0.022 m
         - L = linear_position（线性位移，即步骤1的输出）
         - a = arm_length（连杆长度）= 0.036 m
    3. 归一化: [0.5476, 1.6296] → [0, 1]（π 内部归一化的夹爪角度范围）
       - 0.5476 弧度 ≈ 夹爪完全张开（编码器计数 3110/4096）
       - 1.6296 弧度 ≈ 夹爪完全闭合（编码器计数 2405/4096）
       - 4096 是总编码器计数，2048 是零位

    Args:
        value: ALOHA 线性夹爪位置（归一化到 [0, 1] 区间）

    Returns:
        π 内部角度夹爪位置（归一化到 [0, 1] 区间，但含义不同）
    """
    # 步骤 1：从 [0, 1] 反归一化到 ALOHA 夹爪物理线性位移范围 [米]
    # PUPPET_GRIPPER_POSITION_OPEN = 0.01844（完全张开）
    # PUPPET_GRIPPER_POSITION_CLOSED = 0.05800（完全闭合）
    value = _unnormalize(value, min_val=0.01844, max_val=0.05800)

    # 步骤 2：Interbotix 运动学逆解 —— 线性位移 → 关节角度
    # 这是 ALOHA 代码中角度→线性变换的逆过程
    def linear_to_radian(linear_position, arm_length, horn_radius):
        """线性位移 → 弧度角的转换。

        基于 Interbotix 机械臂的四连杆机构运动学：
          θ = arcsin((r² + L² - a²) / (2 * r * L))

        其中:
        - r: horn_radius（舵机摇臂半径）
        - L: linear_position（线性位移量）
        - a: arm_length（连杆长度）

        注意：
        - 输入值必须满足 arcsin 定义域 [-1, 1]
        - 如果超出范围，会被 clip 到 [-1, 1]
        """
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return np.arcsin(np.clip(value, -1.0, 1.0))

    # Interbotix 机械臂的硬件参数（来自 Interbotix SDK）
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # 步骤 3：将 π 内部的原始夹爪角度范围归一化到 [0, 1]
    # π 的夹爪编码器：2405（闭合）~ 3110（张开），总 4096 计数，零位 2048
    # 转换为弧度后范围：0.5476（闭合）~ 1.6296（张开）
    return _normalize(value, min_val=0.5476, max_val=1.6296)


def _gripper_from_angular(value):
    """将 π 内部的夹爪角度位置转换为 ALOHA 的夹爪角度位置（推理时使用）。

    【注意】这个函数不是 _gripper_to_angular 的严格逆运算！
    它只做角度空间内的范围转换，不涉及线性/角度之间的转换。
    因为 Trossen 模型的预测输出直接是弧度值。

    【转换流程（两步）】
    1. 偏移: value + 0.5476
       - 0.5476 是 π 内部夹爪闭合时的弧度值
       - 将 π 的 [0.5476, 1.6296] 范围平移到 [0, 2.1772]
       - 相当于去掉了偏置，让范围从 0 开始
    2. 归一化到 ALOHA 的夹爪角度范围:
       [0, 2.1772] → _normalize(..., min_val=-0.6213, max_val=1.4910)
       → 输出在 [0.153, 0.899] 左右

    Args:
        value: π 内部的夹爪角度（弧度），范围 [0, 1] 归一化

    Returns:
        ALOHA 夹爪角度（归一化到 [0, 1] 但范围不同）
    """
    # 注意：Trossen 模型预测已经是弧度值，不需要额外转换
    # 只做偏移：将 π 的夹爪角度偏移到以 0 为基准
    value = value + 0.5476

    # 归一化到 ALOHA 期望的夹爪角度范围
    # PUPPET_GRIPPER_JOINT_OPEN = -0.6213（张开）
    # PUPPET_GRIPPER_JOINT_CLOSE = 1.4910（闭合）
    return _normalize(value, min_val=-0.6213, max_val=1.4910)


def _gripper_from_angular_inv(value):
    """_gripper_from_angular 的逆运算（训练时使用）。

    将 ALOHA 夹爪角度位置转换回 π 内部的夹爪角度位置。
    这个函数直接反转 _gripper_from_angular 的两个步骤。

    【转换流程（逆运算）】
    1. 反归一化: [0, 1] → [-0.6213, 1.4910]（ALOHA 夹爪角度范围）
    2. 去偏移: value - 0.5476 → 回到 π 内部夹爪角度空间

    Args:
        value: ALOHA 夹爪角度（归一化到 [0, 1]）

    Returns:
        π 内部夹爪角度（归一化到 [0, 1]）
    """
    # 步骤 1：反归一化到 ALOHA 角度范围
    value = _unnormalize(value, min_val=-0.6213, max_val=1.4910)
    # 步骤 2：减去 π 的夹爪偏置
    return value - 0.5476


# ============================================================================
# 顶层变换函数
# ============================================================================

def _decode_tron2(data: dict, *, adapt_to_pi: bool = False) -> dict:
    """Tron2 原始数据的解码函数。

    这是数据进入 Tron2Inputs 后调用的第一个函数，负责：
    1. 关节状态的预处理（根据 adapt_to_pi 决定是否转换空间）
    2. 图像的格式转换（数据类型 + 通道排列顺序）

    【图像格式说明】
    - 输入: [C, H, W] 格式，C=3 (RGB)，类型可能是 uint8 或 float
    - 输出: [H, W, C] 格式，类型为 uint8（模型输入标准）

    【状态格式说明】
    state = [左臂关节角7, 右臂关节角7, 左夹爪1, 右夹爪1] = 16 维

    Args:
        data: 原始输入数据字典
        adapt_to_pi: 是否将 ALOHA 关节空间转换为 π 内部运行时空间

    Returns:
        dict: 处理后的数据字典（图像为 HWC 格式，状态为 [16] 向量）
    """
    # ---- 处理关节状态 ----
    # state 结构：[left_arm_joint_angles(7), right_arm_joint_angles(7),
    #              left_arm_gripper(1),       right_arm_gripper(1)]
    # 总维度：7 + 7 + 1 + 1 = 16
    state = np.asarray(data["state"])
    state = _decode_state(state, adapt_to_pi=adapt_to_pi)

    # ---- 处理图像 ----
    def convert_image(img):
        """单张图像的转换函数。

        执行两个操作：
        1. 如果图像是浮点类型（值域 [0, 1]），转换为 uint8（值域 [0, 255]）
        2. 将通道维度从第一个位置移到最后一个位置 [C, H, W] → [H, W, C]

        Args:
            img: 输入图像，形状为 [C, H, W]

        Returns:
            处理后的图像，形状为 [H, W, C]，类型 uint8
        """
        img = np.asarray(img)
        # 浮点类型图像（值范围 [0.0, 1.0]）→ 转换为 uint8（值范围 [0, 255]）
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        # 通道重排：[channel, height, width] → [height, width, channel]
        # 这是因为模型内部（通常是 CNN 或 ViT）期望 HWC 格式
        return einops.rearrange(img, "c h w -> h w c")

    images = data["images"]
    images_dict = {name: convert_image(img) for name, img in images.items()}

    # ---- 更新数据字典 ----
    data["images"] = images_dict
    data["state"] = state
    return data


def _decode_state(state: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    """解码关节状态：从 ALOHA 空间转换到 π 内部运行时空间。

    当 adapt_to_pi=True 时执行两个操作：
    1. 关节方向翻转：乘以 _joint_flip_mask()
    2. 夹爪空间转换：使用 ALOHA 的逆运动学将线性夹爪转换为角度夹爪

    当 adapt_to_pi=False 时是恒等变换（Tron2 默认）。

    Args:
        state: 原始关节状态 [16]
        adapt_to_pi: 是否进行空间转换

    Returns:
        解码后的关节状态 [16]
    """
    if adapt_to_pi:
        # Step 1: 关节方向翻转
        # ALOHA 和 π 的关节正方向定义不同，某些关节需要取反
        state = _joint_flip_mask() * state

        # Step 2: 夹爪线性位置 → 角度位置
        # 状态向量的索引 6 是左夹爪，13 是右夹爪
        # 这两个位置需要从线性位移转换为关节角度
        state[[6, 13]] = _gripper_to_angular(state[[6, 13]])

    return state


def _encode_actions(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    """编码动作：从 π 内部运行时空间转换回 ALOHA 空间（推理时使用）。

    这是 _decode_state 的"逆操作"，用于推理阶段：
    - 模型在 π 内部空间中进行预测
    - 但机器人底层控制器期望 ALOHA 空间的动作
    - 所以需要将预测结果转换回 ALOHA 空间

    当 adapt_to_pi=False 时是恒等变换（Tron2 默认）。

    Args:
        actions: 模型预测的动作 [action_horizon, 16]
        adapt_to_pi: 是否进行空间转换

    Returns:
        编码后的动作 [action_horizon, 16]
    """
    if adapt_to_pi:
        # Step 1: 关节方向取反（恢复 ALOHA 的符号约定）
        actions = _joint_flip_mask() * actions

        # Step 2: 夹爪角度 → ALOHA 夹爪角度（角度空间内的范围转换）
        actions[:, [6, 13]] = _gripper_from_angular(actions[:, [6, 13]])

    return actions


def _encode_actions_inv(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    """编码动作的逆变换：从 ALOHA 空间转换到 π 内部运行时空间（训练时使用）。

    这个函数是 _encode_actions 的逆运算，用于训练阶段：
    - 数据集中的动作标签是 ALOHA 格式
    - 但模型在 π 内部空间中进行训练
    - 所以需要将标签转换到 π 空间

    当 adapt_to_pi=False 时是恒等变换（Tron2 默认）。

    注意：_encode_actions_inv 和 _encode_actions 形成一对互逆变换：
        _encode_actions_inv:  ALOHA 空间 → π 空间  （训练标签预处理）
        _encode_actions:      π 空间 → ALOHA 空间  （推理输出后处理）

    Args:
        actions: 数据集中的动作标签 [action_horizon, 16]
        adapt_to_pi: 是否进行空间转换

    Returns:
        转换后的动作标签 [action_horizon, 16]
    """
    if adapt_to_pi:
        # Step 1: 关节方向取反（同 _encode_actions）
        actions = _joint_flip_mask() * actions

        # Step 2: 夹爪角度反向转换
        # 注意这里使用的是 _gripper_from_angular_inv 而不是 _gripper_to_angular
        # 因为数据集中的标签已经是经过 ALOHA 运行时处理的，
        # 夹爪值已经处于"角度空间但 ALOHA 范围"，
        # 需要反向转换到 π 内部范围
        actions[:, [6, 13]] = _gripper_from_angular_inv(actions[:, [6, 13]])

    return actions

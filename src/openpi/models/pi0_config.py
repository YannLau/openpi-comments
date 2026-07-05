# ============================================================
#  src/openpi/models/pi0_config.py — π₀ / π₀.₅ 模型配置
#
#  本文件定义了 π₀（flow matching）和 π₀.₅（升级版 flow matching）
#  模型的配置类。主要职责：
#
#    1. Pi0Config — 模型超参数容器（视觉编码器变体、动作专家变体、
#       动作维度、动作预测步数、是否启用 π₀.₅ 特性等）
#    2. inputs_spec() — 定义模型输入/输出的形状与 dtype 规范
#       （用于 FakeDataset 生成假数据、jax.eval_shape 形状推断）
#    3. get_freeze_filter() — 根据 LoRA 配置确定哪些参数在训练中冻结
#       （控制微调时的可训练参数范围）
#
#  π₀ vs π₀.₅ 核心差异：
#    - π₀.₅ 的状态输入作为离散语言 token 的一部分（而非连续 suffix）
#    - π₀.₅ 的动作专家使用 adaRMSNorm 注入 flow matching 时间步
#    - π₀.₅ 默认更大的 max_token_len (200 vs 48)
# ============================================================

import dataclasses  # 创建不可变的数据类（frozen=True）
from typing import TYPE_CHECKING  # 仅用于类型检查时的条件导入，避免循环依赖

import flax.nnx as nnx       # Flax NNX — JAX 神经网络 API
import jax                   # JAX 核心
import jax.numpy as jnp      # JAX 版 NumPy
from typing_extensions import override  # 显式标记方法重写

from openpi.models import model as _model   # 基础模型定义（BaseModelConfig, Observation, Actions 等）
import openpi.models.gemma as _gemma        # Gemma 模型变体（gemma_2b, gemma_300m 等字符串类型定义）
from openpi.shared import array_typing as at  # 数组类型检查与标注
import openpi.shared.nnx_utils as nnx_utils    # Flax NNX 工具函数（PathRegex 等）

# 条件导入：仅在类型检查时导入 Pi0 模型类
# 避免循环依赖（pi0.py 导入了 pi0_config.py，pi0_config.py 又导入 pi0.py）
if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    """π₀ / π₀.₅ 模型的配置类。

    π₀ 是 Physical Intelligence 的开源 VLA（Vision-Language-Action）模型，
    使用 flow matching 生成连续动作。π₀.₅ 是升级版，改进了状态输入
    的处理方式并引入了 adaRMSNorm。

    关键设计决策：
      - 两个 Gemma 子网络：大模型（paligemma_variant）处理视觉-语言编码，
        小模型（action_expert_variant）专门解码动作
      - 支持 LoRA 微调：通过 variant 名称中的 "lora" 标识开启
      - 支持 π₀.₅ 模式：通过 pi05=True 启用
      - 冻结策略：根据 LoRA 配置自动决定哪些参数需要冻结
    """

    # ---- 数值精度 ----
    dtype: str = "bfloat16"  # 模型计算精度（bfloat16 兼顾精度与显存效率）

    # ---- 子网络变体选择 ----
    # paligemma_variant: 视觉-语言主干网络的 Gemma 变体
    #   可选值: "gemma_2b"（默认，完整微调）、"gemma_2b_lora"（LoRA 微调）
    paligemma_variant: _gemma.Variant = "gemma_2b"

    # action_expert_variant: 动作专家网络的 Gemma 变体
    #   可选值: "gemma_300m"（默认）、"gemma_300m_lora"
    #   动作专家是一个较小的模型，专门负责从视觉-语言特征中解码动作序列
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # ---- 动作空间配置 ----
    action_dim: int = 32           # 每个时间步的动作维度（例如 7-DoF 关节角 + 夹爪）
    action_horizon: int = 50       # 一次预测的动作步数（action chunking）

    # ---- 文本 token 配置 ----
    max_token_len: int = None  # type: ignore  # 语言 prompt 的最大 token 数
    #   π₀：   默认 48（较短，适合简单指令）
    #   π₀.₅：默认 200（较长，支持更复杂的指令）

    # ---- π₀.₅ 特性开关 ----
    # Pi05 与 Pi0 有两个核心区别：
    # 1) 状态输入作为离散语言 token 的一部分，而非连续 suffix
    # 2) 动作专家使用 adaRMSNorm 注入 flow matching 时间步
    pi05: bool = False  # 是否启用 π₀.₅ 模式

    # 是否使用离散状态输入（将状态通过 tokenizer 转为 token ID）
    # 此配置不由模型直接使用，而是由 ModelTransformFactory 读取
    discrete_state_input: bool = None  # type: ignore

    # ---- PyTorch 编译模式 ----
    # 仅在 PyTorch 推理时使用，控制 torch.compile 的优化级别
    # "max-autotune" = 最大性能优化（编译时间长）
    # None = 禁用编译
    pytorch_compile_mode: str | None = "max-autotune"

    def __post_init__(self):
        """初始化后处理：设置默认值并验证配置。

        Python 的 dataclass __post_init__ 在 __init__ 完成后自动调用。
        由于类被标记为 frozen=True，需要用 object.__setattr__ 绕过冻结限制。
        """
        # 设置 max_token_len 默认值（如果用户未指定）
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)

        # 设置 discrete_state_input 默认值（与 pi05 一致）
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)

        # 验证 pytorch_compile_mode 的合法性（如果设置了的话）
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ], f"无效的 pytorch_compile_mode: {self.pytorch_compile_mode}"

    @property
    @override
    def model_type(self) -> _model.ModelType:
        """返回模型类型枚举。

        这用于训练脚本区分 π₀（flow matching）和 π₀.₅（升级版 flow matching），
        从而在日志、检查点命名等处做不同处理。
        """
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        """根据配置创建一个 π₀/π₀.₅ 模型实例。

        Pi0 模型的构造函数需要：
          1. 配置对象（self）
          2. 随机数生成器（nnx.Rngs）

        使用延迟导入避免循环依赖：pi0.py 会导入 Pi0Config，
        而 Pi0Config.create() 需要 Pi0 类。

        参数：
          rng: JAX 随机数键，用于模型参数的随机初始化

        返回：
          一个初始化的 Pi0 模型实例
        """
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        """定义模型输入/输出的形状（shape）和数据类型（dtype）规范。

        这些规范用于：
          1. FakeDataset：生成形状匹配的假数据用于调试
          2. jax.eval_shape：在不实际计算的情况下获取参数形状
          3. 验证数据加载器产生的 batch 形状是否正确

        参数：
          batch_size: batch 大小（默认 1）

        返回：
          (Observation, Actions) 元组，其中每个字段都是 jax.ShapeDtypeStruct
          （仅包含形状和类型信息，不包含实际数据）
        """
        # ---- 图像输入规范 ----
        # 图像形状: (batch, 224, 224, 3)，RGB 三通道
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        # 图像掩码：标记哪些图像存在（用于处理缺失的摄像头视图）
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            # Observation 是模型的标准输入格式，包含：
            #   - images: 多个摄像头的 RGB 图像（字典，键为摄像头名称）
            #   - image_masks: 图像有效掩码（处理缺失视图）
            #   - state: 机器人状态（如关节角、末端位姿等）
            #   - tokenized_prompt: 语言指令的 token ID 序列
            #   - tokenized_prompt_mask: token 有效掩码（处理 padding）
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,          # 基础摄像头
                    "left_wrist_0_rgb": image_spec,    # 左腕摄像头
                    "right_wrist_0_rgb": image_spec,   # 右腕摄像头
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),       # 机器人状态
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),  # token 化指令
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),  # 指令掩码
            )

        # ---- 动作标签规范 ----
        # 形状: (batch, action_horizon, action_dim)
        # 即每个样本预测未来 action_horizon 步动作，每步 action_dim 维
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """根据模型配置返回参数冻结过滤器。

        这是 LoRA 微调的关键函数，决定哪些参数在训练中"冻结"（不更新梯度）。

        冻结逻辑（按优先级）：
          1. 仅 paligemma 使用 LoRA → 冻结所有 Gemma 参数，但排除 action_expert
             → 只微调 action_expert + LoRA 适配器
          2. 仅 action_expert 使用 LoRA → 冻结 action_expert 参数
             → 只微调 paligemma + LoRA 适配器
          3. 两者都用 LoRA → 冻结两个 Gemma 的参数
             → 只微调 LoRA 适配器
          4. 两者都不用 LoRA → 不冻结任何参数（全量微调）
          5. 在任何 LoRA 场景下，所有 LoRA 适配器参数始终可训练

        注意：nnx.Nothing 表示"不匹配任何参数"（即不冻结任何参数），
        而非 nnx.All(nnx.Nothing) 这种错误的组合。
        """
        filters = []       # 收集需要冻结的参数过滤器
        has_lora = False   # 标记是否使用了任何 LoRA

        # 定义路径正则表达式过滤器
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")             # 匹配所有 Gemma 参数
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")  # 匹配 action_expert 参数

        # ---- 情况 1 & 3：paligemma 使用 LoRA ----
        if "lora" in self.paligemma_variant:
            filters.append(gemma_params_filter)  # 冻结 paligemma 参数
            if "lora" not in self.action_expert_variant:
                # 情况 1：action_expert 未使用 LoRA（即全量微调）
                # 所以 action_expert 参数不应该被冻结
                # 逻辑：冻结 gemma_params ∩ !action_expert_params
                filters.append(nnx.Not(action_expert_params_filter))
                # 等效于：冻结 paligemma 参数，但不冻结 action_expert 参数
            # 情况 3：两者都用 LoRA，则直接冻结所有 Gemma 参数
            has_lora = True

        # ---- 情况 2：仅 action_expert 使用 LoRA ----
        elif "lora" in self.action_expert_variant:
            filters.append(action_expert_params_filter)  # 冻结 action_expert 参数
            has_lora = True

        # ---- 所有 LoRA 场景下：确保 LoRA 参数可训练 ----
        if has_lora:
            # nnx.Not(.*lora.*) 表示"不匹配 LoRA 参数"
            # 与前面的过滤器组合为 nnx.All(...) 后，LoRA 参数不会被任何过滤器捕获
            # 因此它们不属于冻结集合，保持可训练
            filters.append(nnx.Not(nnx_utils.PathRegex(".*lora.*")))

        # ---- 返回最终的过滤器 ----
        if not filters:
            # 情况 4：全量微调，不冻结任何参数
            return nnx.Nothing
        # 组合所有过滤器：参数必须同时满足所有条件才被冻结
        return nnx.All(*filters)

"""
============================================================
  policy_config.py — 策略（Policy）构造器

  本文件提供了一个核心工厂函数 create_trained_policy()，
  负责从"训练好的检查点（checkpoint）"构建出一个可用的推理策略（Policy）。

  简单来说，这个函数就是"把硬盘上的训练产物变成能用的推理引擎"：
    训练检查点 + 训练配置 → 组装变换流水线 → 完整的 Policy 对象

  它要做的事包括：
    - 判断模型是 JAX 还是 PyTorch（通过检测文件后缀）
    - 加载模型权重
    - 加载归一化统计信息（norm stats）
    - 组装完整的输入/输出变换流水线
    - 创建并返回 Policy 对象

  这个函数的调用者是：
    - scripts/serve_policy.py（启动策略推理服务器）
    - scripts/eval_policy.py（评估策略性能）
    - 任何需要加载已训练策略的脚本
============================================================
"""

import logging  # 日志记录，方便在控制台看到加载进度
import os  # 路径操作（检查文件是否存在）
import pathlib  # 跨平台路径处理（Path 对象）
from typing import Any  # 类型提示

import jax.numpy as jnp  # JAX 的 NumPy 变体（用于指定数据类型）

import openpi.models.model as _model  # 模型基类（用于加载参数和模型结构）
import openpi.policies.policy as _policy  # 我们刚注释过的 Policy 类
import openpi.shared.download as download  # 下载工具（支持从 GCS/HTTP 自动下载检查点）
from openpi.training import checkpoints as _checkpoints  # 检查点管理（加载 norm stats 等）
from openpi.training import config as _config  # 训练配置（TrainConfig，包含模型、数据等所有配置）
import openpi.transforms as transforms  # 数据变换模块（归一化、反归一化、标记化等）


# ============================================================================
# 函数：create_trained_policy
#
# 这是整个系统的"策略工厂"函数。它的职责非常清晰：
#   输入：训练配置 + 检查点目录
#   输出：一个可以直接调用的 Policy 对象
#
# 从软件架构的角度看，这个函数处于"训练阶段"和"推理阶段"的交界处：
#   - 训练阶段产物：检查点（权重）、norm stats（归一化统计）、训练配置
#   - 推理阶段产物：Policy 对象（可直接用于 infer()）
#
# 命名惯例说明：
#   "trained"（训练好的）——强调这是从已训练完成的检查点恢复的，
#   区别于未训练的裸模型。这个区分很重要，因为裸模型不能直接用来做推理。
# ============================================================================
def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    # 以下是关键字参数（调用时必须显式写出参数名）
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
) -> _policy.Policy:
    """从训练好的检查点创建一个推理策略（Policy）。

    这是将训练产物（检查点权重 + 配置）转化为可用策略的主要入口点。
    函数会自动完成文件格式检测、模型加载、变换流水线组装等工作。

    Args:
        train_config:     训练配置对象。包含了模型架构、数据配置、优化器等
                          所有训练时使用的配置信息。
                          用于决定：
                          - 用什么模型结构加载权重
                          - 用什么数据变换流水线
                          - 用什么归一化方式

        checkpoint_dir:   检查点目录路径。包含：
                          - params/      （JAX 模型参数目录）
                          - model.safetensors （PyTorch 模型权重文件）
                          - assets/      （归一化统计信息等附属文件）
                          注意：路径可以是 GCS 远程路径（如 gs://bucket/...），
                          函数会自动下载到本地缓存。

        repack_transforms: 可选的"预变换"操作。
                           在标准的输入变换之前执行。
                           用于需要调整输入字段结构的场景。
                           例如：将不同相机名的图像字段重命名为统一的 "image"。
                           Group 对象包含 inputs 和 outputs 两个变换序列。

        sample_kwargs:     传递给 model.sample_actions 方法的额外参数。
                           常用选项：
                           - num_steps: 扩散/流匹配的采样步数（默认通常 10 步）
                           - noise: 自定义噪声（用于可控生成或复现）
                           - guidance_scale: 无分类器引导强度（类似扩散模型的 CFG）
                           如果为 None，会使用训练配置中的默认值。

        default_prompt:    默认文本指令。当输入中没有 "prompt" 字段时，
                           会用这个默认提示替代。
                           典型值：
                           - "pick up the cube"（拿起方块）
                           - "place the object in the bin"（把物体放进箱子）
                           如果为 None，则必须由调用者在推理时提供 prompt。

        norm_stats:        归一化统计信息字典。包含每个字段的 mean（均值）
                           和 std（标准差）或 quantiles（分位数）。
                           如果为 None，函数会自动从检查点目录加载。
                           格式示例：
                           {
                               "state": NormStats(mean=..., std=...),
                               "actions": NormStats(mean=..., std=...),
                           }

        pytorch_device:    PyTorch 模型的运行设备。
                           取值：
                           - "cpu"       → CPU 推理
                           - "cuda"      → 任意可用 GPU
                           - "cuda:0"    → 第一块 GPU
                           - None        → 自动选择（GPU 优先，退化为 CPU）
                           仅在模型是 PyTorch 格式时生效。

    Returns:
        配置完毕的 Policy 对象。可以直接调用 policy.infer(obs) 进行推理。

    Raises:
        ValueError: 当需要使用归一化统计信息（norm stats）但无法确定
                    资产 ID（asset_id）时抛出。

    Note:
        JAX vs PyTorch 的自动检测方式：
        函数通过检查检查点目录中是否存在 "model.safetensors" 文件来判断。
        safetensors 是 PyTorch 社区常用的安全权重格式。
        如果存在 → PyTorch 模型；不存在（但有 params/ 目录）→ JAX 模型。
    """
    # ========================================================================
    # 第 1 步：配置默认值 & 处理路径
    # ========================================================================

    # 如果没有提供 repack_transforms，使用空的变换组（什么都不做）
    repack_transforms = repack_transforms or transforms.Group()

    # maybe_download 支持从 GCS（Google Cloud Storage）或 HTTP URL 下载。
    # 如果 checkpoint_dir 是本地路径，直接返回原路径。
    # 如果是 "gs://bucket/path" 或 "https://..."，会下载到 ~/.cache/openpi/ 中。
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    # ========================================================================
    # 第 2 步：自动检测模型类型（JAX vs PyTorch）
    #
    # 通过文件系统探测来判断：
    #   - model.safetensors 存在 → PyTorch 模型
    #   - 不存在              → JAX 模型
    #
    # safetensors 是 Hugging Face 推出的安全张量存储格式。
    # 它类似于 PyTorch 的 .pt 文件，但更安全（没有 pickle 反序列化漏洞）。
    # JAX 的 checkpoint 则使用 orbax 格式，存储在 params/ 子目录中。
    # ========================================================================
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    is_pytorch = os.path.exists(weight_path)

    # ========================================================================
    # 第 3 步：加载模型
    # ========================================================================
    logging.info("Loading model...")
    if is_pytorch:
        # ---------- PyTorch 分支 ----------
        # load_pytorch() 是 BaseModel 的一个类方法，用于构建 PyTorch 模型并加载权重。
        # 它会：
        #   1. 根据 train_config 中的配置创建模型实例
        #   2. 从 safetensors 文件加载权重
        #   3. 返回模型实例
        model = train_config.model.load_pytorch(train_config, weight_path)

        # to_bfloat16_for_selected_params() 将模型的部分参数转换为 bfloat16 精度。
        #
        # 为什么要转 bfloat16？
        #   - 节省显存：bfloat16 占 2 字节，float32 占 4 字节，直接省一半
        #   - 保持精度：bfloat16 的指数位与 float32 相同（8 位），
        #     所以不会像 float16 那样容易数值溢出
        #   - 现代 GPU（A100, H100）对 bfloat16 有专门的加速单元
        #
        # "selected_params" 表示只对特定层转换（如 Gemma 语言模型部分），
        # 而图像编码器（SigLIP/ViT）可能保持 float32 以保留图像特征精度。
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        # ---------- JAX 分支 ----------
        # restore_params() 从检查点目录恢复模型参数。
        # checkpoint_dir / "params" 是 orbax 保存的参数目录。
        # dtype=jnp.bfloat16 指定以 bfloat16 精度加载。
        model = train_config.model.load(
            _model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)
        )

    # ========================================================================
    # 第 4 步：构建数据配置（DataConfig）
    #
    # train_config.data 是一个 DataConfigFactory。
    # .create() 方法会根据 assets_dirs 和 model 创建真正的 DataConfig 对象。
    #
    # DataConfig 包含了：
    #   - data_transforms：   数据层变换（如 delta 动作转换、状态填充）
    #   - model_transforms：  模型层变换（如文本标记化、图像预处理）
    #   - use_quantile_norm： 是否使用分位数归一化
    #   - asset_id：          资产 ID，用于加载 norm stats
    #
    # 注意：这里不是在创建数据加载器（dataloader），而是在做变换流水线的配置。
    # 变换流水线会在后面的 Policy 构造函数中组装。
    # ========================================================================
    data_config = train_config.data.create(
        train_config.assets_dirs, train_config.model
    )

    # ========================================================================
    # 第 5 步：加载（或获取）归一化统计信息（Norm Stats）
    #
    # 归一化是机器人学习中的关键步骤。
    # 不同类型的传感器数据数量级差异很大，例如：
    #   - 关节角度：通常在 -3.14 ~ 3.14 弧度之间
    #   - 图像像素：0 ~ 255 （通常还会归一化到 0 ~ 1）
    #   - 末端速度：可能达到几十 m/s
    #
    # 如果不做归一化，模型会很难训练（大数值特征会主导梯度更新）。
    # Norm Stats 记录了每个字段的均值和标准差（或分位数），
    # 用于统一缩放到模型友好的范围（通常是接近标准正态分布）。
    # ========================================================================
    if norm_stats is None:
        # 用户没有提供 norm_stats，我们从检查点目录中的 assets 子目录加载。
        #
        # 为什么要从检查点加载而不是从配置？
        #   配置中的 assets_dirs 是训练时的路径，可能包含多个候选。
        #   而检查点中的 assets 是训练时实际使用的归一化统计信息，
        #   用这个可以确保推理时使用的归一化与训练时完全一致。
        #   否则如果使用不同统计信息，会导致输入分布偏移，影响效果。
        if data_config.asset_id is None:
            raise ValueError(
                "Asset id is required to load norm stats."
            )
        # load_norm_stats() 从指定目录加载指定 asset_id 的归一化统计信息
        norm_stats = _checkpoints.load_norm_stats(
            checkpoint_dir / "assets", data_config.asset_id
        )

    # ========================================================================
    # 第 6 步：自动选择 PyTorch 设备
    #
    # 如果是 PyTorch 模型但用户没有指定设备：
    #   - 如果有 CUDA GPU 可用 → 使用 "cuda"
    #   - 否则 → 使用 "cpu"
    #
    # 为什么不在没有 GPU 时直接报错？
    #   有些 PyTorch 模型可以在 CPU 上跑（虽然慢），
    #   对于快速验证或小模型，CPU 也够用。
    # ========================================================================
    if is_pytorch and pytorch_device is None:
        try:
            import torch  # 只在需要时才导入 PyTorch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            # 连 PyTorch 都没装，那就只能跑 CPU 了
            pytorch_device = "cpu"

    # ========================================================================
    # 第 7 步：组装 Policy 对象
    #
    # 这是最关键的一步——构建完整的输入/输出变换流水线。
    #
    # 变换（Transform）的执行顺序非常重要！
    #   输入方向（数据流向模型）：
    #     repack_transforms.inputs     → 可能重组输入字段
    #     InjectDefaultPrompt          → 添加默认语言指令
    #     data_config.data_transforms  → 数据层变换（如 delta 动作转换）
    #     Normalize                    → 归一化到标准范围
    #     data_config.model_transforms → 模型层变换（如文本标记化、图像缩放）
    #
    #   输出方向（模型流回数据）：
    #     data_config.model_transforms → 模型输出后处理
    #     Unnormalize                  → 反归一化回真实范围
    #     data_config.data_transforms  → 数据层反变换
    #     repack_transforms.outputs    → 输出字段重组
    #
    # 这种对称的设计（输入变换 - 模型 - 输出反变换）是经典的编码器-解码器模式。
    # 模型在"归一化后的标准空间"中工作，不关心原始数据的尺度。
    # ========================================================================
    return _policy.Policy(
        model,
        # ---- 输入变换（原始数据 → 模型可接受格式） ----
        transforms=[
            # 1. 字段预重组（如果提供了 repack_transforms）
            *repack_transforms.inputs,
            # 2. 如果输入没有 prompt 字段，注入默认文本指令
            transforms.InjectDefaultPrompt(default_prompt),
            # 3. 数据层变换（如增量动作转换、状态填充等）
            *data_config.data_transforms.inputs,
            # 4. 归一化（将不同尺度的传感器数据映射到标准范围）
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            # 5. 模型层变换（文本标记化、图像预处理等）
            *data_config.model_transforms.inputs,
        ],
        # ---- 输出变换（模型输出 → 可执行动作） ----
        output_transforms=[
            # 1. 模型输出的后处理变换
            *data_config.model_transforms.outputs,
            # 2. 反归一化（将模型输出的标准范围数据还原到真实物理范围）
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            # 3. 数据层反变换（如将增量动作转换为绝对位置）
            *data_config.data_transforms.outputs,
            # 4. 输出字段重组（如果提供了 repack_transforms）
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,  # 从训练配置中获取策略元数据（版本、描述等）
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
    )

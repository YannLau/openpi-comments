#!/usr/bin/env python3
"""
加载 JAX 模型并打印所有参数键名，可选地将其转换为 PyTorch 格式。

本脚本使用 orbax 库加载 JAX 模型检查点（checkpoint），支持两种功能：
1. 打印所有参数的层次化键名，方便检查和调试模型结构
2. 使用我们的 PI0Pytorch 模型将 JAX 模型转换为 PyTorch 格式

核心概念说明：
- JAX / Flax: Google 开发的深度学习框架，采用函数式编程范式
- orbax: JAX 生态中的检查点管理库
- PaliGemma: Google 的多模态视觉-语言模型（Vision-Language Model, VLM）
- Gemma: Google 的开源语言模型系列，PaliGemma 使用 Gemma 作为文本解码器
- SafeTensors: Hugging Face 推出的安全且快速的张量序列化格式
- π₀ (pi0): Physical Intelligence 的流匹配（flow matching）VLA 模型
- π₀.₅ (pi05): 升级版 π₀，使用"知识绝缘"（knowledge insulation）技术

用法:
    # 仅检查参数键名（不转换）：
    python examples/convert_jax_model_to_pytorch.py --checkpoint_dir /path/to/checkpoint --inspect_only

    # 转换为 PyTorch 格式：
    python examples/convert_jax_model_to_pytorch.py --checkpoint_dir /path/to/checkpoint --output_path /path/to/output

示例:
    # pi0_droid 模型
    python examples/convert_jax_model_to_pytorch.py --checkpoint_dir /home/$USER/.cache/openpi/openpi-assets/checkpoints/pi0_droid --output_path /home/$USER/.cache/openpi/openpi-assets/checkpoints/pi0_droid_pytorch

    # pi0_aloha_sim 模型
    python examples/convert_jax_model_to_pytorch.py --checkpoint_dir /home/$USER/.cache/openpi/openpi-assets/checkpoints/pi0_aloha_sim --output_path /home/$USER/.cache/openpi/openpi-assets/checkpoints/pi0_aloha_sim_pytorch

    # pi05_droid 模型
    python examples/convert_jax_model_to_pytorch.py --checkpoint_dir /home/$USER/.cache/openpi/openpi-assets/checkpoints/pi05_droid --output_path /home/$USER/.cache/openpi/openpi-assets/checkpoints/pi05_droid_pytorch
"""

import json  # JSON 读写，用于保存模型配置信息
import os  # 文件和路径操作
import pathlib  # 面向对象的路径操作库（比 os.path 更现代）
import shutil  # 高级文件操作（复制、移动、删除目录树）
from typing import Literal  # 类型提示：限制变量只能取特定字符串值

# ========== JAX/Flax 相关导入 ==========
from flax.nnx import traversals  # Flax NNX 的遍历工具，用于展平嵌套参数字典

# ========== 数值计算库 ==========
import numpy as np  # NumPy：Python 最流行的数值计算库
import orbax.checkpoint as ocp  # orbax：JAX 的检查点管理库，负责保存/加载模型权重
import safetensors  # SafeTensors：安全的张量序列化格式（比 pickle 更安全、更快）
import torch  # PyTorch：另一个主流深度学习框架，本脚本的转换目标

# ========== 参数解析 ==========
import tyro  # tyro：类型安全的命令行参数解析库（比 argparse 更简洁）

# ========== openpi 内部模块导入 ==========
import openpi.models.gemma  # Gemma 语言模型配置（用于 action expert 部分）
import openpi.models.model  # 基础模型定义：BaseModel, Observation, Actions 等
import openpi.models.pi0_config  # π₀ 模型配置类 Pi0Config
import openpi.models_pytorch.pi0_pytorch  # π₀ 的 PyTorch 实现
from openpi.training import utils  # 训练工具函数（如 array_tree_to_info）
import openpi.training.config as _config  # 所有训练配置定义（_CONFIGS 列表）


def slice_paligemma_state_dict(state_dict, config):
    """
    将 PaliGemma 的 JAX 格式参数转换为 PyTorch 格式。

    背景知识：
    - JAX 和 PyTorch 的参数命名规范完全不同，需要一一映射
    - JAX 使用 Flax 的命名约定（如 "img/embedding/kernel"），权重布局通常是 [out, in]
    - PyTorch 使用 HuggingFace 的命名约定（如 "vision_tower.vision_model.embeddings.patch_embedding.weight"），
      权重布局通常是 [in, out]
    - 因此转换时需要：① 重命名键名  ② 转置（transpose）权重矩阵

    Args:
        state_dict: JAX 参数字典（键名以 "/" 分隔的层次结构）
        config: PaliGemmaConfig 对象，包含 vision_config 和 text_config

    Returns:
        final_state_dict: 转换后的 PyTorch 参数字典（PaliGemma 部分）
        expert_dict: 属于 action expert 的参数（需要单独处理）
    """

    # 判断是否存在 "/value" 后缀 —— 这是 JAX 中某些参数的分片后缀（来自 FSDP 分片训练）
    # 如果 "img/embedding/kernel/value" 存在，说明参数被分片存储了
    suffix = "/value" if "img/embedding/kernel/value" in state_dict else ""

    # ====================================================================
    # 1. 图像编码器的 Patch Embedding（将图像分割成小块并嵌入到向量空间）
    # ====================================================================
    # JAX 中的键名（Flax 风格）：img/embedding/kernel
    # PyTorch 中的键名（HF 风格）：paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight
    #
    # transpose(3, 2, 0, 1) 说明：
    #   JAX 卷积权重形状通常是 [H, W, C_in, C_out]（在 Flax 中）
    #   PyTorch 卷积权重形状是 [C_out, C_in, H, W]
    #   所以要将索引 3 移到最前面，索引 2 移到第二位，然后 0,1 保持不变
    #   即 (3, 2, 0, 1) 的转置顺序
    jax_key = f"img/embedding/kernel{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key).transpose(3, 2, 0, 1)

    # 偏置（bias）不需要转置，直接复制
    jax_key = f"img/embedding/bias{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.bias"
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    # ====================================================================
    # 2. 位置嵌入（Positional Embedding）：告诉模型每个图像块的空间位置
    # ====================================================================
    jax_key = f"img/pos_embedding{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.position_embedding.weight"
    # reshape(-1, hidden_size)：JAX 中位置嵌入可能是 (num_patches, 1, hidden_size) 或其他形状，
    # 需要展平成 (num_patches, hidden_size) 以匹配 PyTorch
    state_dict[pytorch_key] = state_dict.pop(jax_key).reshape(-1, config.vision_config.hidden_size)

    # ====================================================================
    # 3. ViT 编码器层参数提取
    # ====================================================================
    # PaliGemma 的 ViT (Vision Transformer) 有 27 层 transformer 编码器块。
    # 在 Flax（JAX）中，所有层的参数被合并存储在一起（stacked），形状为 [27, ...]。
    # 而在 PyTorch 的 HuggingFace 实现中，每层参数是分开存储的。
    # 所以这里先通过 pop() 一次性取出所有层的合并参数，然后在下面的循环中逐层拆分。
    #
    # 各层的命名含义：
    #   LayerNorm_0: 注意力前的层归一化（Pre-Attention LayerNorm）
    #   LayerNorm_1: MLP 前的层归一化（Pre-FFN LayerNorm）
    #   MlpBlock_0/Dense_0: MLP 的第一层（通常是升维/门控层）
    #   MlpBlock_0/Dense_1: MLP 的第二层（降维层）
    #   MultiHeadDotProductAttention_0: 多头注意力机制
    encoderblock_layernorm0_scale = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_0/scale{suffix}")
    encoderblock_layernorm0_bias = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_0/bias{suffix}")
    encoderblock_layernorm1_scale = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_1/scale{suffix}")
    encoderblock_layernorm1_bias = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_1/bias{suffix}")

    encoderblock_mlp_dense0_kernel = state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_0/kernel{suffix}")
    encoderblock_mlp_dense0_bias = state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_0/bias{suffix}")
    encoderblock_mlp_dense1_kernel = state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_1/kernel{suffix}")
    encoderblock_mlp_dense1_bias = state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_1/bias{suffix}")

    # 注意力机制的 K(Key)、V(Value)、Q(Query)、Out(Output) 投影
    encoderblock_attention_0_key_kernel = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/kernel{suffix}"
    )
    encoderblock_attention_0_key_bias = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/bias{suffix}"
    )
    encoderblock_attention_0_value_kernel = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/kernel{suffix}"
    )
    encoderblock_attention_0_value_bias = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/bias{suffix}"
    )
    encoderblock_attention_0_query_kernel = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/kernel{suffix}"
    )
    encoderblock_attention_0_query_bias = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/bias{suffix}"
    )
    encoderblock_attention_0_out_kernel = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/kernel{suffix}"
    )
    encoderblock_attention_0_out_bias = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/bias{suffix}"
    )

    # ====================================================================
    # 4. 逐层分配参数（将合并存储的 JAX 参数拆分到 PyTorch 的逐层结构中）
    # ====================================================================
    # config.vision_config.num_hidden_layers = 27（PaliGemma 的 ViT 编码器层数）
    for i in range(config.vision_config.num_hidden_layers):
        # LayerNorm 的 scale 在 PyTorch 中叫 weight，需要 .transpose()（实际上 1D 向量的 transpose 是无操作）
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm1.weight"
        ] = encoderblock_layernorm0_scale[i].transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm1.bias"
        ] = encoderblock_layernorm0_bias[i]
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm2.weight"
        ] = encoderblock_layernorm1_scale[i].transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm2.bias"
        ] = encoderblock_layernorm1_bias[i]

        # MLP: fc1（升维层，相当于 JAX 的 Dense_0）和 fc2（降维层，相当于 JAX 的 Dense_1）
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc1.weight"
        ] = encoderblock_mlp_dense0_kernel[i].transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc1.bias"
        ] = encoderblock_mlp_dense0_bias[i]
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc2.weight"
        ] = encoderblock_mlp_dense1_kernel[i].transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc2.bias"
        ] = encoderblock_mlp_dense1_bias[i]

        # 自注意力（Self-Attention）：Q、K、V、Out 四个投影矩阵
        # reshape(-1, hidden_size)：JAX 中注意力权重可能是合并的 [heads, dim_per_head, hidden] 形状，
        # 需要展平成 [heads * dim_per_head, hidden] 以匹配 PyTorch 的线性层
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.k_proj.weight"
        ] = encoderblock_attention_0_key_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.k_proj.bias"
        ] = encoderblock_attention_0_key_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.v_proj.weight"
        ] = encoderblock_attention_0_value_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.v_proj.bias"
        ] = encoderblock_attention_0_value_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj.weight"
        ] = encoderblock_attention_0_query_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj.bias"
        ] = encoderblock_attention_0_query_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.out_proj.weight"
        ] = encoderblock_attention_0_out_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.out_proj.bias"
        ] = encoderblock_attention_0_out_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)

    # ====================================================================
    # 5. 编码器最后的层归一化（Post-LayerNorm）
    # ====================================================================
    jax_key = f"img/Transformer/encoder_norm/scale{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.post_layernorm.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key).transpose()

    jax_key = f"img/Transformer/encoder_norm/bias{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.post_layernorm.bias"
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    # ====================================================================
    # 6. 多模态投影器（Multi-modal Projector）
    #    将视觉编码器的输出投影到语言模型的嵌入空间
    #    这样语言模型就能"理解"图像信息
    # ====================================================================
    jax_key = f"img/head/kernel{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key).transpose()

    jax_key = f"img/head/bias{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.bias"
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    # ====================================================================
    # 7. 文本解码器（Gemma 语言模型）的 Token 嵌入层
    #    将文本 token 映射到向量空间
    # ====================================================================
    jax_key = f"llm/embedder/input_embedding{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    # ====================================================================
    # 8. 输入层（attention）的 Einsum 参数提取（合并存储的 JAX 格式）
    #
    # Einsum（Einstein Summation）是 Flax/JAX 中实现注意力计算的一种高效方式。
    # 在 JAX 中，Q/K/V/O 投影通过 einsum 操作实现（比单独的线性层更灵活）。
    # 需要将它们分解为标准的 PyTorch 线性层权重。
    #
    # 名称解析：
    #   q_einsum: Query 投影的 einsum 权重
    #   kv_einsum: Key 和 Value 投影的合并 einsum 权重
    #   attn_vec_einsum: 注意力输出投影（合并多头后）的 einsum 权重
    # ====================================================================
    llm_attention_attn_vec_einsum = state_dict.pop(f"llm/layers/attn/attn_vec_einsum/w{suffix}")
    llm_attention_kv_einsum = state_dict.pop(f"llm/layers/attn/kv_einsum/w{suffix}")
    llm_attention_q_einsum = state_dict.pop(f"llm/layers/attn/q_einsum/w{suffix}")

    # MLP 层参数
    llm_mlp_gating_einsum = state_dict.pop(f"llm/layers/mlp/gating_einsum{suffix}")
    llm_mlp_linear = state_dict.pop(f"llm/layers/mlp/linear{suffix}")

    # RMS LayerNorm 参数（Gemma 使用 RMSNorm 而不是标准 LayerNorm）
    llm_input_layernorm = state_dict.pop(f"llm/layers/pre_attention_norm/scale{suffix}")
    llm_post_attention_layernorm = state_dict.pop(f"llm/layers/pre_ffw_norm/scale{suffix}")

    # ====================================================================
    # 9. 逐层解包 LLM 参数（18 层 Gemma 文本解码器）
    # ====================================================================
    # config.text_config.num_hidden_layers = 18（PaliGemma 中 Gemma 的层数）
    for i in range(config.text_config.num_hidden_layers):
        # ---- Q 投影 ----
        # JAX 中的 q_einsum 形状: [layers, num_heads, hidden, dim_per_head]
        # transpose(0, 2, 1): 变为 [layers, dim_per_head, num_heads, hidden]（在逐层循环中 layers 维度被 i 索引掉了）
        # 实际上对单个层: transpose(0, 2, 1) 后形状为 [num_heads, dim_per_head, hidden]
        # reshape 后: [num_heads * dim_per_head, hidden] -> 标准的 PyTorch Linear 权重形状
        # 最后 .transpose() -> PyTorch Linear 权重是 [out, in] = [num_heads * dim_per_head, hidden]
        q_proj_weight_reshaped = (
            llm_attention_q_einsum[i]
            .transpose(0, 2, 1)
            .reshape(
                config.text_config.num_attention_heads * config.text_config.head_dim, config.text_config.hidden_size
            )
        )
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.q_proj.weight"] = (
            q_proj_weight_reshaped
        )

        # ---- K 和 V 投影 ----
        # kv_einsum 形状: [layers, 2, 1, ...] 其中第 1 维的 0 是 K，1 是 V
        k_proj_weight_reshaped = llm_attention_kv_einsum[i, 0, 0].transpose()
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.k_proj.weight"] = (
            k_proj_weight_reshaped
        )
        v_proj_weight_reshaped = llm_attention_kv_einsum[i, 1, 0].transpose()
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.v_proj.weight"] = (
            v_proj_weight_reshaped
        )

        # ---- O 投影（注意力输出投影） ----
        o_proj_weight_reshaped = (
            llm_attention_attn_vec_einsum[i]
            .transpose(2, 0, 1)
            .reshape(
                config.text_config.num_attention_heads * config.text_config.head_dim, config.text_config.hidden_size
            )
        )
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.o_proj.weight"] = (
            o_proj_weight_reshaped
        )

        # ---- MLP 层 ----
        # Gemma 使用门控 MLP（Gated MLP），包含三个投影：
        # gate_proj: 门控投影（控制信息流）
        # up_proj: 升维投影（扩展特征维度）
        # down_proj: 降维投影（恢复原始维度）
        # gating_einsum[i, 0] 是 gate_proj, gating_einsum[i, 1] 是 up_proj
        gate_proj_weight = llm_mlp_gating_einsum[i, 0]
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.mlp.gate_proj.weight"] = (
            gate_proj_weight.transpose()
        )
        up_proj_weight = llm_mlp_gating_einsum[i, 1]
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.mlp.up_proj.weight"] = (
            up_proj_weight.transpose()
        )
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.mlp.down_proj.weight"] = (
            llm_mlp_linear[i].transpose()
        )

        # ---- LayerNorm（RMSNorm） ----
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.input_layernorm.weight"] = (
            llm_input_layernorm[i]
        )
        state_dict[
            f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.post_attention_layernorm.weight"
        ] = llm_post_attention_layernorm[i]

    # ---- 最终层归一化 ----
    jax_key = f"llm/final_norm/scale{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.language_model.norm.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    # ====================================================================
    # 10. 分离 PaliGemma 参数和 Action Expert 参数
    #
    # 在 π₀ 模型中，除了 PaliGemma 多模态模型外，还有一个独立训练的
    # "action expert"（动作专家）—— 它是另一个 Gemma 模型，专门用于
    # 处理机器人动作相关的 token。
    # 这些参数以 "_1" 结尾的键名区分（如 attn_vec_einsum_1, mlp_1 等），
    # 需要单独提取出来，在 slice_gemma_state_dict 中处理。
    # ====================================================================
    expert_dict = {}
    final_state_dict = {}

    # 列出所有属于 action expert 的键名
    # 这些键名都带有 "_1" 后缀，表示"第二个 Gemma"（编号 0 的 Gemma 已经作为 PaliGemma 的文本解码器了）
    expert_keys = [
        f"llm/final_norm_1/scale{suffix}",
        f"llm/final_norm_1/Dense_0/bias{suffix}",
        f"llm/final_norm_1/Dense_0/kernel{suffix}",
        f"llm/layers/attn/attn_vec_einsum_1/w{suffix}",
        f"llm/layers/attn/kv_einsum_1/w{suffix}",
        f"llm/layers/attn/q_einsum_1/w{suffix}",
        f"llm/layers/mlp_1/gating_einsum{suffix}",
        f"llm/layers/mlp_1/linear{suffix}",
        f"llm/layers/pre_attention_norm_1/scale{suffix}",
        f"llm/layers/pre_attention_norm_1/Dense_0/bias{suffix}",
        f"llm/layers/pre_attention_norm_1/Dense_0/kernel{suffix}",
        f"llm/layers/pre_ffw_norm_1/scale{suffix}",
        f"llm/layers/pre_ffw_norm_1/Dense_0/bias{suffix}",
        f"llm/layers/pre_ffw_norm_1/Dense_0/kernel{suffix}",
    ]

    # 遍历剩余的所有参数，普通参数（不是 expert 的）转换为 torch.Tensor 加入 final_state_dict
    # expert 参数则放入 expert_dict 等待下一步处理
    for key, value in state_dict.items():
        if key not in expert_keys:
            # 转换为 PyTorch 张量：先从 numpy.ndarray 转为 torch.Tensor
            final_state_dict[key] = torch.from_numpy(value)
        else:
            expert_dict[key] = value

    return final_state_dict, expert_dict


def slice_gemma_state_dict(state_dict, config, *, num_expert, checkpoint_dir, pi05):
    """
    将 Gemma（Action Expert）的 JAX 参数转换为 PyTorch 格式。

    这个函数处理的是 π₀ 模型中的"第二个 Gemma"（action expert），它专门负责
    处理机器人动作相关的 token 生成。在 π₀.₅ 中，这个 expert 使用了"自适应归一化"
    （adaptive normalization）技术，通过 Dense 层动态生成归一化参数。

    参数命名规则：
    - num_expert=1 表示"第二个 Gemma"（PaliGemma 内部的 Gemma 是第一个，编号 0）
    - 在 π₀.₅ 中，归一化层用 Dense_0 替代了简单的 scale

    Args:
        state_dict: 包含 expert 参数的字典（来自 slice_paligemma_state_dict 的 expert_dict 输出）
        config: Gemma 配置对象（如 gemma_300m 配置）
        num_expert: expert 编号（通常为 1）
        checkpoint_dir: 检查点路径（用于判断是 pi0 还是 pi05）
        pi05: 是否为 π₀.₅ 模型

    Returns:
        final_state_dict: 转换后的 PyTorch 参数字典（只包含 action expert 部分）
    """
    # 补全配置对象中可能缺少的属性
    # 因为传入的 config 是来自 openpi.models.gemma.get_config() 的原始配置，
    # 可能缺少 HuggingFace 风格配置所要求的字段
    if not hasattr(config, "vocab_size"):
        config.vocab_size = 257152  # PaliGemma 的词汇表大小（PALIGEMMA_VOCAB_SIZE）
    if not hasattr(config, "hidden_size"):
        config.hidden_size = config.width  # Gemma 配置中用 width 表示隐藏层大小
    if not hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = config.depth  # Gemma 配置中用 depth 表示层数
    if not hasattr(config, "num_attention_heads"):
        config.num_attention_heads = config.num_heads  # Gemma 配置中用 num_heads 表示注意力头数

    # 检查参数是否有 /value 后缀（同上，处理 FSDP 分片存储的情况）
    suffix = "/value" if f"llm/layers/attn/attn_vec_einsum_{num_expert}/w/value" in state_dict else ""

    # ====================================================================
    # 提取 Attention 和 MLP 的合并参数（与 slice_paligemma_state_dict 类似，
    # 但键名以 _{num_expert} 结尾区分不同的 expert）
    # ====================================================================
    llm_attention_attn_vec_einsum = state_dict.pop(f"llm/layers/attn/attn_vec_einsum_{num_expert}/w{suffix}")
    llm_attention_kv_einsum = state_dict.pop(f"llm/layers/attn/kv_einsum_{num_expert}/w{suffix}")
    llm_attention_q_einsum = state_dict.pop(f"llm/layers/attn/q_einsum_{num_expert}/w{suffix}")

    llm_mlp_gating_einsum = state_dict.pop(f"llm/layers/mlp_{num_expert}/gating_einsum{suffix}")
    llm_mlp_linear = state_dict.pop(f"llm/layers/mlp_{num_expert}/linear{suffix}")

    # ====================================================================
    # 处理归一化层参数的差异
    #
    # π₀（标准）: 使用 RMS LayerNorm，参数只有 scale（权重向量）
    # π₀.₅（升级版）: 使用"自适应归一化"（Adaptive RMSNorm），
    #   参数是两个 Dense 层（kernel + bias），可以动态生成归一化参数
    #   这种设计被称为"知识绝缘"（knowledge insulation），
    #   目的是让 action expert 的归一化参数不依赖于预训练的语言模型
    # ====================================================================
    if "pi05" in checkpoint_dir:
        # π₀.₅: 自适应归一化 —— 使用 Dense 层参数
        # pre_attention_norm_1/Dense_0 用于生成注意力前的归一化参数
        # pre_ffw_norm_1/Dense_0 用于生成 FFN 前的归一化参数
        llm_input_layernorm_bias = state_dict.pop(f"llm/layers/pre_attention_norm_{num_expert}/Dense_0/bias{suffix}")
        llm_post_attention_layernorm_bias = state_dict.pop(f"llm/layers/pre_ffw_norm_{num_expert}/Dense_0/bias{suffix}")
        llm_input_layernorm_kernel = state_dict.pop(
            f"llm/layers/pre_attention_norm_{num_expert}/Dense_0/kernel{suffix}"
        )
        llm_post_attention_layernorm_kernel = state_dict.pop(
            f"llm/layers/pre_ffw_norm_{num_expert}/Dense_0/kernel{suffix}"
        )
    else:
        # 标准 π₀: 普通的 RMSNorm
        llm_input_layernorm = state_dict.pop(f"llm/layers/pre_attention_norm_{num_expert}/scale{suffix}")
        llm_post_attention_layernorm = state_dict.pop(f"llm/layers/pre_ffw_norm_{num_expert}/scale{suffix}")

    # ====================================================================
    # 逐层解包参数
    # ====================================================================
    # config.num_hidden_layers: action expert 的 Gemma 层数（如 300M 参数的 Gemma 是 12 层）
    for i in range(config.num_hidden_layers):
        # ---- Q 投影 ----
        q_proj_weight_reshaped = (
            llm_attention_q_einsum[i]
            .transpose(0, 2, 1)
            .reshape(config.num_attention_heads * config.head_dim, config.hidden_size)
        )
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.q_proj.weight"] = (
            q_proj_weight_reshaped
        )

        # ---- K 和 V 投影 ----
        k_proj_weight_reshaped = llm_attention_kv_einsum[i, 0, 0].transpose()
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.k_proj.weight"] = (
            k_proj_weight_reshaped
        )
        v_proj_weight_reshaped = llm_attention_kv_einsum[i, 1, 0].transpose()
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.v_proj.weight"] = (
            v_proj_weight_reshaped
        )

        # ---- O 投影 ----
        # 注意：这里与 PaliGemma 的 O 投影 reshape 方式不同
        # 直接 reshape + transpose(1, 0) 而不是 transpose(2, 0, 1) + reshape
        # 这是因为 expert 的 einsum 权重布局可能与 PaliGemma 的不同
        o_proj_weight_reshaped = (
            llm_attention_attn_vec_einsum[i]
            .reshape(config.num_attention_heads * config.head_dim, config.hidden_size)
            .transpose(1, 0)
        )
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.o_proj.weight"] = (
            o_proj_weight_reshaped
        )

        # ---- 门控 MLP ----
        gate_proj_weight = llm_mlp_gating_einsum[i, 0]
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.mlp.gate_proj.weight"] = (
            gate_proj_weight.transpose()
        )
        up_proj_weight = llm_mlp_gating_einsum[i, 1]
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.mlp.up_proj.weight"] = (
            up_proj_weight.transpose()
        )
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.mlp.down_proj.weight"] = llm_mlp_linear[
            i
        ].transpose()

        # ---- 归一化层（根据 π₀ 或 π₀.₅ 不同处理方式） ----
        if "pi05" in checkpoint_dir:
            # π₀.₅：自适应归一化的 Dense 层参数
            # 输入层归一化的 Dense 层（生成注意力前的 RMSNorm 参数）
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.input_layernorm.dense.bias"] = (
                llm_input_layernorm_bias[i]
            )
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.post_attention_layernorm.dense.bias"] = (
                llm_post_attention_layernorm_bias[i]
            )
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.input_layernorm.dense.weight"] = (
                llm_input_layernorm_kernel[i].transpose()
            )
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.post_attention_layernorm.dense.weight"] = (
                llm_post_attention_layernorm_kernel[i].transpose()
            )
        else:
            # 标准 π₀：普通的 RMSNorm weight
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.input_layernorm.weight"] = (
                llm_input_layernorm[i]
            )
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.post_attention_layernorm.weight"] = (
                llm_post_attention_layernorm[i]
            )

    # ====================================================================
    # 处理最终的归一化层（所有层之后的 LayerNorm）
    # ====================================================================
    if "pi05" in checkpoint_dir:
        # π₀.₅：使用 Dense 层
        final_norm_bias = state_dict.pop(f"llm/final_norm_{num_expert}/Dense_0/bias{suffix}")
        final_norm_kernel = state_dict.pop(f"llm/final_norm_{num_expert}/Dense_0/kernel{suffix}")
        state_dict["paligemma_with_expert.gemma_expert.model.norm.dense.bias"] = final_norm_bias
        state_dict["paligemma_with_expert.gemma_expert.model.norm.dense.weight"] = final_norm_kernel.transpose()
    else:
        # 标准 π₀：使用 scale 向量
        state_dict["paligemma_with_expert.gemma_expert.model.norm.weight"] = state_dict.pop(
            f"llm/final_norm_{num_expert}/scale{suffix}"
        )

    # 在 Gemma 中，lm_head（语言模型头，即最终输出层）的权重与 token 嵌入层的权重是"绑定的"
    # （weight tying），所以不需要单独加载 lm_head 权重
    # state_dict["paligemma_with_expert.gemma_expert.lm_head.weight"] = embedding_vector

    # ====================================================================
    # 将剩余的 numpy 数组转换为 torch.Tensor
    # ====================================================================
    final_state_dict = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            final_state_dict[key] = torch.from_numpy(value)
        else:
            final_state_dict[key] = value

    return final_state_dict


def slice_initial_orbax_checkpoint(checkpoint_dir: str, restore_precision: str | None = None):
    """
    加载并处理初始的 orbax 检查点。

    这个方法通过 JAX 模型加载器（使用 openpi 的 restore_params 函数）来恢复检查点，
    这样可以正确处理 dtype 转换（如 float32 → bfloat16 等）。

    Args:
        checkpoint_dir: 检查点目录路径
        restore_precision: 恢复精度，如 "float32"、"bfloat16" 或 None（使用原始精度）

    Returns:
        包含两个键的字典：
        - "paligemma_params": 展平后的 PaliGemma 参数（使用 traversals.flatten_mapping）
        - "projection_params": 投影层参数（state_proj, action_proj, time_mlp 等）
    """
    # 使用仓库提供的 restore 工具加载检查点，返回纯参数字典（已移除 /value 后缀）
    # restore_params 会从 checkpoint_dir/params/ 目录加载参数
    params = openpi.models.model.restore_params(
        f"{checkpoint_dir}/params/", restore_type=np.ndarray, dtype=restore_precision
    )

    # traversals.flatten_mapping 将嵌套的字典展平为单层字典
    # 例如：{"PaliGemma": {"img": {"embedding": ...}}} → {"img/embedding": ...}
    # 这样便于后续通过字符串键名直接访问各个参数
    return {"paligemma_params": traversals.flatten_mapping(params["PaliGemma"], sep="/"), "projection_params": params}


def load_jax_model_and_print_keys(checkpoint_dir: str):
    """
    加载 JAX 模型检查点并打印所有参数键名和结构。

    这个函数适用于只想查看模型结构而不做转换的场景。
    使用 orbax 的 PyTreeCheckpointer 来读取元数据。

    Args:
        checkpoint_dir: 检查点目录的本地路径或 GCS 路径（gs:// 开头）
    """
    # 如果是本地路径，先转为绝对路径；GCS 路径保持不变
    checkpoint_dir = os.path.abspath(checkpoint_dir) if not checkpoint_dir.startswith("gs://") else checkpoint_dir

    # 创建 orbax 的 PyTreeCheckpointer 实例
    # PyTreeCheckpointer 是 orbax 中处理任意嵌套结构（PyTree）的检查点工具
    checkpointer = ocp.PyTreeCheckpointer()

    # 读取检查点的元数据（参数结构信息），但不加载实际权重数据
    metadata = checkpointer.metadata(f"{checkpoint_dir}/params")

    # 使用 openpi 的 array_tree_to_info 将元数据格式化为可读的树状结构并打印
    print(utils.array_tree_to_info(metadata))


def convert_pi0_checkpoint(
    checkpoint_dir: str, precision: str, output_path: str, model_config: openpi.models.pi0_config.Pi0Config
):
    """
    将 π₀ JAX 检查点转换为 PyTorch 格式的主函数。

    转换流程：
    1. 使用 orbax + JAX 加载原始检查点（按 float32 精度恢复，避免精度损失）
    2. 提取投影层参数（state_proj、action_proj、time_mlp 等）
    3. 创建 PaliGemma 虚拟配置用于键名映射
    4. 通过 slice_paligemma_state_dict 转换视觉部分的参数
    5. 通过 slice_gemma_state_dict 转换 action expert 部分的参数
    6. 合并所有参数并加载到 PI0Pytorch 模型中
    7. 转换为目标精度并保存为 SafeTensors 格式

    Args:
        checkpoint_dir: JAX 检查点路径
        precision: 目标精度（float32、bfloat16、float16）
        output_path: 转换后模型的保存路径
        model_config: π₀ 模型配置（Pi0Config）
    """
    print(f"Converting PI0 checkpoint from {checkpoint_dir} to {output_path}")
    print(f"Model config: {model_config}")

    # ====================================================================
    # Step 1: 加载 JAX 检查点
    # ====================================================================
    # 通过 JAX 先恢复为 float32，确保精度不损失
    # 后续再根据用户指定的 precision 进行转换
    initial_params = slice_initial_orbax_checkpoint(checkpoint_dir=checkpoint_dir, restore_precision="float32")

    # ====================================================================
    # Step 2: 处理投影层参数
    #
    # 投影层是 π₀ 模型中一些独立的线性层（Linear），用于处理：
    # - state_proj: 状态投影（将机器人状态映射到模型维度）
    # - action_in_proj: 动作输入投影（将动作数据映射到模型维度）
    # - action_out_proj: 动作输出投影（将模型输出映射到动作空间）
    # - time_mlp_in/out: 时间 MLP（流匹配模型中处理时间步长 t 的 MLP）
    #
    # π₀.₅ 和 π₀ 的投影层命名略有不同：
    # - π₀: state_proj, action_in_proj, action_out_proj, action_time_mlp_in, action_time_mlp_out
    # - π₀.₅: action_in_proj, action_out_proj, time_mlp_in, time_mlp_out
    # ====================================================================
    if model_config.pi05:
        # π₀.₅ 没有 state_proj（状态直接通过 tokenize 处理）
        # 也没有 action_ 前缀
        keys = [
            "action_in_proj",
            "action_out_proj",
            "time_mlp_in",
            "time_mlp_out",
        ]
    else:
        keys = [
            "state_proj",
            "action_in_proj",
            "action_out_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
        ]

    projection_params = {}
    for key in keys:
        kernel_params = initial_params["projection_params"][key]["kernel"]
        bias_params = initial_params["projection_params"][key]["bias"]

        # 检查是否有 /value 子键（处理 FSDP 分片）
        if isinstance(kernel_params, dict):
            weight = kernel_params["value"]
            bias = bias_params["value"]
        else:
            weight = kernel_params
            bias = bias_params

        # JAX Linear 层的 kernel 形状是 [out_features, in_features]
        # PyTorch Linear 层的 weight 形状也是 [out_features, in_features]
        # 但 JAX Flax 的 Dense 层 kernel 布局可能与 PyTorch 的 Linear 有转置关系
        # 这里做了 .T（转置）以匹配 PyTorch 的约定
        pytorch_weight_key = f"{key}.weight"
        pytorch_bias_key = f"{key}.bias"

        projection_params[pytorch_weight_key] = torch.from_numpy(np.array(weight)).T
        projection_params[pytorch_bias_key] = torch.from_numpy(np.array(bias))

    # ====================================================================
    # Step 3: 创建模型配置
    #
    # 为键名映射创建 PaliGemma 配置对象。
    # 这里使用 type("obj", (object,), {...}) 动态创建类实例，
    # 相当于创建一个简单的配置对象，而无需定义完整的配置类。
    # ====================================================================
    class PaliGemmaConfig:
        """PaliGemma 模型的配置，只包含键名映射所需的字段。"""
        def __init__(self):
            # 视觉编码器配置（SigLIP ViT）
            self.vision_config = type(
                "obj",
                (object,),
                {
                    "hidden_size": 1152,          # 隐藏层维度
                    "num_hidden_layers": 27,       # ViT 编码器层数（PaliGemma 使用 27 层）
                    "num_attention_heads": 16,     # 注意力头数
                    "intermediate_size": 4304,     # MLP 中间层维度
                    "patch_size": 14,              # 图像块大小（14x14 像素）
                    "projection_dim": 2048,        # 投影维度（视觉到语言的对齐维度）
                },
            )()
            # 文本解码器配置（Gemma）
            self.text_config = type(
                "obj",
                (object,),
                {
                    "hidden_size": 2048,           # 隐藏层维度
                    "num_hidden_layers": 18,       # Gemma 解码器层数
                    "num_attention_heads": 8,      # 注意力头数
                    "head_dim": 256,               # 每个注意力头的维度
                    "intermediate_size": 16384,    # MLP 中间层维度
                },
            )()

    paligemma_config = PaliGemmaConfig()

    # 获取 action expert 的 Gemma 配置（使用 3 亿参数的 Gemma）
    action_expert_config = openpi.models.gemma.get_config("gemma_300m")

    # ====================================================================
    # Step 4: 处理 PaliGemma 参数（视觉 + 语言模型主体）
    # ====================================================================
    paligemma_params, expert_params = slice_paligemma_state_dict(initial_params["paligemma_params"], paligemma_config)

    # ====================================================================
    # Step 5: 处理 Action Expert 参数（专门处理机器人动作的第二个 Gemma）
    # ====================================================================
    gemma_params = slice_gemma_state_dict(
        expert_params, action_expert_config, num_expert=1, checkpoint_dir=checkpoint_dir, pi05=model_config.pi05
    )

    # ====================================================================
    # Step 6: 实例化 PyTorch 模型并加载参数
    # ====================================================================
    # 创建 PI0Pytorch 模型实例（这是 π₀ 的 PyTorch 实现）
    pi0_model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_config)

    # 合并所有部分（PaliGemma主体 + Action Expert + 投影层）
    all_params = {**paligemma_params, **gemma_params, **projection_params}

    # 加载状态字典到模型中
    # strict=False 表示允许部分参数不匹配（某些参数如缓存、掩码等不会被加载）
    pi0_model.load_state_dict(all_params, strict=False)

    # ====================================================================
    # Step 7: 设置精度并保存
    # ====================================================================
    if precision == "float32":
        pi0_model = pi0_model.to(torch.float32)
    elif precision == "bfloat16":
        pi0_model = pi0_model.to(torch.bfloat16)
    else:
        raise ValueError(f"Invalid precision: {precision}")

    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)

    # 使用 SafeTensors 格式保存模型权重
    # safetensors 是一种安全的序列化格式，不会像 pickle 那样存在执行任意代码的风险
    # save_model 函数会自动处理权重绑定（weight tying）
    safetensors.torch.save_model(pi0_model, os.path.join(output_path, "model.safetensors"))

    # ====================================================================
    # Step 8: 复制额外的资产文件（assets）
    #
    # assets 目录通常包含归一化统计信息（norm_stats），
    # 用于对输入数据（状态、动作等）进行归一化/反归一化。
    # 这些文件在推理时是必需的。
    #
    # 检查点结构：
    #   checkpoints/pi0_droid/
    #   ├── 20000/              # 训练步数命名的子目录
    #   │   ├── params/         # 模型参数（由 orbax 管理）
    #   │   └── train_state/    # 训练状态（优化器等）
    #   └── assets/             # 共享资产（归一化统计信息等）
    # ====================================================================
    assets_source = pathlib.Path(checkpoint_dir).parent / "assets"
    if assets_source.exists():
        assets_dest = pathlib.Path(output_path) / "assets"
        if assets_dest.exists():
            shutil.rmtree(assets_dest)
        shutil.copytree(assets_source, assets_dest)

    # 保存模型配置信息为 JSON 供参考
    config_dict = {
        "action_dim": model_config.action_dim,                   # 动作维度（关节数）
        "action_horizon": model_config.action_horizon,           # 动作预测的时序范围
        "paligemma_variant": model_config.paligemma_variant,     # PaliGemma 变体（如 gemma_2b）
        "action_expert_variant": model_config.action_expert_variant,  # Action Expert 变体
        "precision": precision,                                  # 保存精度
    }
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    print("Model conversion completed successfully!")
    print(f"Model saved to {output_path}")


def main(
    checkpoint_dir: str,
    config_name: str,
    output_path: str | None = None,
    precision: Literal["float32", "bfloat16", "float16"] = "bfloat16",
    *,
    inspect_only: bool = False,
):
    """
    主函数：加载 JAX 模型并可选择性地转换为 PyTorch。

    Args:
        checkpoint_dir: JAX 检查点目录路径
        config_name: 训练配置名称（在 config.py 的 _CONFIGS 中注册的配置名，如 "pi0_libero"）
        output_path: 转换后的 PyTorch 模型保存路径（仅在转换时需要）
        precision: 模型精度（默认为 bfloat16，在保持良好精度的同时节省存储空间）
        inspect_only: 仅检查参数键名，不进行转换
    """
    # 根据配置名称获取对应的训练配置，并提取模型配置部分
    model_config = _config.get_config(config_name).model

    # 确保模型配置是 π₀ 类型的配置（Pi0Config），如果不是则报错
    # 因为本转换脚本只支持 π₀ 和 π₀.₅ 架构
    if not isinstance(model_config, openpi.models.pi0_config.Pi0Config):
        raise ValueError(f"Config {config_name} is not a Pi0Config")

    if inspect_only:
        # 仅检查模式：打印参数键名结构
        load_jax_model_and_print_keys(checkpoint_dir)
    else:
        # 转换模式：需要提供输出路径
        if not output_path:
            print("Error: --output_path is required for conversion. Use --inspect_only to only view keys.")
            return
        convert_pi0_checkpoint(checkpoint_dir, precision, output_path, model_config)


if __name__ == "__main__":
    # 使用 tyro 库自动生成命令行参数解析器
    # tyro 可以根据函数签名（类型注解）自动推导参数类型、默认值和帮助信息
    # 用户可以通过命令行的 --checkpoint_dir, --config_name 等参数来指定
    tyro.cli(main)

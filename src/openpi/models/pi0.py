"""
===================================================================
  π₀（Pi-Zero）模型 —— 流匹配（Flow Matching）架构的核心实现
===================================================================

什么是 π₀ 模型？
  π₀ 是 Physical Intelligence 开发的视觉-语言-动作（VLA）基础模型。
  它采用"流匹配（Flow Matching）"作为核心的生成范式。

三个模型架构的版本：
  ┌──────────────┬────────────────────────────────────────────┐
  │  π₀          │ 原始流匹配模型，使用 state_proj 编码状态    │
  │  π₀.₅ (pi05) │ 升级版：知识隔离（adaRMS）+ 时间 MLP       │
  │  π₀-FAST     │ 自回归架构（在 pi0_fast.py 中实现）        │
  └──────────────┴────────────────────────────────────────────┘

流匹配（Flow Matching）的核心思想：
  流匹配是一种生成式建模方法，它学习一个"速度场"（velocity field），
  将噪声分布平滑地"流动"到目标数据分布。

  训练时：
    1. 取一个真实动作 a₀
    2. 取一个随机噪声 ε ~ N(0, I)
    3. 随机采样一个时间 t ∈ (0, 1)
    4. 构造插值：x_t = t·ε + (1-t)·a₀（带噪声的概率路径）
    5. 计算真实速度：u_t = ε - a₀
    6. 让模型预测速度 v_t ≈ u_t
    7. 损失函数：L = MSE(v_t, u_t)

  推理时（采样）：
    1. 从 x₁ = ε（纯噪声）出发
    2. 沿 v_t 的负方向逐步去噪（欧拉采样）
    3. 经过 N 步得到 x₀ ≈ 真实动作

  这种方法的优势：
    - 比扩散模型（Diffusion）更灵活（不局限于特定噪声调度）
    - 比自回归（Autoregressive）更快（可以并行生成动作块）
    - 数值稳定，训练简单

模型架构概览：
  ┌──────────────────────────────────────────────────────────────────┐
  │  Pi0 模型                                                        │
  │                                                                  │
  │  ┌──────────────────────────────────┐                            │
  │  │  PaliGemma（语言+视觉编码器）    │                            │
  │  │  ├── img (SigLIP 图像编码器)    │   ← 将图像编码为 token    │
  │  │  └── llm (Gemma 语言模型)       │   ← 处理语言 + 视觉 token │
  │  └──────────────────────────────────┘                            │
  │                                                                  │
  │  ┌──────────────────────────────────┐                            │
  │  │  动作专家模块（Action Expert）   │                            │
  │  │  ├── action_in_proj: 投影噪声动作│                           │
  │  │  ├── time_mlp: 时间嵌入 MLP     │   ← π₀.₅ 用 adaRMS       │
  │  │  ├── state_proj: 投影状态 token │   ← π₀ 用                │
  │  │  └── action_out_proj: 预测速度   │                           │
  │  └──────────────────────────────────┘                            │
  └──────────────────────────────────────────────────────────────────┘

  工作流程：
    前缀（Prefix）：图像 → SigLIP → 图像 token
                   文本 → Gemma 嵌入 → 文本 token
                   合并 → 前缀 token 序列

    后缀（Suffix）：状态 token + 动作 token → 拼接 → 动作专家处理
                   前缀和后缀一起通过 Gemma LLM 处理
                   最后从动作部分提取速度预测 v_t
"""

import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """构造注意力掩码（Attention Mask），控制 token 之间的可见性。

    这段代码改编自 big_vision（Google 的视觉模型库）。

    注意力掩码的作用：
      在 Transformer 的自注意力（Self-Attention）中，每个 token 可以"看到"
      其他 token。注意力掩码决定了哪些 token 对之间是可见的。

    两种掩码的组合：
      1. input_mask: 哪些 token 是真实输入（True）vs 填充（False，padding）
      2. mask_ar:   自回归掩码（Autoregressive mask），指示哪些位置是"块的边界"

    mask_ar 的工作方式：
      mask_ar 中为 True 的位置表示"这是一个新块的开始，前面的 token 不能看到这个
      块内的 token"。通过累积和（cumsum）可以计算出每个 token 所属的"块索引"。

    示例 —— 各种注意力模式：
      mask_ar = [1, 1, 1, 1, 1] → 纯因果注意力（Causal Attention）
        每个 token 只能看到自己和前面的 token。

      mask_ar = [0, 0, 0, 1, 1, 1] → Prefix-LM 注意力
        前 3 个 token（前缀）可以互相看到（双向注意力），
        后 3 个 token（后缀）只能看到前面的 token（因果注意力）。

      mask_ar = [1, 0, 1, 0, 1, 0] → 块因果注意力
        形成 3 个块，每个块内可以互相看到，但块与块之间是因果的。

    在 π₀ 中的具体应用：
      前缀（Prefix）：图像 token + 文本 token
        mask_ar 全为 False（0）
        → 前缀内的所有 token 可以互相看到（全注意力）

      后缀（Suffix）：状态 token + 动作 token
        第一个 token 的 mask_ar = True（1），其余为 False（0）
        state(1 0 0 0 ...) token(1 0 0 0 0 0 ...)
        → 每个时间步只能看到当前和之前的 token
        → 动作 token 不能"偷看"未来的动作

    Args:
      input_mask: 形状 [B, N] 的布尔张量。True 表示有效输入，False 表示填充。
      mask_ar: 形状 [B, N] 的布尔张量。True 表示该位置是"块边界"（前面的
               token 不能依赖它）。False 表示与前面的 token 共享注意力掩码。

    Returns:
      attn_mask: 形状 [B, N, N] 的布尔注意力掩码。
    """
    # 将 mask_ar 广播到 input_mask 的形状（如果维度不匹配）
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)

    # 计算累积和：每个 token 被分配一个"块索引"
    # 例如 mask_ar = [0, 0, 1, 0, 0] → cumsum = [0, 0, 1, 1, 1]
    cumsum = jnp.cumsum(mask_ar, axis=1)

    # 核心规则：token i 能 attend to token j 当且仅当
    # token i 的块索引 <= token j 的块索引
    # 这意味着：同一个块内的 token 可以互相看到（双向），
    # 但后面的块不能看到前面块内的 token（因果）
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]

    # 填充 token（input_mask=False）不能 attend 到任何其他 token，
    # 其他 token 也不能 attend 到填充 token
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]

    """
代码中的 None 是干什么的？
在 NumPy / JAX 中，None（等同于 np.newaxis）用于在指定位置插入一个新维度，目的是触发广播机制（Broadcasting），从而生成二维的注意力矩阵。
cumsum[:, None, :] 的形状为 (Batch, 1, Seq)（行视角）。
cumsum[:, :, None] 的形状为 (Batch, Seq, 1)（列视角）。
两者广播后得到 (Batch, Seq, Seq) 的矩阵，其中 [i, j] 位置的元素就代表了 第 i 个 token（行）能否看到第 j 个 token（列）。
    """

    # 最终掩码 = 注意力规则 AND 有效输入
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """计算标量位置的正弦-余弦位置编码（Sinusoidal Positional Embedding）。

    位置编码的作用：
      在 Transformer 模型中，自注意力机制本身是"置换不变"（permutation invariant）的，
      即它不知道 token 的顺序。位置编码就是用来告诉模型"token 之间的相对位置关系"。

    这里为什么用正弦-余弦编码？
      这是 Transformer 论文（Vaswani et al., 2017）中提出的经典方法。
      它用不同频率的正弦和余弦波来编码位置信息。
      优点是：
        1. 不需要学习参数
        2. 可以外推到训练时未见过的位置
        3. 不同频率的组合提供了多尺度的位置信息

    在 π₀ 中的用途：
      编码"扩散时间步"（timestep）t ∈ [0, 1]。
      时间步 t 告诉模型当前处于去噪过程的哪个阶段，
      不同阶段需要不同的"去噪力度"。

    计算细节：
      1. 将 embedding_dim 分成两半，分别用 sin 和 cos
      2. 在 [min_period, max_period] 范围内均匀采样频率（对数空间）
      3. 对每个位置 pos，用所有频率计算 sin(pos * freq) 和 cos(pos * freq)

    Args:
        pos: 位置值（标量或向量），形状 [b]（b 为批大小）。
        embedding_dim: 嵌入维度，必须是偶数。
        min_period: 最小周期（控制最高频率）。
        max_period: 最大周期（控制最低频率）。

    Returns:
        形状为 [b, embedding_dim] 的正弦-余弦位置编码。
    """
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    # 在 [0, 1] 范围内均匀采样 embedding_dim/2 个点
    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)

    # 在对数尺度上从 min_period 到 max_period 生成周期序列
    # 例如 min_period=0.004, max_period=4.0:
    #   fraction=0 → period=0.004（最高频）
    #   fraction=1 → period=4.0（最低频）
    period = min_period * (max_period / min_period) ** fraction

    # 计算输入：pos * (2π / period)
    # 这就是 sin/cos 的输入角度（以弧度为单位）
    sinusoid_input = jnp.einsum(
        "i,j->ij",  # 外积：每个位置 × 每个频率
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,  # 最高精度
    )

    # 拼接 sin 和 cos 输出，得到 [b, embedding_dim]
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    """π₀（Pi-Zero）模型：流匹配架构的 VLA（视觉-语言-动作）模型。

    这是 openpi 中最核心的模型类。它继承自 BaseModel，实现了：
      - compute_loss(): 训练阶段使用的流匹配损失计算
      - sample_actions(): 推理阶段使用的去噪采样

    模型的两大组件：
      1. PaliGemma（视觉-语言部分）：
         - SigLIP 图像编码器：将图像编码为视觉 token
         - Gemma 语言模型：理解语言指令，融合视觉和语言信息
      2. 动作专家（Action Expert）：
         - 处理机器人状态和动作
         - 预测流匹配的速度场 v_t

    π₀ 与 π₀.₅ 的区别：
      - π₀.₅ 使用了"知识隔离"（Knowledge Isolation）技术：
        通过 adaRMS（Adaptive RMS Normalization）将时间步信息
        注入到模型的归一化层中，而不是通过简单的拼接。
      - π₀.₅ 没有 state_proj（状态投影层），状态通过离散 token 输入
    """

    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        """初始化 π₀ 模型。

        初始化流程：
          1. 调用父类 BaseModel 的 __init__，设置 action_dim、action_horizon、max_token_len
          2. 创建 PaliGemma 视觉-语言模型（使用 nnx_bridge 包装 Flax 模块）
          3. 创建图像编码器 SigLIP
          4. 创建动作专家模块的各种线性投影层
          5. 设置 deterministic = True（默认在评估模式）

        Args:
            config: π₀ 模型配置（包含模型变体、维度、超参数等）。
            rngs: 随机数生成器状态（NNX 框架用于参数初始化）。
        """
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05  # 是否使用 π₀.₅ 变体

        # 获取 PaliGemma 和动作专家的配置
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        # ---- 创建 PaliGemma（视觉-语言模型） ----
        # Gemma 目前还是 Flax Linen 模块（非 NNX），所以用 nnx_bridge.ToNNX 包装
        # configs=[paligemma_config, action_expert_config] 表示两个 Gemma 模块共享
        #   同一个 Transformer 的大部分层，但有不同的"专家头"
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,  # 嵌入层的数据类型
                adarms=config.pi05,  # π₀.₅ 使用 adaRMS 归一化
            )
        )
        # 延迟初始化：使用 dummy 输入来确定参数形状
        # use_adarms=[False, True]：
        #   第一个 Gemma（PaliGemma）不使用 adaRMS
        #   第二个 Gemma（动作专家）使用 adaRMS（仅 π₀.₅）
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])

        # ---- 创建 SigLIP 图像编码器 ----
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,  # 输出维度与 Gemma 对齐
                variant="So400m/14",  # SigLIP 变体（400M 参数，patch size 14）
                pool_type="none",  # 不使用池化（保留所有 patch token）
                scan=True,  # 使用 scan 优化减少显存
                dtype_mm=config.dtype,  # 数据类型
            )
        )
        # 用一张假图像初始化 SigLIP，确定其参数形状
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)

        # 将 llm 和 img 组成 PaliGemma 字典
        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        # ---- 创建动作专家模块 ----
        # 动作输入投影：将噪声动作映射到动作专家嵌入空间
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)

        if config.pi05:
            # π₀.₅ 的时间 MLP：将时间嵌入通过两个 MLP 层处理后作为 adaRMS 的条件
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            # π₀ 的状态编码器：将机器人状态映射到嵌入空间
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            # π₀ 的时间-动作融合 MLP：拼接时间嵌入和动作嵌入后融合
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)

        # 动作输出投影：从嵌入空间预测速度场 v_t
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # 这个属性会被 model.train() 和 model.eval() 自动设置。
        # True = 推理模式（确定性的，不使用 Dropout 等随机操作）
        # False = 训练模式（使用 Dropout 等正则化）
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        """编码前缀（Prefix）：将观测数据中的图像和文本编码为 token 序列。

        什么是"前缀"？
          前缀是模型中所有输入 token 的前半部分，包括：
            - 图像 token：来自 SigLIP 编码器，描述当前场景
            - 文本 token：来自 Gemma 嵌入层，描述语言指令

          前缀中的 token 之间使用"全注意力"（Full Attention），
          即它们可以互相"看到"彼此。这允许模型建立图像和文本之间的
          跨模态关联。

        输出：
           - tokens:      [B, N_prefix, emb]  编码后的 token 序列
           - input_mask:  [B, N_prefix]        哪些 token 是有效的
           - ar_mask:     [N_prefix]          自回归掩码（全部为 False，全注意力）

        Args:
            obs: 观测数据，包含多张图像和分词后的文本提示。

        Returns:
            包含 (tokens, input_mask, ar_mask) 的三元组。
        """

        """ 这里处理的是一条数据还是整个batch？
不是的——`embed_prefix` **直接处理整个 batch**，不是一条一条处理的。

## 从数据流看

追踪一下数据到 `embed_prefix` 的完整路径：

```
1. TransformedDataset.__getitem__(index)
   → 返回：一条数据，没有 batch 维度                    ← 单个样本
        state: (8,),  image: (224,224,3), ...

2. _collate_fn(items)                                ← 多条样本堆叠成 batch
   → np.stack([样本0, 样本1, 样本2, ...], axis=0)
   → 返回：state: (B, 8),  image: (B, 224, 224, 3), ...

3. Observation.from_dict(batch)                      ← 转换为 Observation 对象
   → 此时 Observation 里所有字段都已经有 B 维度了

4. model.compute_loss(rng, observation, actions)     ← 传入的是整个 batch
   → embed_prefix(observation)
```

所以进入 `embed_prefix` 时，`obs.images["base_0_rgb"]` 的形状已经是 `(B, 224, 224, 3)`。

## embed_prefix 内部是向量化处理的

```python
# embed_prefix 中的操作都是 batch 维度的：
for name in obs.images:
    image_tokens = self.PaliGemma.img(obs.images[name], train=False)
    #                               ↑ 形状 (B, H, W, 3)，直接传给 SigLIP
    #                输出形状：[B, num_patches, embedding_dim]
```

SigLIP 接收的形状是 `(B, H, W, 3)`，它**在 batch 维度上并行**编码所有图像。这不是循环调用 B 次，而是一次 `img()` 调用同时处理 B 张图像。

所有后续操作同样保留 batch 维度：

```python
# concat along sequence dim, batch dim stays
tokens = jnp.concatenate(tokens, axis=1)      # [B, total_prefix_tokens, emb]
input_mask = jnp.concatenate(input_mask, axis=1)  # [B, total_prefix_tokens]

# ar_mask 是 1D 的，因为它对所有 batch 样本都一样
ar_mask = jnp.array(ar_mask)                   # [total_prefix_tokens]
```

## 为什么 ar_mask 是 1D 的？

注意 `ar_mask` 的输出类型标注是 `" s"`（没有 `b`）：

```python
def embed_prefix(self, obs: Observation) -> tuple[
    at.Float[at.Array, "b s emb"],     # tokens:    有 batch 维度
    at.Bool[at.Array, "b s"],          # input_mask: 有 batch 维度
    at.Bool[at.Array, " s"],           # ar_mask:    没有 batch 维度！
]:
```

这是因为 `ar_mask` 只取决于 **token 的顺序结构**，而这个结构对所有 batch 样本是相同的——每个样本都有相同数量的图像 token 和文本 token。
所以它不需要 batch 维度，调用时会被广播（`make_attn_mask` 中通过 `jnp.broadcast_to` 展开）。

## 所以结论是

`embed_prefix` **不是对一条数据处理的，而是直接对整个 batch 做向量化处理**。batch 维度在所有操作中始终保留，JAX 负责在底层利用 GPU 并行加速这些 batch 运算。

这与 PyTorch 的常规做法一致——**模型层面永远以 batch 为单位工作**，单条数据和 batch 的区别在于最前面是否多了一个 `B` 维度。
        """

        input_mask = []
        ar_mask = []
        tokens = []

        # ---- 编码图像 ----
        # 遍历所有摄像头图像（如"base_0_rgb", "left_wrist_0_rgb"）
        for name in obs.images:
            # 通过 SigLIP 图像编码器提取视觉特征
            # 输出形状：[B, num_patches, embedding_dim]
            # train=False：不使用图像编码器中的随机增强
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)

            # 创建图像 token 的 input_mask：
            # 如果该图像是真实存在的（image_mask=True），所有 token 都有效；
            # 如果是填充图像（image_mask=False），所有 token 都无效
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],  # 当前图像产生的 token 数量
                )
            )

            # 图像 token 之间可以互相 attend（全注意力，非因果）
            # ar_mask=False 表示"不是自回归边界"
            ar_mask += [False] * image_tokens.shape[1]

        # ---- 编码文本（语言指令）----
        if obs.tokenized_prompt is not None:
            # 通过 Gemma 的嵌入层将 token ID 转换为嵌入向量
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)

            # 文本 token 的掩码
            input_mask.append(obs.tokenized_prompt_mask)

            # 文本 token 也可以互相 attend（全注意力）
            ar_mask += [False] * tokenized_inputs.shape[1]

        # ---- 拼接所有前缀 token ----
        # 将所有 token 在序列维度上拼接
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        # ar_mask 在批次维度上广播，所以用一维数组
        ar_mask = jnp.array(ar_mask)

        return tokens, input_mask, ar_mask

    """
这个问题切中要害。答案是：**`input_mask` 实际上也可以是一维的，但设计上选择保留 batch 维度是因为不同样本的掩码确实可能不同。**

## 什么时候它们相同？

最简单的场景——所有样本都有完整的 3 个摄像头、相同的 prompt 长度：

```
ar_mask (1D):     [F, F, F, F, F, F]  ← 6个 prefix token，全 False
input_mask (2D):  [[T, T, T, T, T, T],   ← 样本0，全部有效
                   [T, T, T, T, T, T],   ← 样本1，全部有效
                   [T, T, T, T, T, T]]   ← 样本2，全部有效
                    ↑ 完全可以压缩成 1D ↑
```

这种场景下 `input_mask` 确实是多余的 B 维。

## 什么时候它们不同？

看 `embed_prefix` 中的构造代码：

```python
for name in obs.images:
    input_mask.append(
        einops.repeat(
            obs.image_masks[name],         # ← 形状 [B]，不同样本可能不同！
            "b -> b s",
            s=image_tokens.shape[1],
        )
    )
```

`obs.image_masks["right_wrist_0_rgb"]` 的形状是 `[B]`。假设 batch=3：

```
right_wrist_0_rgb mask: [True, False, True]
                          ↑      ↑      ↑
                        样本0   样本1   样本2
```

- 样本 0 和 2 有右腕摄像头 → mask=True
- 样本 1 的右腕摄像头缺失（或填充）→ mask=False

展开成 token 级别的 mask（假设图像产 256 个 token）：

```
input_mask (right wrist):  [[T, T, ..., T],    ← 样本0，256个有效token
                            [F, F, ..., F],    ← 样本1，256个无效token
                            [T, T, ..., T]]    ← 样本2，256个有效token
```

**如果压缩成一维，样本 1 和样本 2 的差异就无法表达。**

同理，`obs.tokenized_prompt_mask` 也是 `[B, L]` 的——不同样本的 prompt 长度不同，padding 的位置不同。

## 那为什么 `ar_mask` 可以是一维？

因为 `ar_mask` 只取决于 **token 的类别结构**——"前 N 个 token 是图像，接着 M 个是文本"。这个结构**对所有 batch 样本是一样的**：

```python
# ar_mask 的构建
ar_mask += [False] * image_tokens.shape[1]   # 图像 token 全是 False
ar_mask += [False] * tokenized_inputs.shape[1]  # 文本 token 全是 False
```

不同样本可能图像有效/无效、文本长/短，但"哪些位置是图像 token、哪些是文本 token"的**排列结构**是一样的——因为所有样本都有相同数量的图像（只是某些图像内容可能无效）和相同 `max_token_len` 的文本（只是某些 token 是 padding）。

## 总结

| 掩码         | 有 B 维？       | 原因                                                                 |
| ------------ | --------------- | -------------------------------------------------------------------- |
| `input_mask` | **是** `[B, N]` | 同一位置对样本 A 可能有效（真实图像），对样本 B 可能无效（填充图像） |
| `ar_mask`    | **否** `[N]`    | 所有样本的 token 结构类型（图像/文本的位置）是固定的                 |

`ar_mask` 回答的是"这个位置是图像还是文本"——这是模型架构决定的，与具体数据无关。  
`input_mask` 回答的是"这个 token 是有效输入还是填充"——这是由具体数据决定的，不同样本可以不同。

所以 `make_attn_mask` 中 `ar_mask` 能被 `broadcast_to` 成 `[B, N]`：

```python
mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
```

因为它对所有样本都一样，广播复制成 B 份即可。而 `input_mask` 不行——它的 B 维承载着真实的信息差异。
    """

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        """编码后缀（Suffix）：将机器人状态和噪声动作编码为 token 序列。

        什么是"后缀"？
          后缀是模型中所有输入 token 的后半部分，包括：
            - 状态 token（仅 π₀）：编码当前机器人状态
            - 动作 token：编码当前时间步的噪声动作

          后缀的注意力模式：
            - 第一个 token（状态或第一个动作）的 ar_mask=True →
              表示这是一个新块的开始，前缀不能 attend 到它
            - 后缀内的最后一个动作 token 的 ar_mask=True →
              表示生成到动作块边界，之后的 token 之间是因果的

        π₀ 与 π₀.₅ 的关键区别：
          π₀：
            - 有一个额外的 state token（通过 state_proj 编码）
            - 时间嵌入和动作嵌入在输入 MLP 之前就拼接（concat）
          π₀.₅：
            - 没有 state token（状态通过离散 token 输入到前缀中）
            - 时间嵌入通过 adaRMS 归一化条件注入，而不是拼接

        Args:
            obs: 观测数据（包含状态信息）。
            noisy_actions: 噪声动作，形状 [B, action_horizon, action_dim]。
            timestep: 当前扩散时间步，形状 [B]。取值范围 [0, 1]，
                     1 = 纯噪声，0 = 目标数据。

        Returns:
            包含 (tokens, input_mask, ar_mask, adarms_cond) 的四元组。
            adarms_cond 是 π₀.₅ 的 adaRMS 条件（π₀ 返回 None）。
        """
        """
        ar_mask 是自回归掩码（Autoregressive Mask），它控制 Transformer 注意力中 token 之间的"块归属"——决定哪些 token 属于同一个组（可以互相看到），哪些属于不同的组（只能单向因果看）。
        """

        input_mask = []
        ar_mask = []
        tokens = []

        if not self.pi05:
            # ---- π₀：添加状态 token ----
            # 通过 state_proj 将状态编码为单个 token
            state_token = self.state_proj(obs.state)[:, None, :]  # [B, 1, emb]
            tokens.append(state_token)

            # 状态 token 总是有效的
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))

            # 状态 token 的 ar_mask=True：
            # 表示"前面的 token（图像/文本）不能 attend 到状态和动作 token"
            ar_mask += [True]

        # ---- 编码动作 ----
        # 将噪声动作投影到嵌入空间
        action_tokens = self.action_in_proj(noisy_actions)  # [B, action_horizon, emb]

        # ---- 编码时间步 ----
        # 使用正弦-余弦位置编码将标量时间步 t 编码为向量
        # 敏感度范围 [4e-3, 4.0]：控制频率范围
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)

        if self.pi05:
            # ---- π₀.₅ 路径：时间嵌入通过 MLP + adaRMS ----
            # 时间嵌入通过两层 MLP（带 Swish 激活）处理
            # 输出作为 adaRMS 归一化的条件（condition）
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)

            # 动作 token 保持原样（时间信息通过 adaRMS 注入，不在这里混合）
            action_expert_tokens = action_tokens
            adarms_cond = time_emb  # adaRMS 条件
        else:
            # ---- π₀ 路径：时间嵌入与动作嵌入拼接 ----
            # 将时间嵌入扩展到与动作 token 相同的序列长度
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)

            # 在最后一个维度拼接时间和动作嵌入 → [B, T, 2*emb]
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)

            # 通过两层 MLP（带 Swish 激活）融合时间和动作信息
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)

            action_expert_tokens = action_time_tokens
            adarms_cond = None  # π₀ 不使用 adaRMS

        # ---- 添加动作 token ----
        tokens.append(action_expert_tokens)

        # 所有动作 token 都是有效的
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))

        # 动作 token 的 ar_mask：
        #   第一个动作 token：True（新块开始，前缀不能 attend 到这里）
        #   其余动作 token：False（在同一动作块内可以互相 attend）
        ar_mask += [True] + ([False] * (self.action_horizon - 1))

        # ---- 拼接所有后缀 token ----
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)

        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        """计算流匹配损失（Flow Matching Loss）。

        这是训练过程中的核心方法。它实现了流匹配的训练目标：

        损失函数：
          L = MSE(v_t, u_t)

          其中：
            v_t = 模型预测的速度场（velocity field）
            u_t = 真实速度 = ε - a₀（noise - 真实动作）

        训练步骤：
          1. 对观测数据进行预处理（随机增强等）
          2. 从 Beta(1.5, 1) 分布采样时间步 t
             （Beta(1.5, 1) 倾向于采样靠近噪声端的时间步，
             这有助于模型学习去噪的早期阶段）
          3. 构造带噪声的插值：x_t = t·ε + (1-t)·a₀
          4. 计算真实速度：u_t = ε - a₀
          5. 前向传播：前缀(图像+文本) + 后缀(状态+噪声动作+时间)
          6. 从输出中提取速度预测 v_t
          7. 计算 MSE 损失

        Args:
            rng: 随机数生成器，将被拆分为预处理、噪声采样、时间采样三个子生成器。
            observation: 观测数据（图像、文本、状态等）。
            actions: 真实动作（目标动作）。
            train: 是否在训练模式（影响观测预处理）。

        Returns:
            每个样本的损失值，形状 [batch_size, action_horizon]。
        """
        # ---- 步骤1：准备随机数 ----
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)

        # ---- 步骤2：预处理观测数据 ----
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        # ---- 步骤3：采样噪声和时间步 ----
        batch_shape = actions.shape[:-2]  # 通常就是 [batch_size]

        # 采样高斯噪声 ε ~ N(0, I)，形状与动作相同
        noise = jax.random.normal(noise_rng, actions.shape)

        # 从 Beta(1.5, 1) 采样时间步 t，然后缩放到 [0.001, 1.0)
        # Beta(1.5, 1) 的概率密度偏向 0 附近（靠近数据端），
        # 但乘以 0.999 并加 0.001 后偏移到靠近噪声端
        # 这样做是为了让模型更多地学习"去噪早期"阶段
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001

        # 扩展维度以便广播到动作的 [B, T, D] 形状
        time_expanded = time[..., None, None]
        # x_t = t·ε + (1-t)·a₀：从数据到噪声的线性插值
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        # u_t = ε - a₀：真实速度场（速度方向从数据指向噪声）
        u_t = noise - actions

        # ---- 步骤4：模型前向传播 ----
        # 一次完整的前向传播，包含前缀和后缀
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)

        # 拼接前缀和后缀的掩码
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)

        # 构建注意力掩码
        attn_mask = make_attn_mask(input_mask, ar_mask)

        # 计算位置编码（每个 token 在有效序列中的位置）
        positions = jnp.cumsum(input_mask, axis=1) - 1

        # 通过 Gemma LLM 处理完整的 token 序列
        # 传入了两个 token 序列：[前缀, 后缀]，分别对应两个 Gemma 配置
        # adarms_cond=[None, adarms_cond]：只有动作专家部分使用 adaRMS
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )

        # 从后缀输出中提取动作部分的预测（取最后 action_horizon 个 token）
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        # ---- 步骤5：计算损失 ----
        # MSE(v_t, u_t)：模型预测的速度场应与真实速度场一致
        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        """从噪声中采样动作（推理/部署阶段）。

        这实现了"流匹配采样"（Flow Matching Sampling）过程。
        与扩散模型的去噪过程类似，但使用的是"速度场"而不是"噪声预测"。

        采样过程：
          1. 从高斯噪声 x₁ ~ N(0, I) 出发
          2. 从 t=1（纯噪声）到 t=0（目标数据），沿速度场 v_t 的负方向步进
          3. 使用简单的欧拉方法（Euler method）进行数值积分
          4. 最终得到 x₀ ≈ 真实动作

        数学公式：
          x_{t+dt} = x_t + dt * v_t
          其中 dt = -1/num_steps（从 1 走向 0）

        重要约定：
          这里的 t=1 是噪声，t=0 是目标数据。
          这与 π₀ 论文中的符号相反（论文中 t=0 是噪声，t=1 是数据）。
          注释中明确为此道歉了 😅

        KV 缓存优化：
          在采样过程中，前缀（图像+文本）是固定不变的。
          我们可以在第一步就计算并缓存前缀的 Key/Value，
          后续步骤只处理后缀部分，大幅减少计算量。

        Args:
            rng: 随机数生成器（用于初始化噪声）。
            observation: 观测数据（图像、文本、状态等）。
            num_steps: 去噪步数。步数越多，质量越高但速度越慢。默认 10 步。
            noise: 初始噪声。如果为 None，则从标准正态分布采样。

        Returns:
            预测的动作，形状 [batch_size, action_horizon, action_dim]。
        """
        observation = _model.preprocess_observation(None, observation, train=False)

        # 时间步长（负数，因为从 t=1 走向 t=0）
        dt = -1.0 / num_steps

        batch_size = observation.state.shape[0]

        # 如果没有提供初始噪声，从标准正态分布采样
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # ============ 第一步：填充 KV 缓存 ============
        # 对前缀（图像+文本）做一次前向传播，缓存 Key/Value
        # 这样后续步骤只需要处理后缀，大幅加速
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        # ============ 第二步：逐步去噪 ============
        def step(carry):
            """单步去噪：从 x_t 走到 x_{t+dt}。

            carry: (x_t, time) 当前带噪声的动作和时间步
            """
            x_t, time = carry

            # 编码后缀（状态 + 噪声动作 + 时间）
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )

            # 构建注意力掩码：
            # suffix_attn_mask: 后缀 token 之间的注意力 [B, S, S]
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)

            # prefix_attn_mask: 后缀 token 对前缀 token 的注意力 [B, S, P]
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])

            # full_attn_mask: 后缀 token 对全部 token（前缀+后缀）的注意力 [B, S, P+S]
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)

            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )

            # 计算后缀 token 的位置（在完整序列中的索引）
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            # 前向传播（使用 KV 缓存，只计算后缀）
            # 传入 [None, suffix_tokens]：前缀传 None（使用缓存），后缀传新 token
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None  # 验证前缀确实被跳过了

            # 从输出提取速度预测
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            # 欧拉步进：x_{t+dt} = x_t + dt * v_t
            return x_t + dt * v_t, time + dt

        def cond(carry):
            """停止条件：当 time < -dt/2（接近于 0）时停止。

            使用 -dt/2 作为阈值是为了数值稳定性：
            避免浮点误差导致无限循环。
            """
            x_t, time = carry
            return time >= -dt / 2

        # 使用 jax.lax.while_loop 执行去噪循环
        # 这会被 JAX 编译为 XLA 的高效循环（不会在 Python 层面迭代）
        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))

        return x_0


"""
`sample_actions` 方法是 π₀ 模型**推理（部署）阶段的核心方法**，它的作用是：**从纯噪声出发，通过流匹配的逆过程，逐步去噪得到预测的机器人动作**。

## 整体流程

```
纯噪声 x₁ ~ N(0, I)  ───→  去噪步 × num_steps  ───→  预测动作 x₀
     t=1                                              t=0
```

## 分步详解

### 1. 预处理观测（第805行）
```python
observation = _model.preprocess_observation(None, observation, train=False)
```
对图像等观测数据做预处理（不进行随机增强，因为推理时是确定性的）。

### 2. 第零步：填充 KV 缓存（第818-822行）
```python
prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
_, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
```
这是**关键优化**：前缀（图像 token + 文本 token）在去噪过程中是**不变的**。所以只做一次前向传播，把 Key/Value 缓存下来（`kv_cache`），后续每一步只需要处理后缀（状态 + 噪声动作）。

### 3. 逐步去噪（第824-884行）

在 `jax.lax.while_loop` 中执行 `cond → step → cond → step → ...`：

**`step`（单步去噪）：**
1. 编码后缀 → 状态 + 当前噪声动作 `x_t` + 时间步 `t`
2. 用 KV 缓存做一次前向传播（只算后缀）
3. 从输出中提取**速度预测** `v_t`
4. **欧拉步进**：`x_{t+dt} = x_t + dt * v_t`

**`cond`（停止条件）：**
```python
return time >= -dt / 2  # 当 t 接近 0 时停止
```

## 直观理解

想象你有一张被噪声覆盖的照片（`x₁`），你要一步步把它恢复清晰（`x₀`）：

| 概念             | 类比                                   |
| ---------------- | -------------------------------------- |
| `x_t`            | 当前"半模糊半清晰"的动作               |
| `v_t`            | 模型预测的"去噪方向"——该往哪个方向调整 |
| `dt`             | 每一步的步长（负数，因为朝清晰方向走） |
| `x_t + dt * v_t` | 沿着去噪方向走一小步                   |

**每一步的核心公式：** `x_{t+dt} = x_t + dt * v_t`

- `dt = -1/num_steps`：总共走 `num_steps` 步
- `v_t` 由模型根据（当前噪声动作 + 状态 + 图像 + 语言指令）预测

## 与训练时的关系

```
训练： compute_loss
       真实动作 a₀ + 噪声 ε → 插值 x_t → 模型预测 v_t → MSE(v_t, ε - a₀)
       让模型学会"预测从噪声到数据的去噪方向"

推理： sample_actions
       纯噪声 x₁ → 循环 x_t + dt·v_t → 预测动作 x₀
       应用模型学会的"去噪方向"，一步步还原出动作
```

## 默认参数

- `num_steps=10`：默认 10 步去噪（步数越多质量越高，但速度越慢）
- `noise=None`：未指定时从标准正态分布采样初始噪声
"""


"""
**`num_steps` 和训练完全无关，可以推理时随便改。**

## 为什么？

对比训练和推理的数学：

### 训练时（`compute_loss`）

模型训练的目标是：给定当前时间 `t` 和带噪动作 `x_t`，**预测速度场 `v_t`**。

```python
# 第 724 行：随机采样 t，永远只做 1 步
time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
x_t = time * noise + (1 - time) * actions    # 线性插值
u_t = noise - actions                        # 真实速度
v_t = self.action_out_proj(suffix_out[...])  # 模型预测速度
loss = MSE(v_t, u_t)                         # 让预测速度逼近真实速度
```

每个 batch 模型只看**一个随机时间步** `t`，学的是"在这个时间点上速度是多少"。因为 `t` 均匀覆盖 [0.001, 1)，**模型学到的是整个连续速度场**。

### 推理时（`sample_actions`）

```python
dt = -1.0 / num_steps           # 步长
for t in [1.0, 0.9, 0.8, ...]: # 从 t=1 逐步走到 t=0
    v_t = model(x_t, t)         # 模型在每个时间点预测速度
    x_{t+dt} = x_t + dt * v_t   # 欧拉积分：沿速度场走一步
```

**`num_steps` 只控制欧拉积分的步长。步数越多 → 每步跨得越小 → 结果越精确。** 这和训练完全解耦。

### 类比

```
你训练了一个"在龙卷风里任意位置感知风速"的模型。
  - 训练：模型在不同位置感知风速（学习速度场）
  - 推理：你想从龙卷风里飞出来

num_steps=1 ： 跨一大步飞出来  →  可能飞偏
num_steps=10： 走 10 小步      →  更精准
num_steps=50： 走 50 小步      →  更精准但更慢
```

**模型的能力是一样的，你只是控制积分精度。** 实际上默认 10 步就是经验上平衡速度和质量的折中值，你想要更好的质量就调到 20、50；想要更快就调到 3、5。
"""




"""
@src/openpi/models/pi0.py 中 Pi0模型是否限制了Observation中图像的数量？如果不同平台相机数量不一样有何影响？


现在清楚了，来完整回答。

---

## 模型层面：不限制相机数量

Pi0 模型本身对相机数量是**完全无限制**的。核心原因在 `embed_prefix` 的实现：

```python
# pi0.py:427
for name in obs.images:
    image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
    tokens.append(image_tokens)
    input_mask.append(...)   # 根据 image_masks 设置有效/无效
```

它是一个 **`for` 循环遍历字典**——不管你传 1 张、3 张还是 5 张图像，只要 `obs.images` 字典里有，它就逐张编码，然后全部拼到序列里。

再看 `Observation` 的类型定义：

```python
# model.py:141
images: dict[str, at.Float[ArrayT, "*b h w c"]]      # 字典，不限制键数量
image_masks: dict[str, at.Bool[ArrayT, "*b"]]         # 同样
```

**没有任何长度检查、数量校验或固定长度的假设。** 从模型架构角度，你传 1 个还是 10 个摄像头都能跑。

---

## 那影响在哪里？

影响不在**模型架构**，在**数据流水线的一致性**：

### 1. 前缀长度 = 所有图像 token 数之和

```
每张图像 → SigLIP → ~256 个图像 token
提示文本 → Gemma → ~48~200 个文本 token

总前缀长度 = 相机数 × 256 + 文本 token 数
```

如果训练时用 3 个相机（前缀 = `3×256 + N`），推理时突然传 5 个相机（前缀 = `5×256 + N`）：

- **JAX 模型** → 新输入形状会导致 XLA **重新编译**（首次很慢）
- **PyTorch 模型** → 不一定报错，但推理结果可能不同（训练时未见过这么多相机输入的情况）

### 2. 相机数量和训练数据决定了模型"见过"什么

模型虽然架构上不限制相机数，但**预训练时用的是固定的相机配置**：

```python
# pi0_config.py:179-187 — 基座模型训练时的输入规格
images={
    "base_0_rgb": ...,           # 主摄像头
    "left_wrist_0_rgb": ...,     # 左腕
    "right_wrist_0_rgb": ...,    # 右腕
}
```

所以实际的"约定"是 3 个相机。平台策略通过 image_mask 机制处理缺失摄像头：

```python
# droid_policy.py:48-51 - DROID 只有 2 个真实摄像头
match self.model_type:
    case _model.ModelType.PI0 | _model.ModelType.PI05:
        images = (base_image, wrist_image, np.zeros_like(base_image))  # 第3个用黑图填充
        image_masks = (np.True_, np.True_, np.False_)                   # 标记无效
```

### 3. 实际平台对比

| 平台       | 真实相机数              | inputs_spec 定义的相机 | image_mask                                |
| ---------- | ----------------------- | ---------------------- | ----------------------------------------- |
| **ALOHA**  | 3（主摄 + 左腕 + 右腕） | 3                      | 全部 True                                 |
| **DROID**  | 2（外景 + 左腕）        | 3                      | `[True, True, False]` — 第 3 个是黑图填充 |
| **LIBERO** | 2（主摄 + 腕部）        | 3                      | `[True, True, False]` — 第 3 个是黑图填充 |

---

## 一句话总结

**模型不对图像数量做任何限制**——它就是遍历 `obs.images` 字典，来多少处理多少。
唯一的要求是**训练时的配置和推理时的配置一致**（相同的 inputs_spec），否则 XLA 会重新编译。
多出来的摄像头直接加到前缀序列里，缺失的摄像头用 `image_mask=False` 让模型跳过——这就是整个机制的弹性所在。
"""

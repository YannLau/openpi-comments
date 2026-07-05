# ============================================================
#  scripts/train.py — OpenPI 模型训练主脚本
#
#  本脚本是 openpi 库的训练入口，负责：
#    1. 解析训练配置（模型结构、数据集、超参数等）
#    2. 初始化模型参数（支持预训练权重加载）
#    3. 创建数据加载器并构建训练循环
#    4. 执行前向传播 → 计算损失 → 反向传播 → 参数更新
#    5. 自动保存检查点、记录日志（wandb）
#
#  支持 JAX (Flax NNX) 框架，使用 FSDP (Fully Sharded Data Parallel)
#  进行分布式训练。
# ============================================================

import dataclasses  # 用于操作 dataclass（如复制、替换字段）
import functools   # 函数式工具，这里用于 partial 固定部分函数参数
import logging     # 日志模块
import platform    # 获取运行平台信息
from typing import Any  # 类型标注

# --- JAX / Flax 生态核心库 ---
import etils.epath as epath        # Google 的路径库，统一处理本地/云存储路径（如 GCS）
import flax.nnx as nnx             # Flax NNX — JAX 的新一代神经网络 API（基于命名轴）
from flax.training import common_utils  # Flax 训练通用工具，如 stack_forest 聚合指标
import flax.traverse_util as traverse_util  # Flax 树形结构遍历工具（如 flatten_dict / unflatten_dict）
import jax                          # JAX 核心
import jax.experimental            # JAX 实验性功能
import jax.numpy as jnp            # JAX 的 NumPy 接口（在 GPU/TPU 上运行）
import numpy as np                 # 标准 NumPy
import optax                       # JAX 优化器库（AdamW、梯度裁剪、学习率调度等）

import tqdm_loggable.auto as tqdm  # 增强版 tqdm 进度条（兼容日志系统）
import wandb                       # Weights & Biases — 实验跟踪与可视化平台

# --- openpi 内部模块（以 _ 前缀命名以与标准库区分）---
import openpi.models.model as _model            # 基础模型定义（BaseModel / Observation / Actions）
import openpi.shared.array_typing as at          # 数组类型检查与标注工具（@at.typecheck）
import openpi.shared.nnx_utils as nnx_utils      # Flax NNX 工具函数（state_map 等）
import openpi.training.checkpoints as _checkpoints    # 检查点管理（保存 / 恢复）
import openpi.training.config as _config                # 训练配置（TrainConfig 与所有命名配置）
import openpi.training.data_loader as _data_loader      # 数据加载器（支持 LeRobot / RLDS 数据集）
import openpi.training.optimizer as _optimizer          # 优化器配置（AdamW + 学习率调度）
import openpi.training.sharding as sharding             # FSDP 分布式分片工具
import openpi.training.utils as training_utils          # 训练工具函数（TrainState 等）
import openpi.training.weight_loaders as _weight_loaders # 预训练权重加载器


def init_logging():
    """初始化日志格式，使终端输出更易读。

    将标准日志级别缩写为单个字母（如 INFO → I），
    并统一时间戳格式，方便快速扫视训练日志。
    """
    # 标准 logging 级别到单字母的映射
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    # 自定义 Formatter，将日志级别名替换为缩写
    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    # 日志格式示例: "14:30:01.123 [I] Training step 100 ... (12345:train.py:42)"
    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    # 获取根日志器并设置格式
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    """初始化 wandb（Weights & Biases）实验跟踪。

    功能：
      - 如果 enabled=False，将 wandb 设为 disabled 模式（不发送数据）
      - 如果是恢复训练（resuming=True），从 checkpoint 目录读取之前的 run_id
        并恢复同一个 wandb 运行
      - 如果是新实验，创建一个新的 wandb 运行，并将 run_id 保存到 checkpoint 目录
      - 可选择将代码上传到 wandb（log_code=True）

    参数：
      config:     训练配置对象
      resuming:   是否为恢复训练
      log_code:   是否将项目代码上传到 wandb
      enabled:    是否启用 wandb（可以在配置中关闭）
    """
    if not enabled:
        # 不发送任何数据到 wandb，但代码无需改动
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"检查点目录 {ckpt_dir} 不存在。")

    if resuming:
        # 恢复训练：读取之前保存的 wandb run ID，继续同一条记录
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        # 新实验：创建新 run，保存 run ID 以便后续恢复
        wandb.init(
            name=config.exp_name,                               # 实验名称
            config=dataclasses.asdict(config),                   # 将配置序列化为 dict 并记录
            project=config.project_name,                         # wandb 项目名
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)    # 持久化 run ID

    if log_code:
        # 将项目代码（父目录下的所有文件）上传到 wandb
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """加载预训练权重并进行验证。

    流程：
      1. 用 loader 从磁盘 / GCS 加载权重
      2. 验证加载后的参数结构与预期形状匹配（形状 + 数据类型）
      3. 过滤掉未加载的占位符（jax.ShapeDtypeStruct），只返回实际加载的参数

    在 JAX 中，未初始化的参数用 jax.ShapeDtypeStruct（仅类型信息，无实际数据）
    表示。如果一个键对应的值是 ShapeDtypeStruct 而不是真正的数组，说明
    weight_loader 没有加载该参数（即由随机初始化填充）。

    参数：
      loader:       权重加载器，定义了从哪加载以及如何映射权重名称
      params_shape: 完整的参数结构（仅形状信息，用于校验）

    返回：
      一个 dict，仅包含实际加载了权重的参数（子集）
    """
    loaded_params = loader.load(params_shape)                            # 执行加载
    at.check_pytree_equality(expected=params_shape, got=loaded_params,   # 校验形状和类型
                              check_shapes=True, check_dtypes=True)

    # 从加载结果中移除 jax.ShapeDtypeStruct（未加载的占位符）
    # 返回的 dict 只包含真正有数据的参数
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    """初始化训练状态（模型参数、优化器、EMA 等）。

    这是训练启动的关键函数，执行以下操作：
      1. 创建优化器（AdamW + 学习率调度）
      2. 初始化模型参数（随机初始化或加载预训练权重）
      3. 将冻结参数转为 bfloat16 以节省显存
      4. 计算参数分片方案（FSDP）
      5. 如果 resume=True，只返回状态形状（实际参数从检查点恢复）
      6. 否则加载预训练权重并混合到模型中

    参数：
      config:   训练配置
      init_rng: JAX 随机数生成器键
      mesh:     JAX 设备网格（用于 FSDP 分片）
      resume:   是否为恢复训练

    返回：
      (TrainState, sharding_spec) 元组
    """
    # 创建优化器：包含优化算法（AdamW）和学习率调度（如 cosine decay）
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    # --- 内部函数：实际初始化逻辑 ---
    # 这是被 jax.jit 编译的核心函数
    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)

        # 根据配置创建模型（仅定义结构，随机初始化参数）
        model = config.model.create(model_rng)

        # 如果有预训练权重，将其合并到模型中
        if partial_params is not None:
            # nnx.split 将模型拆分为：计算图定义(graphdef) + 参数状态(state)
            graphdef, state = nnx.split(model)
            # 用加载的部分权重替换模型中对应的参数
            # 如果 partial_params 包含模型中不存在的键，这里会报错
            state.replace_by_pure_dict(partial_params)
            # 重新合并为完整模型
            model = nnx.merge(graphdef, state)

        # 提取模型的所有参数
        params = nnx.state(model)

        # 将需要冻结（不训练）的参数转为 bfloat16，节省显存
        # bfloat16 在保持足够精度的同时，将内存占用减半
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        # 构建完整的训练状态
        return training_utils.TrainState(
            step=0,                                                         # 当前训练步数（从 0 开始）
            params=params,                                                  # 模型参数
            model_def=nnx.graphdef(model),                                  # 模型计算图定义（用于重建模型）
            tx=tx,                                                          # optax 优化器
            opt_state=tx.init(params.filter(config.trainable_filter)),      # 优化器状态（如 Adam 动量）
            ema_decay=config.ema_decay,                                     # EMA 衰减率（可选）
            ema_params=None if config.ema_decay is None else params,        # EMA 参数（可选）
        )

    # ---- 先在不执行计算的情况下获取状态形状 ----
    # jax.eval_shape 只跟踪形状和 dtype，不实际分配内存
    # 这样我们可以提前设计分片方案
    train_state_shape = jax.eval_shape(init, init_rng)

    # 计算 FSDP 分片方案：决定每个参数分布在哪些设备上
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    # ---- 如果是恢复训练，直接返回形状信息（后续从 checkpoint 加载）----
    if resume:
        return train_state_shape, state_sharding

    # ---- 否则：加载预训练权重，初始化真实状态 ----
    # 加载预训练权重（只加载匹配的部分，不匹配的保持随机初始化）
    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())

    # 权重参数不需要分片（在所有设备上复制一份，replicated）
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # JIT 编译并执行 init 函数
    # donate_argnums=(1,) 表示 partial_params 的内存可以被"捐赠"（覆盖），节省内存
    train_state = jax.jit(
        init,
        donate_argnums=(1,),          # 捐赠 partial_params 缓冲区
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,  # 输出按照 FSDP 方案分片
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """单步训练（前向 + 反向传播）。

    这是训练循环的核心函数，完成一个 batch 的完整训练步骤：
      1. 从 graphdef + params 重建模型（nnx.merge）
      2. 计算损失（前向传播）
      3. 计算梯度（反向传播，自动微分）
      4. 更新优化器状态并应用梯度更新
      5. 如果需要，更新 EMA（指数移动平均）参数
      6. 记录训练指标（损失、梯度范数、参数范数）

    参数：
      config: 训练配置
      rng:    JAX 随机数生成器键
      state:  当前训练状态（参数、优化器状态等）
      batch:  一个 batch 的数据（观测值 + 动作标签）

    返回：
      (new_state, info_dict) — 更新后的训练状态和训练指标
    """
    # === 第一步：从计算图定义 + 参数重建模型 ===
    model = nnx.merge(state.model_def, state.params)
    model.train()  # 切换到训练模式（启用 Dropout 等训练时特有的操作）

    # === 第二步：定义损失函数 ===
    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        # compute_loss 根据模型类型（π₀ 流匹配 / π₀-FAST 自回归）计算不同的损失
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)  # 对 batch 取平均

    # 为当前步骤生成专用的随机数（确保每步的 dropout mask 不同）
    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch  # 解包 batch

    # === 第三步：自动计算梯度 ===
    # 使用 nnx.DiffState(0, trainable_filter) 标记只对可训练参数求导
    # 冻结参数（如某些预训练权重）不参与梯度计算
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    # === 第四步：更新参数 ===
    # 提取当前可训练参数
    params = state.params.filter(config.trainable_filter)
    # optax 优化器计算参数更新量（考虑 Adam 动量、学习率等）
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    # 应用更新：new_params = params - learning_rate * updates
    new_params = optax.apply_updates(params, updates)

    # 将更新后的可训练参数写回模型
    nnx.update(model, new_params)
    # 提取完整的模型参数（包括冻结参数，它们没变）
    new_params = nnx.state(model)

    # === 第五步：更新训练状态 ===
    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)

    # 如果启用了 EMA，更新 EMA 参数
    # EMA 是对参数做指数移动平均，通常用于推理（比原始参数更稳定）
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                state.ema_params,
                new_params,
            ),
        )

    # === 第六步：收集训练指标 ===
    # 筛选出"核"参数（kernel weights，即权重矩阵而非偏置/scale等）
    # 通常核参数的范数能更好地反映模型规模
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),  # 排除 bias/scale 等
            lambda _, x: x.value.ndim > 1,  # 只取多维参数（排除标量和向量）
        ),
    )
    info = {
        "loss": loss,                                            # 平均损失值
        "grad_norm": optax.global_norm(grads),                   # 全局梯度范数（判断梯度爆炸）
        "param_norm": optax.global_norm(kernel_params),          # 参数范数（辅助监控）
    }
    return new_state, info


def main(config: _config.TrainConfig):
    """训练主函数 —— 组装所有组件并执行训练循环。

    完整流程：
      1. 初始化日志系统
      2. 验证 batch_size 可被设备数整除
      3. 创建 JAX 设备网格（mesh）和分片方案
      4. 初始化/恢复检查点管理器
      5. 初始化 wandb 实验跟踪
      6. 创建数据加载器
      7. 初始化训练状态（模型参数 + 优化器）
      8. JIT 编译训练步骤函数
      9. 执行训练循环：迭代数据 → 前反向传播 → 记录日志 → 保存检查点
    """
    # ---- 1. 初始化日志 ----
    init_logging()
    logging.info(f"Running on: {platform.node()}")  # 打印运行节点名（便于分布式调试）

    # ---- 2. 验证 batch 大小配置 ----
    # JAX 中每个设备处理 batch_size / num_devices 个样本
    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    # ---- 3. JAX 配置与随机数 ----
    # 启用 JAX 编译缓存目录，避免重复编译（跨进程复用）
    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    # 初始化 JAX 随机数生成器（JAX 使用显式的随机数键，而非全局种子）
    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)  # 分为训练用 rng 和初始化用 rng

    # ---- 4. 创建设备网格与分片方案 ----
    # mesh 定义了一组设备和它们的逻辑拓扑（用于 FSDP 分布式训练）
    mesh = sharding.make_mesh(config.fsdp_devices)

    # 数据分片方案：batch 维度分布在所有数据并行设备上
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    # 复制分片方案：所有设备都保留完整副本（用于小数据或控制流）
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    
    
    """ 分片
## JAX 设备网格（Mesh）与 FSDP 分片方案

### 一句话总结

**创建 mesh + 分片方案 = 告诉 JAX "你有多少块 GPU/TPU，以及每块上放参数的哪一部分"**，这是分布式训练（FSDP）的基石。

---

### 1. 设备网格（Mesh）是什么？

```python
# scripts/train.py:372
mesh = sharding.make_mesh(config.fsdp_devices)
```

```python
# sharding.py:17-23
def make_mesh(num_fsdp_devices: int) -> jax.sharding.Mesh:
    mesh_shape = (jax.device_count() // num_fsdp_devices, num_fsdp_devices)
    return jax.make_mesh(mesh_shape, (BATCH_AXIS, FSDP_AXIS))
```

**Mesh = 把一堆 GPU 排列成逻辑网格**。以 8 块 GPU、`num_fsdp_devices=4` 为例：

```
mesh_shape = (8 // 4, 4) = (2, 4)

            ─── FSDP 轴 (4 台设备) ───→
            ┌────┬────┬────┬────┐
            │GPU0│GPU1│GPU2│GPU3│   ← 数据并行轴 (batch 分摊)
    Batch   ├────┼────┼────┼────┤     每块 GPU 处理 BS/2 个样本
    轴 (2)  │GPU4│GPU5│GPU6│GPU7│
            └────┴────┴────┴────┘
```

- **Batch 轴（数据并行）**：batch 被拆成 2 份，每份发给这一行的所有 GPU
- **FSDP 轴（模型分片）**：每个参数的**部分切片**放在这一列的 GPU 上

> 为什么叫"网格"？因为每个设备由 `(batch_idx, fsdp_idx)` 两个坐标唯一标识。

---

### 2. 分片方案（Sharding）是什么？

分片方案告诉 JAX **"这个张量的每一维放在网格的哪个轴上"**。

```python
# sharding.py:30-33
# 三种分片模式：

# (a) 数据分片 —— batch 维度分到所有数据并行设备
data_sharding = NamedSharding(mesh, PartitionSpec(BATCH_AXIS, FSDP_AXIS))
#                          ↑ 相当于 PartitionSpec("batch", "fsdp")

# (b) 复制分片 —— 所有设备保留完整副本
replicated_sharding = NamedSharding(mesh, PartitionSpec())
#                          ↑ 空 = 不切分，每块 GPU 都有一份完整拷贝

# (c) FSDP 参数分片 —— 对模型参数的智能分片
#      每个参数，沿着最大的、能被 fsdp 设备数整除的维度切开
state_sharding = sharding.fsdp_sharding(train_state_shape, mesh)
```

#### (a) 数据分片 — 处理 batch

假设 batch_size=16, 网格 2×4：

```
PartitionSpec("batch", "fsdp")
指：batch 维度按 batch 轴分（切成 2 份，每份 8 个样本）
再按 fsdp 轴分... 但 batch 没有第 2 维了，所以每台 GPU 拿到 8/4 = 2 个样本
                                                              ← 实际是 16 / (2×4) = 每块GPU 2 个
```

#### (b) 复制分片 — 用于小数据

```python
# PartitionSpec() —空 PartitionSpec
# 所有设备都保留完整数据（用于随机种子、控制流等）
```

#### (c) FSDP 参数分片 — 核心

```python
# sharding.py:83-93
# 对每个大矩阵（如 Gemma 的 attention 权重）：
# 沿着最大且能被 FSDP 轴长度整除的维度切开
spec[i] = FSDP_AXIS  # 在这个维度上做切片，分散到 4 台设备
```

权重矩阵形状 `[4096, 4096]`，`fsdp_devices=4`：

```
        沿 axis=0 切开
    ┌──────────────┐
    │   GPU0 拥有  │  1024 行
    ├──────────────┤
    │   GPU1 拥有  │  1024 行
    ├──────────────┤
    │   GPU2 拥有  │  1024 行
    ├──────────────┤
    │   GPU3 拥有  │  1024 行
    └──────────────┘

每块 GPU 只存 1/4 的参数，显存省了 75%！
但计算时需要 all-gather 通信收集完整矩阵。
```

---

### 3. 训练时三者如何配合？

```
                        mesh = make_mesh(4)
                      ┌───────────────────┐
                      │   2 × 4 设备网格    │
                      └───────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                     ▼
  data_sharding         state_sharding        replicated_sharding
  (batch 分片)          (参数 FSDP 分片)       (小数据复制)

      ▼                      ▼                     ▼
  每个设备              每个设备              每个设备
  拿到部分 batch       持有部分参数            持有完整副本
                        (权重切分)            (优化器状态?)
```

**JIT 编译时指定输入/输出的分片方案**：

```python
# scripts/train.py:423-428
ptrain_step = jax.jit(
    train_step,
    in_shardings=(replicated_sharding,    # rng：所有设备相同（复制）
                  train_state_sharding,   # state：参数按 FSDP 分片
                  data_sharding),         # batch：按数据并行分片
    out_shardings=(train_state_sharding,  # 输出 state 保持 FSDP 分片
                   replicated_sharding), # 输出指标所有设备汇总
)
```

---

### 4. 为什么不把所有参数都切开？

`fsdp_sharding()` 里有几个保护机制：

```python
# 太小的数组不切分（默认 < 4MB 就复制）
# 因为这 4MB 的 all-gather 通信开销 > 节省的显存收益
if arr_size < min_size_bytes:
    return PartitionSpec()   # 复制

# 1D 向量不切分（如 bias, layer_norm scale）
if len(array.shape) < 2:
    return PartitionSpec()

# 如果找不到能整除的维度，也不切分
# 例如一个 [7, 7] 的矩阵不能被 4 整除
```

### 直观总结

| 概念                                        | 一句话                                                   |
| ------------------------------------------- | -------------------------------------------------------- |
| **Mesh**                                    | 把 GPU 排成 2D 网格，每个设备有 (batch_id, fsdp_id) 坐标 |
| **数据分片 PartitionSpec("batch", "fsdp")** | 每个 batch 切片后分散到所有设备                          |
| **FSDP 参数分片**                           | 大权重矩阵沿最大可整除维度切开，省 75% 显存              |
| **复制分片 PartitionSpec()**                | 小数据（rng, 指标）所有设备各存一份                      |
| **最终效果**                                | 8 块 GPU 训练一个放不进的模型 → 每块只装 1/8，通信补全   |    
    """

    # ---- 5. 检查点管理 ----
    # 初始化 checkpoint 目录，判断是否在恢复训练
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,   # 每隔多少个 step 保留一个 checkpoint
        overwrite=config.overwrite,       # 是否覆盖已有目录
        resume=config.resume,             # 是否尝试恢复
    )

    # ---- 6. 初始化 wandb ----
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # ---- 7. 创建数据加载器 ----
    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,           # 数据自动分片到各设备
        shuffle=True,                     # 打乱数据
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)               # 预取第一个 batch
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # 将第一个 batch 中的部分图像记录到 wandb，用作 sanity check
    # 将不同摄像头的图像水平拼接，展示前 5 个样本
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    # ---- 8. 初始化训练状态 ----
    # 创建模型实例、优化器，并加载预训练权重（如果有）
    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)  # 等待异步计算完成，确保初始化完毕
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    # 如果是恢复训练，从 checkpoint 恢复参数和优化器状态
    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    # ---- 9. JIT 编译训练步骤 ----                                JIT 编译的是"数值计算图"（computational graph），不是"Python 函数"（Python function）。
    # jax.jit 将 train_step 函数编译为高效的 XLA 计算图
    # in_shardings 指定输入的分片方案，out_shardings 指定输出分片方案
    # donate_argnums=(1,) 表示 state 参数的内存可被复用（in-place 更新），节省显存
    ptrain_step = jax.jit(
        functools.partial(train_step, config),  # 固定 config 参数
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    # ---- 10. 主训练循环 ----
    start_step = int(train_state.step)  # 起始步数（恢复训练时 > 0）
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,  # 进度条宽度自适应终端
    )

    # 存储最近若干步的训练指标，用于平滑日志输出
    infos = []

    for step in pbar:
        # 设置当前设备网格上下文（使分片方案生效）
        with sharding.set_mesh(mesh):
            # 执行一步训练：前向 + 反向 + 参数更新
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)

        # 定期记录日志（默认每 10 步）
        if step % config.log_interval == 0:
            # stack_forest 将列表中的 dict 堆叠为数组
            stacked_infos = common_utils.stack_forest(infos)
            # 对每个指标取平均，并将结果从设备拉取到主机
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            # 格式化输出
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            # 记录到 wandb
            wandb.log(reduced_info, step=step)
            infos = []  # 清空缓存

        # 获取下一个 batch 的数据
        batch = next(data_iter)

        # 定期保存检查点（默认每 5000 步）
        # 条件：到达保存间隔且超过了起始步数，或是最后一步
        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()  # 确保所有检查点写入完成


# ---- 脚本入口 ----
# 使用 _config.cli() 从命令行解析配置名称（如 "pi05_libero"），
# 然后调用 main() 启动训练流程。
#
# 用法示例：
#   XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero --exp-name=my_experiment
if __name__ == "__main__":
    main(_config.cli())

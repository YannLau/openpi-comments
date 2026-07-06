# CLAUDE.md — 中文版

本文档为 Claude Code (claude.ai/code) 在操作此仓库时提供指导。

## 构建 / 测试 / 代码检查 命令

```bash
# 安装依赖（需要 uv）
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
uv sync --group rlds  # 仅在 DROID RLDS 训练时需要

# 代码检查与格式化（pre-commit）
pre-commit run --all-files
ruff check .
ruff format . --check

# 运行测试
uv run pytest -xvs
uv run pytest -xvs src/openpi/transforms_test.py  # 单个测试文件
uv run pytest -xvs -k "test_name"                  # 按名称运行单个测试

# 运行手动标记的测试（默认不会执行）
uv run pytest -xvs --run-manual

# 为某个配置计算归一化统计量
uv run scripts/compute_norm_stats.py --config-name pi05_libero

# 训练模型（JAX）
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero --exp-name=my_experiment

# 训练模型（PyTorch）
uv run scripts/train_pytorch.py pi0_aloha_sim --exp-name pytorch_test

# 启动策略服务
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=checkpoints/pi05_libero/my_experiment/20000

# 将 JAX 检查点转换为 PyTorch 格式
uv run examples/convert_jax_model_to_pytorch.py --checkpoint_dir /path/to/jax/checkpoint --config_name pi0_libero --output_path /path/to/output

# 调试训练（快速冒烟测试）
uv run scripts/train.py debug --exp-name=debug_test
```

## 项目架构

openpi 是 Physical Intelligence 公司开源的机器人 VLA（视觉-语言-动作）库，支持三种模型架构：π₀（流匹配）、π₀-FAST（自回归）和 π₀.₅（升级版流匹配，带知识隔离）。每个模型都有 JAX（Flax NNX）和 PyTorch 两种实现。

### 模块说明

- **`src/openpi/models/`** — 模型架构（JAX/Flax NNX）。`model.py` 定义了 `BaseModel` / `BaseModelConfig` / `Observation` / `Actions` 等核心基类。`pi0_config.py` / `pi0.py` 是 π₀ / π₀.₅ 流匹配模型。`pi0_fast.py` 是 FAST 自回归模型。`gemma.py` / `siglip.py` / `vit.py` 是子组件。`lora.py` 是 LoRA 适配层（用于 Einsum 模块）。`tokenizer.py` 包含 PaligemmaTokenizer 和 FASTTokenizer。

- **`src/openpi/models_pytorch/`** — PyTorch 版本的模型实现（`pi0_pytorch.py`、`gemma_pytorch.py`、`preprocessing_pytorch.py`）。`transformers_replace/` 目录包含需要复制到已安装的 `transformers` 库中的补丁文件（用于支持 AdaRMSNorm、KV 缓存控制、激活精度控制）。详见 README。

- **`src/openpi/policies/`** — 各机器人平台特有的策略封装（`aloha_policy.py`、`droid_policy.py`、`libero_policy.py`），定义了输入/输出变换，将机器人特有的观测键映射到 `model.py` 中定义的统一 `Observation` 结构。`policy_config.py` 从 `TrainConfig` + 检查点创建 `Policy`。`policy.py` 将模型包装在输入/输出变换中，提供 `infer()` 方法。

- **`src/openpi/training/`** — 完整训练流程。`config.py` 包含 `TrainConfig` 数据类和所有命名配置（在 `_CONFIGS` 列表中）。配置系统使用 `DataConfigFactory` 的各种子类（`SimpleDataConfig`、`LeRobotAlohaDataConfig`、`LeRobotLiberoDataConfig`、`RLDSDroidDataConfig`、`LeRobotDROIDDataConfig`）。`data_loader.py` 同时支持 LeRobot（基于 PyTorch，可混洗，`num_workers > 0`）和 RLDS（针对大规模 DROID 数据集定制，`num_workers=0`）数据加载方式。`checkpoints.py` 使用 orbax CheckpointManager 管理检查点。`optimizer.py` 定义优化器配置（AdamW、余弦退火学习率调度）。`weight_loaders.py` 加载预训练检查点（支持 LoRA 权重合并）。`sharding.py` 管理 FSDP 分布式训练。

- **`src/openpi/transforms.py`** — 数据变换管道：`RepackTransform` 重新映射键名；`Normalize` / `Unnormalize` 处理 Z 分数或分位数归一化；`DeltaActions` / `AbsoluteActions` 转换动作类型（差分/绝对）；`TokenizePrompt` / `TokenizeFASTInputs` 处理语言指令的分词；`PadStatesAndActions` 将状态/动作向模型维度对齐。变换通过 `Group(inputs=[...], outputs=[...])` 组织。

- **`src/openpi/shared/`** — 工具模块集合。`download.py` 处理 GCS / HTTP 下载及缓存（默认缓存到 `~/.cache/openpi`，可通过 `OPENPI_DATA_HOME` 环境变量覆盖）。`normalize.py` 加载/保存归一化统计量。`nnx_utils.py` 是 Flax NNX 的辅助函数。`image_tools.py` 负责图像缩放和填充。

- **`src/openpi/serving/`** — `websocket_policy_server.py` 通过 WebSocket 提供策略推理服务，使用 `msgpack_numpy` 序列化数据，支持 `/healthz` 健康检查端点。

- **`packages/openpi-client/`** — 远程策略推理客户端库（`WebsocketClientPolicy`）。还包含 `action_chunk_broker.py`（用于流式处理动作块）和 `runtime/` 运行时框架（用于构建机器人控制循环）。

- **`examples/`** — 各机器人平台的示例和配置：ALOHA 实机/仿真（Docker Compose、训练数据转换）、DROID（RLDS 数据集转换、训练）、LIBERO（数据转换、Docker 评估）、UR5 使用说明、推理 Notebook、JAX→PyTorch 模型转换。

### 核心数据流

1. 训练从一个命名配置开始（例如 `pi05_libero`），通过 `config.get_config()` 或 `config.cli()` 加载
2. `DataConfigFactory.create()` 构建包含变换管线的 `DataConfig`
3. `data_loader.create_data_loader()` 创建数据加载管线：原始数据集 → 重映射变换 → 数据变换 → 归一化 → 模型变换
4. `train.py` / `train_pytorch.py` 执行训练循环
5. 推理时，`policy_config.create_trained_policy()` 创建一个 `Policy`，包含输入和输出变换（包括反归一化）
6. 策略服务器（`serve_policy.py`）加载策略并通过 WebSocket 提供服务

### 模型配置注册

所有训练配置都定义在 `src/openpi/training/config.py` 的 `_CONFIGS` 列表中。每个配置都有唯一 `name`，组合了模型配置、数据配置、权重加载器、优化器调度和训练超参数。调试配置（`debug`、`debug_pi05`）使用 `FakeDataConfig` 进行快速冒烟测试。

### 检查点格式

训练好的 JAX 检查点由 orbax 保存在 `checkpoints/{config_name}/{exp_name}/{step}/` 目录下，包含 `train_state/`、`params/` 和 `assets/` 子目录。`params/` 目录可以通过 `model.restore_params()` 独立加载。PyTorch 检查点保存 `model.safetensors`、`optimizer.pt` 和 `metadata.pt` 三个文件。

## 常见模式

- **新增机器人平台**：创建策略变换类（类似 `DroidInputs` / `DroidOutputs`），编写 `DataConfigFactory` 子类用于数据集配置，然后在 `_CONFIGS` 中注册 `TrainConfig`。
- **LoRA 微调**：在模型配置中设置 `paligemma_variant="gemma_2b_lora"` / `action_expert_variant="gemma_300m_lora"`，通过 `get_freeze_filter()` 获取冻结过滤器，并设置 `ema_decay=None`。
- **离散状态输入（π₀.₅）**：在 `Pi0Config` 中设置 `pi05=True` 和 `discrete_state_input=True`。这会将状态通过分词器当作离散 token 处理，而不是作为连续的 suffix 拼接到输入中。
- **归一化统计量**：使用 `scripts/compute_norm_stats.py` 预计算。在对已知机器人进行微调时，可以通过 `AssetsConfig` 重新加载预训练阶段的归一化统计量。

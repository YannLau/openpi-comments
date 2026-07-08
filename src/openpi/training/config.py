"""
======================================================================
  openpi 训练配置（Training Config）模块
======================================================================

本模块是整个 openpi 训练系统的"总控制台"和"注册中心"。

核心职责：
  1. 定义 TrainConfig 类——所有训练配置的统一数据结构
  2. 定义 DataConfig 类——训练数据的配置和数据变换流水线
  3. 在 _CONFIGS 列表中注册所有的预定义训练配置（命名配置）
  4. 提供 cli() 和 get_config() 接口，让训练脚本按名称加载配置

如何使用：
  - 训练时通过名称引用配置：`uv run scripts/train.py pi0_libero --exp-name=my_experiment`
  - 代码中获取配置：`config = get_config("pi0_libero")`
  - CLI 覆盖配置：`uv run scripts/train.py pi0_libero --batch_size=64 --num_train_steps=50000`

架构概览 —— 一个 TrainConfig 包含什么？
  ┌─────────────────────────────────────────────────┐
  │  TrainConfig                                    │
  │  ├── name            — 配置的唯一标识符          │
  │  ├── model           — 模型架构配置（π₀ / π₀-FAST / π₀.₅）│
  │  ├── data            — 数据配置（数据集、变换流水线）   │
  │  ├── weight_loader   — 权重加载器（从预训练检查点加载） │
  │  ├── optimizer       — 优化器配置（学习率、优化器类型）  │
  │  ├── freeze_filter   — 冻结哪些参数（用于 LoRA 微调）   │
  │  ├── batch_size      — 批大小                      │
  │  ├── num_train_steps — 训练步数                    │
  │  └── ...             — 其他训练超参数               │
  └─────────────────────────────────────────────────┘

命名配置列表 _CONFIGS：
  文件末尾的 _CONFIGS 列表包含了所有预定义的训练配置，涵盖了：
    - ALOHA 机器人（实物和仿真）的推理和微调配置
    - DROID 机器人的推理和微调配置
    - LIBERO 桌面机器人微调配置（含全量微调和 LoRA 低内存微调）
    - 调试配置（使用假数据进行快速测试）
    - RoboArena 和 PolaRiS 等其他平台配置
"""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro  # 一个强大的命令行参数解析库，支持 dataclass

import openpi.models.dummy_model as dummy_model  # 添加自己自定义的测试模型

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.polaris_config as polaris_config
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# 绕开 tyro 在处理 nnx.filterlib.Filter 时的一个问题
# nnx.Filter 是一种用于选择神经网络中特定参数的模式匹配工具
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """资产配置：确定数据流水线所需的静态资源（如归一化统计量）的位置。

    什么是"资产"（assets）？
      资产是训练/推理所需的辅助文件，最主要的是"归一化统计量"（norm stats）。
      在不同平台上训练时，我们需要提前统计数据集的均值和标准差等信息，
      这些信息就存储在资产目录中。

    资产会被复制到检查点（checkpoint）的 `assets/asset_id` 目录下，
    这样检查点就自包含了所有需要的归一化信息，可以独立使用。

    典型使用场景：
      1. 默认情况：从当前配置的资产目录加载
      2. 微调场景：从基础模型检查点的资产目录加载
         （因为微调时我们想复用基础模型已有的归一化统计量，而不是重新计算）

    示例：
        # 从基础模型检查点加载 Trossen 机器人的归一化统计量
        AssetsConfig(
            assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            asset_id="trossen",
        )
    """

    """
## 归一化是什么？

归一化（Normalization）就是**把不同尺度的数据映射到一个统一的范围内**。

想象两个机器人数据样本：

```python
# 原始数据（不同物理量的尺度差异巨大）
关节角度:   [0.5, -1.2, 2.3, -0.8, 1.5, -0.3]   # 弧度，范围 ~[-3, 3]
夹爪位置:   [0.02]                                 # 米，范围 ~[0, 0.1]
末端力:     [12.5, -3.2]                           # 牛顿，范围 ~[-50, 50]
```

如果不归一化，神经网络训练时会出现一个问题：

```
损失贡献：    力（数值大）▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉  （主导梯度）
             角度（数值中）▉▉▉▉▉
             夹爪（数值小）▍               （几乎不贡献梯度）
```

模型会**主要优化"力"的预测误差**，因为它的数值最大，而"夹爪位置"几乎被忽略。但实际上这三个维度同样重要。

归一化后：

```python
# 每个维度都映射到 ~[-1, 1] 范围
关节角度:   [0.12, -0.38, 0.78, -0.25, 0.50, -0.10]
夹爪位置:   [0.40]
末端力:     [0.25, -0.06]

# 现在每个维度对损失的贡献大致公平
```

## 为什么需要预计算？它为什么是一种"资产"？

### 归一化不是"随手算"的

归一化需要知道数据集的**全局统计量**：

- **Z-score 归一化**：需要整个数据集的**均值**和**标准差**
- **分位数归一化**：需要整个数据集的**1% 和 99% 分位数**

这两个统计量都**不能从一个样本算出**——你需要扫描整个数据集才有意义。

> 就像你要知道"中国成年男性的平均身高"，不能只量一个人，需要统计大量样本。

### 预计算流程

```bash
# 1. 先运行脚本，扫描整个数据集，算好统计量
uv run scripts/compute_norm_stats.py --config-name pi05_libero

# 2. 输出保存为 assets 文件
assets/pi05_libero/
├── state/
│   ├── mean.npy      # 状态每个维度的均值
│   └── std.npy       # 状态每个维度的标准差
├── actions/
│   ├── mean.npy
│   └── std.npy
└── ...
```

这些文件就是 **assets（资产）**——它们是一次性算好、反复使用、不随训练变化的静态资源。

### 为什么它要跟着检查点？

看 `AssetsConfig` 的注释：

> 资产会被复制到检查点的 `assets/asset_id` 目录下，这样检查点就自包含了所有需要的归一化信息，可以独立使用。

意思是：

1. 你在一台机器上训练了一个 π₀ 模型
2. 保存了检查点
3. 把检查点发给同事部署到机器人上
4. 同事的机器上**没有**原始数据集，也没有预计算统计量
5. 但检查点自带 `assets/` 目录 → 同事的推理代码能正确反归一化模型输出 → 机器人执行正确的动作

**检查点 = 模型参数 + 归一化统计量 = 一个完整可部署的产物。**

### "微调时复用统计量"的场景

微调配置中常见的模式：

```python
AssetsConfig(
    assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
    asset_id="trossen",     # ← 复用基础模型在 Trossen 数据集上算好的统计量
)
```

为什么可以复用？因为**微调的数据分布和基础模型训练的数据分布相似**——都是同一台机器人在类似环境中的操作数据。没有必要再扫描一遍微调数据集算统计量，直接用基础模型预计算好的即可。

---

## 一句话总结

归一化是**让不同物理量在数值尺度上公平竞争**的手段。它的统计量需要**扫描整个数据集**才能得到，所以是一次性预计算、保存为文件的"资产"。
`AssetsConfig` 就是管理这些预计算统计量的配置——**告诉训练/推理代码去哪里找这些文件**。
    """

    # 资产目录路径。如果未提供，则使用配置的默认 assets_base_dir。
    # 当需要从不同的检查点（如基础模型检查点）加载资产时设置此值。
    assets_dir: str | None = None

    # 资产 ID（即子目录名）。如果未提供，则使用 repo_id。
    # 不同的机器人平台有不同的资产 ID：
    #   - "trossen"  → Trossen（ALOHA）机器人的归一化统计量
    #   - "droid"    → DROID 机器人的归一化统计量
    #   - "libero"   → LIBERO 桌面的归一化统计量
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    """数据配置：定义数据集、归一化和数据变换流水线。

    这是数据流水线的"总配置"，控制着从原始数据集到模型输入的所有转换步骤。
    数据变换分为三个层级（按应用顺序）：

    层级1: repack_transforms（重排变换）
      ↓  仅用于训练时，从数据集格式重排字段

    层级2: data_transforms（数据变换）
      ↓  训练和推理都用，包含平台特定的变换（图像解析、动作变换）

    层级3: model_transforms（模型变换）
      ↓  模型特定的变换（图像缩放、分词、维度填充）

    最终 → 模型输入
    """

    # LeRobot 仓库 ID（如 "physical-intelligence/libero", "lerobot/aloha_sim_transfer_cube_human"）。
    # 如果为 None，则使用假数据（用于调试）。
    repo_id: str | None = None

    # 资产目录中的子目录名，包含该数据集的归一化统计量。
    asset_id: str | None = None

    # 预计算的归一化统计量。如果为 None，则不进行归一化。
    # 字典的键是数据路径（如 "state", "actions"），值是 NormStats 对象。
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # ---- 三个层级的变换 ----

    # 重排变换（第1层）：将数据集特定的格式转换为统一格式。
    # 例如：将 "observation.images.top" 映射为 "images.cam_high"
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)

    # 数据变换（第2层）：包含平台特定的变换，在归一化之前应用。
    # 例如：图像解析、绝对动作转增量动作
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)

    # 模型变换（第3层）：模型特定的变换，在归一化之后应用。
    # 例如：调整图像尺寸为 224x224、文本分词、状态/动作填充到模型维度
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)

    # 如果为 True，使用分位数归一化（基于百分位数）；否则使用标准 Z-score 归一化。
    use_quantile_norm: bool = False

    # 数据加载器用来生成动作序列的键名。动作序列的长度由模型配置中的
    # action_horizon 决定。如果你的 LeRobot 数据集使用不同的键名来表示动作，
    # 需要调整此字段。
    action_sequence_keys: Sequence[str] = ("actions",)

    # 如果为 True，从 LeRobot 数据集的 task 字段中获取 prompt（语言指令）。
    prompt_from_task: bool = False

    # ---- 仅用于 RLDS 数据加载器（目前只用于 DROID 大数据集） ----
    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()


class GroupFactory(Protocol):
    """变换组工厂协议：根据模型配置创建对应的变换组。

    这是一个"工厂方法"模式的接口定义。不同类型的模型（π₀、π₀-FAST、π₀.₅）
    需要不同的变换流水线，GroupFactory 封装了这种创建逻辑。

    __call__ 参数：
        model_config: 模型配置，决定了如何创建变换（例如不同的模型可能需要不同的分词器）。
    """

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """创建变换组。"""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """模型变换工厂：为标准 π₀ 系列模型创建模型变换（第3层）。

    这是 GroupFactory 协议的一个具体实现。它根据模型类型（π₀ / π₀-FAST / π₀.₅）
    创建对应的输入和输出变换。

    创建流程：
      1. 检查 model_config.model_type 确定模型类型
      2. 为每种模型类型组装不同的变换序列
         - InjectDefaultPrompt: 如果没有 prompt，注入一个默认提示
         - ResizeImages: 将图像统一缩放到 224x224
         - TokenizePrompt / TokenizeFASTInputs: 分词
         - PadStatesAndActions: 填充状态/动作到模型维度
         - ExtractFASTActions: （仅 FAST 模型输出）从 token 解码出动作值
    """

    # 如果设置了默认提示文本，当数据中没有 prompt 字段时会自动注入。
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        # Python 3.10+ 的 match-case 语法，根据模型类型选择不同的变换配置
        match model_config.model_type:
            case _model.ModelType.PI0:
                # π₀ 模型（流匹配架构，Flow Matching）的变换
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                # π₀.₅ 模型（改进的流匹配架构，带知识隔离）
                # 相比 π₀，多了 discrete_state_input 参数控制
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                # π₀-FAST 模型（自回归架构，Autoregressive/Fast）
                # 允许通过 fast_model_tokenizer 配置自定义分词器
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    # FAST 模型的输出变换：将模型输出的动作 token 解码为实际动作值
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )
            case _model.ModelType.DUMMY:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    """数据配置工厂（抽象基类）：创建 DataConfig 实例的工厂。

    这是一个"抽象工厂"模式。不同的数据集类型（LeRobot / RLDS / 假数据）
    有不同的配置方式和变换需求，DataConfigFactory 的子类封装了这些差异。

    职责：
      1. 统一提供 repo_id 和 assets 配置字段
      2. 提供 base_config 机制，允许在保留默认值的同时覆盖部分字段
      3. 定义抽象的 create() 方法，子类必须实现
      4. 提供 create_base_config() 方法作为通用基础配置创建逻辑

    子类：
      - FakeDataConfig: 调试用的假数据
      - SimpleDataConfig: 最简单的真数据配置
      - LeRobotAlohaDataConfig: ALOHA 机器人的 LeRobot 格式数据
      - LeRobotLiberoDataConfig: LIBERO 桌面的 LeRobot 格式数据
      - RLDSDroidDataConfig: DROID 机器人的 RLDS 格式数据（适合大数据集）
      - LeRobotDROIDDataConfig: DROID 机器人的 LeRobot 格式数据（适合小数据集）
    """

    # LeRobot 仓库 ID。如果未设置（tyro.MISSING），则不使用真实数据。
    repo_id: str = tyro.MISSING

    # 资产的加载方式（目录和 ID）。
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)

    # 基础配置。如果提供了基础配置，create() 会先复制它，再覆盖/增加字段。
    # 这样允许在保留 DataConfig 默认值的同时，只修改你需要改变的部分。
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """创建数据配置（子类必须实现此方法）。"""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """创建基础数据配置（所有子类共享的逻辑）。

        这个函数做了三件事：
          1. 确定 repo_id（可能为 None）
          2. 确定 asset_id（可能使用 repo_id 作为默认值）
          3. 加载归一化统计量（norm stats）

        Args:
            assets_dirs: 资产目录路径。
            model_config: 模型配置（影响归一化方式的选择）。

        Returns:
            一个填充了基础字段的 DataConfig 实例。
        """
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id  # 如果未指定 asset_id，使用 repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        """从资产目录中加载归一化统计量。

        归一化统计量是训练前预计算的（通过 scripts/compute_norm_stats.py），
        保存在资产目录下。如果找不到，返回 None（跳过归一化）。

        Args:
            assets_dir: 资产目录的路径。
            asset_id: 资产 ID（子目录名）。

        Returns:
            归一化统计量字典，如果找不到则返回 None。
        """
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    """假数据配置：用于调试和快速测试。

    当你不希望加载真实数据集时使用。它会创建一个 DataConfig，其中
    repo_id="fake"，这意味着数据加载器会生成随机假数据而不是读取真实数据。

    典型使用场景：
      - 快速测试训练代码是否能正确运行
      - 在只有 CPU 的机器上调试数据流水线
      - CI/CD 测试（如 debug、debug_pi05 配置）
    """

    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # 返回一个极简的 DataConfig，只有 repo_id 设置
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    """简单数据配置：适用于不需要复杂数据变换配置的数据集。

    当你只需要指定 data_transforms 和 model_transforms，不需要额外的
    平台特定逻辑时使用。这是最简单的"真数据"配置工厂。

    与 LeRobotAlohaDataConfig 等专用工厂的区别：
      专用工厂会添加平台特定的变换（如 delta actions 转换），
      而 SimpleDataConfig 只应用你通过参数传入的变换。

    典型使用场景：
      - DROID 推理配置
      - 数据变换逻辑已在变换工厂中完全定义好的场景
    """

    # 数据变换工厂（可自定义）
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)

    # 模型变换工厂（默认使用 ModelTransformFactory）
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)
    """
    这两行定义的是 **`SimpleDataConfig`** 的两个字段，用于配置数据变换流水线的第2层和第3层。拆开来看：
---

## 类型拆解

### `tyro.conf.Suppress[GroupFactory]`

- **`GroupFactory`** —— 一个协议（接口），定义了一个可调用对象：`(model_config) -> Group`。它接受模型配置，返回一个变换组。
- **`tyro.conf.Suppress[...]`** —— 告诉 tyro（这个项目的 CLI 参数解析库）：这个字段不要在命令行参数中暴露。用户无法通过命令行覆盖它，只能通过代码修改。

所以这个字段的类型是"一个符合 `GroupFactory` 协议的工厂对象"。

### `dataclasses.field(default_factory=...)`

Python dataclass 的标准写法：指定这个字段的默认值由一个工厂函数创建，而不是一个字面值。

---

## 具体含义

| 字段               | `default_factory`       | 含义                                                                                                                                    |
| ------------------ | ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `data_transforms`  | `GroupFactory`          | **默认不添加任何数据变换**。GroupFactory 是一个协议（空的抽象），直接用它做 factory 会返回一个空的 `Group()`。                          |
| `model_transforms` | `ModelTransformFactory` | **默认使用标准的模型变换**。`ModelTransformFactory` 会根据模型类型（π₀、π₀-FAST、π₀.₅）自动创建对应的变换：图像缩放、分词、维度填充等。 |

---

## 数据流

当调用 `SimpleDataConfig.create()` 时（第 516-522 行）：

```python
def create(self, assets_dirs, model_config):
    return dataclasses.replace(
        self.create_base_config(assets_dirs, model_config),
        data_transforms=self.data_transforms(model_config),   # ← 这里调用了工厂
        model_transforms=self.model_transforms(model_config), # ← 这里调用了工厂
    )
```

它会调用：
1. `self.data_transforms(model_config)` —— 调用 `GroupFactory` 实例，返回空的 `Group()`
2. `self.model_transforms(model_config)` —— 调用 `ModelTransformFactory` 实例，根据模型自动生成变换

---

## 对比两种工厂

```python
class GroupFactory(Protocol):
    """协议：输入 model_config，输出 Group"""
    def __call__(self, model_config) -> Group: ...

@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """具体实现：为 π₀/π₀-FAST/π₀.₅ 创建模型变换"""
    default_prompt: str = ""

    def __call__(self, model_config) -> Group:
        # 根据 model_config.model_type 创建不同的变换组
        ...
```

- **`GroupFactory`** 只是一个空壳协议——你完全可以自己写一个新的类实现它，传入自定义的变换逻辑
- **`ModelTransformFactory`** 是框架自带的默认实现，覆盖了标准模型的常用变换

---

## 一句话总结

这两个字段的设计意图是：**当你继承 `SimpleDataConfig` 或构建新数据集配置时，允许你替换数据/模型变换的生成逻辑**。默认的 `data_transforms` 什么都不做，默认的 `model_transforms` 自动适配标准模型——你可以传入自定义的 `GroupFactory` 实现来改变任一层的变换行为。
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # 先创建基础配置，然后添加数据变换和模型变换
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    """ALOHA 机器人数据配置：使用 LeRobot 格式的 ALOHA 数据集。

    ALOHA（A Low-cost Open-source Hardware for Automation）是一个流行的
    双机械臂遥操作平台。其数据包含两个机械臂的关节位置、两个腕部摄像头
    和一个主摄像头。

    与 π₀ 基础模型的训练数据格式不同，标准的 ALOHA 数据：
      - 使用关节角度值，而不是末端执行器位姿
      - 关节值在 ALOHA 的"标准空间"中
      - 不包含 delta action 转换

    这个配置负责处理这些差异，通过以下方式：
      1. adapt_to_pi=True: 将 ALOHA 标准空间的关节值映射到 π₀ 内部运行时使用的空间
      2. use_delta_joint_actions=True: 将绝对关节角度转换为增量动作
      3. repack_transforms: 将原始数据集的键名映射到统一格式
    """

    # 如果为 True，将关节维度转换为增量动作（相对于当前状态的差值）。
    # 夹爪维度保持绝对值不变（夹爪只有开/合，累加没有意义）。
    use_delta_joint_actions: bool = True

    # 如果提供了默认提示文本，当数据中没有 prompt 字段时自动注入。
    default_prompt: str | None = None

    # 如果为 True，将 ALOHA 标准空间的关节和夹爪值转换为 π₀ 内部运行时的空间。
    # 使用标准 ALOHA 数据的人应该设置为 True。
    adapt_to_pi: bool = True

    # 重排变换：将 LeRobot 数据集的字段名映射到 AlohaInputs 期望的字段名。
    # LeRobot 格式：{"observation.images.top": ..., "observation.state": ..., "action": ...}
    # 变换后：     {"images": {"cam_high": ...}, "state": ..., "actions": ...}
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )

    # 从数据集中读取动作序列所使用的键名。
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # 1. 创建基础的数据变换（包含 ALOHA 特定的输入/输出解析）
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )

        # 2. 如果需要，添加增量动作变换
        if self.use_delta_joint_actions:
            # make_bool_mask(6, -1, 6, -1) 的含义：
            #   第一个机械臂的 6 个关节 → 增量（True）
            #   1 个夹爪维度 → 绝对值（False，由 -1 表示）
            #   第二个机械臂的 6 个关节 → 增量（True）
            #   1 个夹爪维度 → 绝对值（False，由 -1 表示）
            # 总共：6 + 1 + 6 + 1 = 14 维（标准 ALOHA 双机械臂动作空间）
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # 3. 创建模型变换（使用 ModelTransformFactory）
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        # 4. 组合所有配置
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """LIBERO 桌面机器人数据配置：使用 LeRobot 格式的 LIBERO 数据集。

    LIBERO 是一个桌面机器人基准测试平台（单机械臂 + 夹爪）。
    其数据包含 7 维动作空间（6 维位姿 + 1 维夹爪）。

    这个类的注释旨在作为"如何自定义数据集"的参考示例。
    如果你想将自己的数据集适配到 openpi，可以参考注释进行修改。

    关键设计考虑：
      LIBERO 数据集本身已经包含了增量动作（delta actions），
      但有些旧版本的 π₀ 检查点训练时使用了"双重增量"（extra delta）。
      因此提供了 extra_delta_transform 参数来控制是否额外再做一次增量转换。
    """

    # 如果为 True，在现有 delta actions 的基础上再做一次 delta 转换。
    # 这是为了兼容某些旧版 π₀ 检查点而保留的。
    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # ==================== 第1层：重排变换 ====================
        #
        # repack transform 只应用于从数据集读取的数据，推理时不使用。
        # 它让数据集中的键名与推理环境中的键名保持一致。
        #
        # 要适配你自己的数据集时：
        #   1. 先确定你的策略服务器（policy server）会传入什么键名
        #   2. 修改下面的映射，使数据集的键名匹配到推理环境的键名
        #
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # ==================== 第2层：数据变换 ====================
        #
        # data transforms 在训练和推理时都会应用。
        # 我们定义了模型输入（inputs）和模型输出（outputs）的变换。
        # 这些变换定义在 libero_policy.py 中，你可以查看那里的详细注释。
        #
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # LIBERO 数据集本身已经使用 delta actions，但一些旧版 π₀ 检查点
        # 用了额外的 delta 变换，所以这里按需开启。
        if self.extra_delta_transform:
            # 前 6 维（关节位置）做 delta 转换，第 7 维（夹爪）保持绝对值
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # ==================== 第3层：模型变换 ====================
        # 模型变换包括分词（tokenize prompt）等操作。
        # 适配你自己的数据集时，这里通常不需要修改。
        model_transforms = ModelTransformFactory()(model_config)

        # 组合所有变换并返回
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """DROID 数据配置：使用 RLDS 格式的 DROID 数据集。

    DROID（Distributed Robot Interaction and Dexterity）是一个大规模的
    机器人数据集。由于 DROID 数据集非常大（数百小时），使用标准的 LeRobot
    格式处理会非常低效。RLDS（Reinforcement Learning Dataset Standard）
    是 TensorFlow 生态中的一种高效数据格式，支持流式读取。

    什么时候用这个配置？
      - 训练/微调"完整"的 DROID 数据集（100+ 小时数据）
      - 需要高效的流式数据加载

    什么时候使用 LeRobotDROIDDataConfig？
      - 你自己的小型 DROID 数据集（<10 小时数据）
      - 已经转换为 LeRobot 格式的数据

    重要：RLDS 数据加载器要求 num_workers=0，因为它内部自己处理多进程。
    """

    rlds_data_dir: str | None = None  # RLDS 数据集目录路径
    action_space: droid_rlds_dataset.DroidActionSpace | None = None  # 动作空间类型（关节位置 vs 末端执行器位姿）

    # 数据集过滤/采样配置。可以传入一个字典路径，该字典映射片段 ID 到时间步范围。
    # 每个片段通过 f"{recording_folderpath}--{file_path}" 唯一标识，
    # 这两个字段都在 RLDS 片段的元数据中。
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = (
        droid_rlds_dataset.RLDSDataset(
            name="droid",
            version="1.0.1",
            weight=1.0,  # 采样权重（当使用多个数据集时控制混合比例）
            filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",  # 片段过滤配置
        ),
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # 重排变换：将 RLDS 数据集的字段名映射到 DroidInputs 期望的格式
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # 数据变换：使用 DROID 特定的输入/输出变换
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        # 如果动作空间是关节位置（绝对位置），需要转换为增量动作用于训练。
        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # 前 7 个关节做 delta 转换，第 8 维（夹爪）保持绝对值
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            datasets=self.datasets,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """DROID 数据配置（LeRobot 格式版本）：适用于小型自定义 DROID 数据集。

    当你有一个自己的 DROID 数据集（不超过数十小时），并已通过
    examples/droid/convert_droid_data_to_lerobot.py 转换为 LeRobot 格式时使用。

    与 RLDSDroidDataConfig 的区别：
      - 更简单的配置（不需要 rlds_data_dir）
      - 适用于小数据集
      - 使用标准的 PyTorch DataLoader（num_workers > 0）

    注意：这里假设动作已经是关节速度（joint velocity），所以不需要额外的 delta 变换。
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # 假设动作已经是关节速度，所以不需要额外的 delta 变换。
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    """训练配置：定义一次训练任务的所有超参数和配置。

    这是整个配置系统的核心类。一个 TrainConfig 实例就是对一个训练任务
    的完整描述，包括：用什么模型、在什么数据上训练、优化器参数、
    训练步数、检查点保存策略等。

    在 _CONFIGS 列表中注册的每个命名的 TrainConfig 都可以通过
    `get_config("name")` 获取，或者在命令行通过 `train.py name` 启动。

    重要字段说明：
      - name: 配置的唯一名称，也是其标识符
      - model: 模型架构和超参数（π₀ / π₀-FAST / π₀.₅）
      - data: 数据集和数据变换流水线配置
      - weight_loader: 预训练权重加载策略
      - freeze_filter: 微调时冻结哪些参数（LoRA 等）
    """

    # 配置的名称，必须唯一。用于在命令行和代码中引用此配置。
    name: tyro.conf.Suppress[str]

    # 项目名称（用于日志和实验追踪）。
    project_name: str = "openpi"

    # 实验名称。用于命名元数据目录和检查点目录。
    exp_name: str = tyro.MISSING

    # ---- 模型配置 ----
    # 定义模型架构。不同的模型有不同的配置类：
    #   - Pi0Config: π₀ 和 π₀.₅（流匹配架构）
    #   - Pi0FASTConfig: π₀-FAST（自回归架构）
    # 所有模型配置都继承自 BaseModelConfig，共享 action_dim、action_horizon、
    # max_token_len 等字段。
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # ---- 权重加载 ----
    # 权重加载器：在模型初始化后，可以选择从磁盘加载（部分）权重。
    # 例如：从预训练的基础模型检查点加载权重用于微调。
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # ---- PyTorch 训练相关 ----
    pytorch_weight_path: str | None = None  # PyTorch 权重路径（可选）
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"  # 训练精度

    # ---- 优化器配置 ----
    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99  # 指数移动平均（EMA）的衰减率，用于平滑模型参数

    # ---- 冻结参数 ----
    # 指定哪些权重应该被冻结（不更新）。用于：
    #   - 微调时保持预训练权重不变
    #   - LoRA 微调时只更新低秩适配器
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # ---- 数据配置 ----
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # ---- 路径配置 ----
    assets_base_dir: str = "./assets"  # 资产（归一化统计量等）的基础目录
    checkpoint_base_dir: str = "./checkpoints"  # 检查点的基础目录

    # ---- 训练超参数 ----
    seed: int = 42  # 随机种子
    batch_size: int = 32  # 全局批大小
    num_workers: int = 2  # 数据加载器的 worker 数量（增加可加速数据加载，但增加内存/CPU 使用）
    num_train_steps: int = 30_000  # 训练步数

    # ---- 日志和检查点 ----
    log_interval: int = 100  # 每隔多少步记录训练指标
    save_interval: int = 1000  # 每隔多少步保存检查点
    keep_period: int | None = 5000  # 每隔多少步的检查点保留（不会被删除）

    # ---- 训练控制 ----
    overwrite: bool = False  # 如果为 True，覆盖已有的检查点目录
    resume: bool = False  # 如果为 True，从最后一个检查点恢复训练

    # ---- 实验追踪 ----
    wandb_enabled: bool = True  # 是否启用 Weights & Biases 日志

    # ---- 策略服务 ----
    # 传递给策略服务器的元数据。例如重置位姿（reset_pose）等。
    policy_metadata: dict[str, Any] | None = None

    # ---- 分布式训练 ----
    # FSDP（Fully Sharded Data Parallelism）的设备数。如果大于 1，启用 FSDP
    # 并在指定数量的设备上进行模型分片。例如：总设备数为 4，fsdp_devices=2，
    # 则模型被分片到 2 个设备上，数据并行在 2 组设备之间进行。
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """返回此配置的资产目录路径。

        路径格式：{assets_base_dir}/{config_name}/
        例如：./assets/pi0_libero/
        """
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """返回此配置的检查点目录路径。

        路径格式：{checkpoint_base_dir}/{config_name}/{exp_name}/
        例如：./checkpoints/pi0_libero/my_experiment/

        注意：exp_name 必须设置，否则会报错。
        """
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """返回"可训练参数"的过滤器。

        逻辑：所有参数（nnx.Param）中，不属于 freeze_filter 的部分。
        即：trainable = All(Param, Not(freeze_filter))

        在训练循环中，只有通过这个过滤器的参数才会被优化器更新。
        """
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        """初始化后的验证逻辑。

        检查 resume 和 overwrite 不能同时为 True（互斥）。
        """
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# ========================================================================
#  命名配置注册表（_CONFIGS）
# ========================================================================
#
# 下面这个列表是 openpi 中所有预定义训练配置的注册中心。
# 每个配置都有一个唯一的 name，可以通过命令行或 get_config() 引用。
#
# 配置分为以下几类：
#   1. 推理配置（Inference）—— 使用预训练模型直接推理，不训练
#   2. 微调配置（Fine-tuning）—— 在特定数据集上微调预训练模型
#   3. 仿真配置（Sim）—— 在仿真环境中训练和测试
#   4. 调试配置（Debug）—— 快速验证代码正确性
#
# 使用 `get_config("name")` 获取配置，或 `train.py name --exp-name=xxx` 启动训练。
# ========================================================================
_CONFIGS = [
    # =========================================================
    #  ALOHA 推理配置（Inference Aloha configs）
    # =========================================================
    # 这些配置用于在 ALOHA 机器人上运行预训练模型进行推理。
    # 它们不从数据集加载数据，而是等待策略服务器传入实时观测数据。
    # 使用 SimpleDataConfig 的变体（没有 repo_id），意味着不会加载训练数据。
    #
    #  
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),  # 使用 Trossen 机器人的归一化统计量
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},  # 初始复位位姿
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    # 带特定任务的 ALOHA 推理配置（设置了默认提示文本）
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",  # 默认指令：叠毛巾
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",  # 打开保鲜盒，把食物放到盘子里
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    # =========================================================
    #  DROID 推理配置（Inference DROID configs）
    # =========================================================
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,  # 从数据集的 task 字段获取语言指令
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    # =========================================================
    #  LIBERO 微调配置（Fine-tuning Libero configs）
    # =========================================================
    # 这些配置是"微调示范配置"，展示了如何在自定义数据集上微调基础模型。
    # 如果你想在自己的数据集上微调，可以以此为例进行修改。
    #
    TrainConfig(
        # 修改名称以反映你的模型和数据集
        name="pi0_libero",
        # 模型架构——这里使用 π₀ 进行全量微调（full finetune）
        model=pi0_config.Pi0Config(),
        # 数据集配置——这里使用 LIBERO 数据集。
        # 对于你自己的数据集：
        #   1. 将 repo_id 改为你的数据集 ID
        #   2. 使用你为数据集定制的 DataConfig
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # 从 LeRobot 数据集的 task 字段中加载 prompt（语言指令）
                # 如果设置为 True，prompt 会在输入字典的 "prompt" 键中出现
                prompt_from_task=True,
            ),
            extra_delta_transform=True,  # 兼容旧版 π₀ 检查点
        ),
        # 从预训练基础模型加载权重，用于初始化模型参数
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # 其他训练超参数
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # LoRA 微调：使用低秩适配器（Low-Rank Adaptation），可大幅减少显存占用
        # paligemma_variant: 使用 2B 参数的 LoRA 版本
        # action_expert_variant: 使用 300M 参数的 LoRA 版本
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # 冻结过滤器：指定哪些参数在训练中保持冻结。
        # 模型配置中提供了便捷方法 get_freeze_filter()，它会返回适合该模型配置的默认冻结方案。
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,  # LoRA 微调时关闭 EMA
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # π₀-FAST 模型全量微调示例
        # action_dim: 动作空间维度（LIBERO 为 7 维）
        # action_horizon: 动作块长度（模型一次预测多少步未来动作）
        # max_token_len: 最大 token 长度（包括文本提示、状态和动作 token）
        #   太小可能截断序列末尾的 token，太大浪费显存（因为要填充到最大长度）
        #   经验法则：单臂机器人约 180，双臂机器人约 250
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # π₀-FAST 模型 LoRA 微调示例
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero",
        # π₀.₅ 模型全量微调示例（改进版流匹配架构）
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,  # π₀.₅ 不需要额外 delta 变换
        ),
        batch_size=256,  # 更大的批大小
        # 余弦退火学习率调度：前 10k 步预热，峰值 lr=5e-5
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),  # 梯度裁剪
        ema_decay=0.999,  # 更大的 EMA 衰减率
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
    # =========================================================
    #  ALOHA 微调配置（Fine-tuning Aloha configs）
    # =========================================================
    # 这些配置演示如何在自定义 ALOHA LeRobot 数据集上进行微调。
    # 关于如何转换和训练你自己的 ALOHA 数据集，详见 examples/aloha_real/README.md
    #
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",  # 拔开笔帽
            # 自定义重排变换：映射到 ALOHA 的摄像头配置（主摄 + 左腕 + 右腕）
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
    ),
    # =========================================================
    #  DROID 微调配置（Fine-tuning DROID configs）
    # =========================================================
    TrainConfig(
        # 在全量 DROID 数据集上微调 π₀-FAST-base
        # 使用 RLDS 数据加载以支持大规模数据集的训练
        name="pi0_fast_full_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            rlds_data_dir="<path_to_droid_rlds_dataset>",  # 替换为你的 DROID RLDS 数据集路径
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k 步应该足够，在 8x H100 上约需 2 天
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # 重要：RLDS 数据加载器要求 num_workers=0，内部自己处理多进程
    ),
    TrainConfig(
        # 在全量 DROID 数据集上微调 π₀.₅
        name="pi05_full_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # π₀.₅ 使用 32 维动作空间
            action_horizon=16,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            rlds_data_dir="/mnt/pi-data/kevin",  # 需要替换为你的数据集路径
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets/",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=10_000,
        num_workers=0,
    ),
    TrainConfig(
        # 在自定义（较小）的 DROID 数据集上微调 π₀.₅-DROID
        # 使用 LeRobot 格式（与其他微调示例相同）
        name="pi05_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            repo_id="your_hf_username/my_droid_dataset",  # 替换为你的自定义 DROID LeRobot 数据集 ID
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # 重要：微调时复用原始 DROID 的归一化统计量！
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),
    # =========================================================
    #  ALOHA 仿真配置（ALOHA Sim configs）
    # =========================================================
    # 演示如何在简单的仿真环境中训练。
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,  # 仿真中不使用 delta action
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    # =========================================================
    #  调试配置（Debugging configs）
    # =========================================================
    # 这些配置使用 FakeDataConfig（假数据）和 dummy 模型（极小的模型变体），
    # 可以快速验证训练代码是否能正常运行。
    # 通常用于：
    #   - CI/CD 自动测试
    #   - 在低资源环境下调试
    #   - 验证代码修改没有破坏基本功能
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,  # 调试时关闭 wandb
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        # 从之前的 debug 训练恢复权重
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    TrainConfig(
        name="dummy_debug",
        model=dummy_model.DummyModelConfig(),
        data=FakeDataConfig(),
        weight_loader=weight_loaders.NoOpWeightLoader(),  # 不加载任何预训练权重
        batch_size=2,
        num_train_steps=10,
        log_interval=1,
        save_interval=5,
        overwrite=True,
        exp_name="dummy_test",
        wandb_enabled=False,
    ),
    # RoboArena & PolaRiS 配置（其他平台的配置，单独文件中定义）
    *roboarena_config.get_roboarena_configs(),
    *polaris_config.get_polaris_configs(),
]

"""  使用 SimpleDataConfig 的变体（没有 repo_id），意味着不会加载训练数据。  pi0_droid配置使用了SimpleDataConfig，不加载训练数据，那么这个配置还有什么用呢？     
    
    答案就在注释里——**Inference DROID configs**（第 1085 行）。这几个配置的用途是 **推理（inference / serving）**，不是训练。

    ---

    ## 没有训练数据，有什么用？

    这类配置专门用在 `scripts/serve_policy.py` 等推理场景中，例如：

    ```bash
    uv run scripts/serve_policy.py policy:checkpoint \
        --policy.config=pi0_droid \
        --policy.dir=checkpoints/pi0_droid/xxx/20000
    ```

    它的作用：

    ### 1. 定义推理时的输入/输出变换

    虽然不加载训练数据，但 **机器人推理时也需要数据变换**：

    ```python
    data_transforms=lambda model: Group(
        inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],   # 将原始传感器数据转为 Observation
        outputs=[droid_policy.DroidOutputs()],                           # 将模型输出转为机器人指令
    ),
    ```

    DroidInputs 负责把摄像头图像、关节状态等原始传感器数据转换成模型能吃的 `Observation` 格式；DroidOutputs 负责把模型输出的动作反归一化、转换成机器人可执行的指令。

    ### 2. 加载归一化统计量

    ```python
    assets=AssetsConfig(asset_id="droid"),
    ```

    推理时需要加载预计算的归一化统计量（`norm_stats`）来做 **反归一化**——模型输出的是归一化后的动作，要转回真实值。

    ### 3. 定义模型结构

    ```python
    model=pi0_config.Pi0Config(action_horizon=10),
    ```

    告诉 serving 脚本：用什么模型架构加载检查点。

    ---

    ## 对比：训练配置 vs 推理配置

    |            | 训练配置（如 `pi0_libero`） | 推理配置（如 `pi0_droid`）           |
    | ---------- | --------------------------- | ------------------------------------ |
    | `repo_id`  | 有 → 用于加载训练数据       | 无 → 不加载训练数据                  |
    | 数据变换   | 在数据集样本上做预处理      | 在实时传感器输入上做预处理           |
    | 用途       | `scripts/train.py`          | `scripts/serve_policy.py`            |
    | norm_stats | 来自数据集或预计算          | 从 `AssetsConfig` 加载，用于反归一化 |

    ---

    ## 总结

    `pi0_droid` 这类配置本质上是一个 **模型加载和推理的标准模板**——告诉框架：
    - 用什么模型结构加载检查点
    - 从哪加载归一化统计量
    - 生产环境下的输入/输出格式是什么

    它不训练，但它定义了 **训练好的模型怎么在机器人上跑起来**。这是开环推理（open-loop inference）和策略部署（policy serving）时必不可少的配置。
"""

"""
## `repo_id` 是什么？

`repo_id` 是 **Hugging Face LeRobot 数据集的仓库 ID**，格式为 `组织名/数据集名`，比如：

- `"physical-intelligence/libero"`
- `"lerobot/aloha_sim_transfer_cube_human"`
- `"physical-intelligence/aloha_pen_uncap_diverse"`

它告诉框架**去哪里下载真实的训练数据**。

---

## 为什么没有 `repo_id` 就不加载真实数据？

直接从数据加载器追一下调用链：

### 训练时

`scripts/train.py` → `create_data_loader()`（第 514 行）→ `create_torch_dataset()`（第 202 行）

```python
# data_loader.py 第 219-221 行
repo_id = data_config.repo_id
if repo_id is None:
    raise ValueError("未设置数据集 repo_id。无法创建数据集。")
```

**`repo_id` 为 `None` 时直接报错**。训练脚本根本跑不下去。

### `repo_id` 的三个合法值

| `repo_id`                                      | 行为                                           |
| ---------------------------------------------- | ---------------------------------------------- |
| `None`                                         | 抛 ValueError，无法创建数据集                  |
| `"fake"`                                       | 返回 `FakeDataset`，生成随机假数据（用于调试） |
| 真实 ID（如 `"physical-intelligence/libero"`） | 从 Hugging Face 下载并加载真实 LeRobot 数据集  |

注意 `FakeDataConfig` 的实现（第 489 行）：

```python
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"           # ← 硬编码为 "fake"

    def create(self, assets_dirs, model_config):
        return DataConfig(repo_id=self.repo_id)  # 返回一个只有 repo_id="fake" 的极简配置
```

### 为什么推理配置不需要 `repo_id`

因为 **推理根本不走数据加载器**。看推理路径：

`scripts/serve_policy.py` → 只用了 `config.data.create()` 来获取**变换流水线**（`data_transforms` 和 `model_transforms`），从来不会调用 `create_torch_dataset()`。

推理配置只需要三样东西：

1. **模型架构**（`model`）—— 加载检查点
2. **输入/输出变换**（`data_transforms`）—— 把实时传感器数据转成模型输入
3. **归一化统计量**（`assets`）—— 反归一化模型输出的动作

这些都不需要 `repo_id`。

---

## 总结

```python
# SimpleDataConfig 的 create() 方法会调用 create_base_config()
# 而 create_base_config() 将 repo_id 传给 DataConfig：

repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
asset_id = self.assets.asset_id or repo_id
return DataConfig(
    repo_id=repo_id,       # None → 训练时会崩
    ...
)
```

- **`repo_id` 之于训练，相当于"数据源地址"**——告诉框架从 Hugging Face 的哪个仓库下载数据
- 推理配置里 `repo_id` 为 `None` 是**刻意为之**：因为推理用的数据来自真实的机器人传感器，不是从 Hugging Face 读的
- 如果试图用这样的配置训练，`create_torch_dataset()` 第一行就会抛出 `ValueError`，防止跑出"数据为空"的无声错误
"""

    

# 检查所有配置的名称是否唯一
if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    """命令行入口：返回一个通过命令行参数解析/覆盖后的 TrainConfig 实例。

    这个函数使用 tyro 库（类似 argparse 的更强大版本）处理命令行参数。
    它支持：
      - 通过名称选择配置（第一个位置参数）
      - 通过 --key=value 覆盖任何配置字段

    示例：
      train.py pi0_libero --exp-name=my_exp --batch_size=64
      train.py debug --wandb_enabled=False

    实现原理：
      tyro.extras.overridable_config_cli 接收一个配置名字典，
      自动生成命令行参数解析器，并支持在命令行中覆盖任何 dataclass 字段。
    """
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """按名称获取预定义的训练配置。

    这是从 _CONFIGS 列表中查找配置的标准接口。
    如果找不到对应的配置，会尝试提供最接近的名称建议。

    Args:
        config_name: 配置名称（对应 _CONFIGS 列表中某个配置的 name 字段）。

    Returns:
        对应的 TrainConfig 实例。

    Raises:
        ValueError: 如果找不到指定名称的配置。

    示例：
        >>> config = get_config("pi0_libero")
        >>> config.name
        'pi0_libero'
        >>> config.model.action_dim
        32
    """
    if config_name not in _CONFIGS_DICT:
        # 使用 difflib 找到最接近的名称，提供友好的错误提示
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]

# ===========================================================model=
#  src/openpi/training/data_loader.py — 数据加载模块
#
#  本模块负责将原始数据集（LeRobot / RLDS）转换为训练所需的
#  标准化 batch 数据。主要职责：
#    1. 从 Hugging Face LeRobot 数据集或本地 RLDS 目录加载数据
#    2. 应用数据变换（重映射键名、归一化、tokenize 等）
#    3. 将数据分片到各设备（JAX FSDP 分布式训练）
#    4. 以无限循环方式提供 batch（支持重启后继续训练）
#
#  支持两种数据后端：
#    - LeRobot (Torch DataLoader)：适用于小到中型数据集
#    - RLDS (自定义加载器)：适用于大规模 DROID 数据集
# ============================================================

import logging  # 日志
import multiprocessing  # 多进程数据加载
import os  # 环境变量操作
import typing  # typing 模块的 .cast()
from collections.abc import Iterator, Sequence  # 集合类型抽象
from typing import Literal, Protocol, SupportsIndex, TypeVar  # 类型标注

# --- JAX ---
import jax
import jax.numpy as jnp

# --- LeRobot（Hugging Face 上的机器人数据集格式）---
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

import numpy as np  # 标准 NumPy
import torch  # PyTorch（用于 DataLoader）

# --- openpi 内部模块 ---
import openpi.models.model as _model  # Observation / Actions 数据模型
import openpi.training.config as _config  # 训练配置
from openpi.training.droid_rlds_dataset import DroidRldsDataset  # DROID RLDS 数据集
import openpi.transforms as _transforms  # 数据变换（归一化、tokenize 等）

# 泛型类型变量，用于协变协议（covariant）
T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """数据集抽象协议 —— 支持随机访问（通过索引取样本）。

    这是典型的 PyTorch Dataset 风格接口：
    只需实现 __getitem__（按索引取样本）和 __len__（返回总样本数）。
    """

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("子类需实现 __getitem__ 方法。")

    def __len__(self) -> int:
        raise NotImplementedError("子类需实现 __len__ 方法。")


class IterableDataset(Protocol[T_co]):
    """可迭代数据集抽象协议 —— 只能顺序迭代，不可随机访问。

    适用于 RLDS 等无法通过索引随机访问的大规模数据集。
    """

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("子类需实现 __iter__ 方法。")

    def __len__(self) -> int:
        raise NotImplementedError("子类需实现 __len__ 方法。")


class DataLoader(Protocol[T_co]):
    """数据加载器抽象协议。

    任何 DataLoader 必须提供：
      - data_config()：返回数据配置（用于恢复训练时重建数据集状态）
      - __iter__()：迭代产生 batch 数据
    """

    def data_config(self) -> _config.DataConfig:
        """返回此数据加载器对应的数据配置。"""
        raise NotImplementedError("子类需实现 data_config 方法。")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("子类需实现 __iter__ 方法。")


class TransformedDataset(Dataset[T_co]):
    """对随机访问数据集（Dataset）应用一系列数据变换的包装器。

    用法：将原始数据集包装起来，每次 __getitem__ 时自动执行 transform 链。
    典型变换链：键名重映射 → 数据归一化 → 模型输入格式化。
    """

    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        # compose 将多个变换函数按顺序组合成一个函数（从左到右执行）
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        # 先取原始样本，再应用变换链
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    """对可迭代数据集（IterableDataset）应用数据变换的包装器。

    与 TransformedDataset 的区别：
      - 输入是 IterableDataset（不支持随机访问）
      - 支持 is_batched 模式：当数据源已返回 batch（而不是单样本）时，
        需要先将 batch 拆分为单样本，逐个变换后再重新拼回 batch。
    """

    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched  # 数据源是否已返回 batch（而非单样本）

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # ---- batch 模式：拆开 → 逐个变换 → 重新拼接 ----
                # 数据变换（如 Normalize）是为单样本设计的，所以需要:
                #   1. 从 batch 中逐个取出每个样本
                #   2. 对每个样本应用变换
                #   3. 将变换后的样本重新堆叠为 batch

                # 获取 batch size（从某个字段的第一个维度）
                batch_size = next(v.shape[0] for v in sample.values())

                # 1) 把 batch 沿着第 0 维拆成独立样本
                #    jax.tree.map(lambda x: x[i], sample) 对 sample 中每个数组取第 i 个元素
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]

                # 2) 逐个样本应用变换
                transformed = [self._transform(s) for s in individual_samples]

                # 3) 用 np.stack 将独立样本重新拼成 batch（沿第 0 维堆叠）
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                # ---- 单样本模式：直接应用变换 ----
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    """假数据集 —— 用于快速调试训练流程，无需真实数据。

    随机生成符合模型输入规范（inputs_spec()）的假数据：
      - float32 字段 → 均匀分布 [-1, 1]
      - int32 字段 → 整数 [0, 2048)
      - 其他类型 → 全零

    与 debug 配置（FakeDataConfig）配合使用，可以在几秒内启动训练以
    验证模型能否正常跑通，而不必等待真实数据集下载和预处理。
    """

    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        # 从模型配置中获取输入规范（形状和 dtype）
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        # 每个样本用其索引作为随机种子，保证确定的"随机"数据
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            """根据 spec 生成一个随机数组。"""
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # spec.shape[0] 是 batch 维度，取 [1:] 去掉它得到单样本的形状
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        # 为观察和动作分别生成假数据
        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        # 以 dict 格式返回（与真实数据格式保持一致）
        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """创建一个基于 PyTorch / LeRobot 的数据集对象。

    LeRobot 是 Hugging Face 上的机器人数据集格式，支持通过 repo_id 直接下载。
    此函数还支持特殊的 "fake" repo_id，返回 FakeDataset 用于调试。

    参数：
      data_config:    数据配置（指定数据集 repo_id、动作键等）
      action_horizon: 动作预测的步数（用于生成 delta_timestamps，让数据集
                      为每一步都返回对应的动作标签）
      model_config:   模型配置（用于 FakeDataset 生成适当形状的假数据）

    返回：
      一个 Dataset 对象，可通过索引获取原始样本（尚未归一化）
    """
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("未设置数据集 repo_id。无法创建数据集。")
    if repo_id == "fake":  # 特殊标记：使用假数据
        return FakeDataset(model_config, num_samples=1024)

    # ---- 正常流程：加载 LeRobot 数据集 ----
    # 先获取数据集元信息（如帧率 fps、任务列表等）
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)

    # 创建 LeRobot 数据集实例
    # delta_timestamps 指定每个样本需要返回哪些时间步的动作
    # 例如 action_horizon=16 且 fps=10，则 delta_timestamps = [0.0, 0.1, 0.2, ..., 1.5]（秒）
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)]
            for key in data_config.action_sequence_keys  # 可能多个动作键（如 "action", "state"）
        },
    )

    # 如果配置要求从任务描述生成 prompt，则包装一个变换层
    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


"""
    这是一个非常好的问题！确实是 Java 思维和 Python 思维的一个关键差异点。

## 简短答案

**在 Java 中不行，但在 Python 中可以**——因为这里的 `Dataset` 不是普通类，而是 **`Protocol`**（协议/结构类型），Python 的 `Protocol` 用的是**鸭子类型（Duck Typing）**而不是**名义类型（Nominal Typing）**。

---

## 核心原因：Protocol vs 抽象类/接口

看代码关键处：

```python
# 第 43 行
class Dataset(Protocol[T_co]):  # ← 注意：Protocol，不是普通的 class！
    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError(...)

    def __len__(self) -> int:
        raise NotImplementedError(...)
```

### Java 中的接口（名义类型）

```java
// Dataset.java
interface Dataset<T> {
    T get(int index);
    int size();
}

class LeRobotDataset implements Dataset<Something> {  // ← 必须显式 implements
    ...
}

Dataset dataset = new LeRobotDataset();  // √ 因为 LeRobotDataset implements Dataset
```

Java 检查的是：**"你声明了 `implements Dataset` 吗？"**——这是**名义类型系统**。没有继承/实现关系就不行。

### Python 的 Protocol（结构类型）

```python
class Dataset(Protocol):
    def __getitem__(self, index) -> T_co: ...
    def __len__(self) -> int: ...

# LeRobotDataset 没有显式继承 Dataset，但它有 __getitem__ 和 __len__
# → Protocol 说："结构匹配就行！"
dataset: Dataset = lerobot_dataset.LeRobotDataset(...)  # √ 不需要继承
```

Python `Protocol` 检查的是：**"你有 `__getitem__` 和 `__len__` 方法吗？"**——这是**结构类型系统**。

---

## 实际验证：LeRobotDataset 的结构

```python
# 从 PyTorch 查阅
class LeRobotDataset(torch.utils.data.Dataset):
    def __getitem__(self, idx): ...   # √ 有这个
    def __len__(self): ...            # √ 有这个

# 而 torch.utils.data.Dataset 也是个 HasGetItem + HasLen 的结构
```

因为 `LeRobotDataset` 有 `__getitem__` 和 `__len__`，**Python 类型检查器（mypy/pyright）认为它的结构符合 `Dataset` 协议**。

---
@src/openpi/training/data_loader.py 中，create_torch_dataset函数中，返回Dataset类型，这个Dataset是在本文件中定义的，返回的lerobot_dataset.LeRobotDataset是继承的torch中的D 
  ataset。我之前是搞Java的，我记得这是不行，对吗？如果不行，这里可以吗？  
  
## 用 Java 来类比 Python 的 Protocol

想象 Java 有这样一个东西（实际没有）：

```java
// ⚠️ 这是伪代码，Java 没有这个
structural interface Dataset<T> {  // ← imaginary "structural interface"
    T get(int index);
    int size();
}

class LeRobotDataset {  // ← 没有 implements Dataset，但方法签名匹配
    public Something get(int index) { ... }
    public int size() { ... }
}

// ⚠️ 结构类型系统下这合法，但 Java 不支持
Dataset<Something> ds = new LeRobotDataset();  // 结构匹配，通过！
```

**Java 缺少的就是 `structural interface` 这个概念**。最接近的是 C++ 的模板（template）鸭子类型：

```cpp
template<typename D>
void process(D& dataset) {
    auto item = dataset[0];  // 不检查 D 的继承关系，只看有没有 operator[]
    int n = dataset.size();  // 只看有没有 size()
}
```

---

## 文中的 `FakeDataset` 为什么同时继承 `Dataset`？

```python
class FakeDataset(Dataset):  # ← 名义上也继承了
    def __getitem__(self, index): ...
    def __len__(self): ...
```

`FakeDataset` 同时做了两件事：
1. **名义上**继承了 `Dataset`（Java 风格）
2. **结构上**也匹配 `__getitem__` + `__len__`（Python 风格）

而 `LeRobotDataset` 只做了第 2 件。但因为有 `Protocol`，两者都可被当作 `Dataset` 返回。

---

## 总结：Python vs Java 类型系统

| 特性         | Java                          | Python + Protocol                       |
| ------------ | ----------------------------- | --------------------------------------- |
| 类型系统     | **名义类型**（必须声明关系）  | **结构类型**（只看方法签名）            |
| 可以这样吗？ | ❌                            | ✅                                      |
| 核心检查     | "你是不是 `Dataset` 的子孙？" | "你有没有 `__getitem__` 和 `__len__`？" |
| 库的集成     | 要为每个库写适配器/wrapper    | 方法签名对上了就能直接用                |

**所以这里确实行得通**——Python 的 `Protocol` 不要求返回的对象有继承关系，只要求它有正确的方法签名。这是 Java 不具备的特性。
"""


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    """创建一个基于 RLDS 格式的数据集（用于 DROID 等大规模数据集）。

    RLDS (Reinforcement Learning Dataset Standard) 是 TensorFlow 生态的
    数据集格式。与 LeRobot 不同，RLDS 数据集通常是大规模的（数 TB），
    因此采用不同的加载策略（批量预取、流式读取）。

    注意：目前仅支持 DROID 数据集的 RLDS 格式。

    参数：
      data_config:    数据配置（RLDS 数据目录、数据集名称列表等）
      action_horizon: 动作预测步数
      batch_size:     批量大小
      shuffle:        是否打乱数据
    """
    # 目前只支持 DROID 数据集
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,  # RLDS 数据目录
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,  # 动作块大小
        action_space=data_config.action_space,  # 动作空间定义
        datasets=data_config.datasets,  # 要加载的数据集名称列表
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """对随机访问数据集（Dataset）应用标准变换流水线。

    变换链顺序（从左到右依次执行）：
      1. repack_transforms  — 键名重映射（将数据集的本土化键名映射为统一键名）
      2. data_transforms    — 数据级变换（如 DeltaActions 转换动作类型）
      3. Normalize          — 归一化（z-score 或分位数归一化）
      4. model_transforms   — 模型级变换（如 TokenizePrompt 将文本转为 token IDs）

    参数：
      dataset:       原始数据集
      data_config:   数据配置（包含各类变换的定义）
      skip_norm_stats: 是否跳过归一化（用于调试或推理时）

    返回：
      应用了完整变换链的 TransformedDataset
    """
    norm_stats = {}  # 归一化统计量（均值和标准差，或分位数）
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "未找到归一化统计量。请先运行脚本计算：\n"
                "  uv run scripts/compute_norm_stats.py --config-name=<your-config>"
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,  # 第一步：键名重映射
            *data_config.data_transforms.inputs,  # 第二步：数据变换
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),  # 第三步：归一化
            *data_config.model_transforms.inputs,  # 第四步：模型输入变换
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """对可迭代数据集（IterableDataset）应用标准变换流水线。

    与 transform_dataset 功能相同，但适用于 IterableDataset（RLDS 等流式数据集）。
    额外支持 is_batched 参数：如果数据源已经返回 batch，则在变换时先拆开再重组。

    参数：
      dataset:        原始可迭代数据集
      data_config:    数据配置
      skip_norm_stats: 是否跳过归一化
      is_batched:     数据源是否已 batch 化（RLDS 数据加载器通常直接返回 batch）
    """
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "未找到归一化统计量。请先运行脚本计算：\n"
                "  uv run scripts/compute_norm_stats.py --config-name=<your-config>"
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """创建数据加载器的统一入口（调度到 Torch 或 RLDS 实现）。

    这是训练脚本调用的主函数，根据数据配置自动选择：
      - 如果 data_config 指定了 rlds_data_dir → 创建 RLDS 数据加载器
      - 否则 → 创建基于 LeRobot 的 Torch 数据加载器

    参数：
      config:          训练配置（包含数据配置、模型配置、batch_size 等）
      sharding:        JAX FSDP 数据分片方案（将数据分布到各设备）
      shuffle:         是否打乱数据
      num_batches:     限制返回的 batch 数量（None=无限迭代）
      skip_norm_stats: 是否跳过归一化（调试用）
      framework:       运行框架（"jax" 或 "pytorch"），影响数据如何转换为相应框架的张量
    """
    # 从配置工厂创建具体的数据配置实例
    # 这步会解析 assets_dirs 和模型配置，生成完整的 DataConfig
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    # 根据数据集类型选择分支
    if data_config.rlds_data_dir is not None:
        # ---- RLDS 分支（DROID 等大规模数据集）----
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    # ---- LeRobot / Torch 分支（标准数据集）----
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """创建基于 PyTorch DataLoader 的数据加载器（LeRobot 数据集）。

    工作流程：
      1. 调用 create_torch_dataset() 创建 LeRobot 数据集
      2. 调用 transform_dataset() 应用变换流水线
      3. 创建 TorchDataLoader 进行 batch 采样和多进程加载
      4. 用 DataLoaderImpl 包装，使其符合 openpi 的 DataLoader 协议

    参数：
      data_config:    数据配置
      model_config:   模型配置（用于 FakeDataset / 输入规范）
      action_horizon: 动作预测步数
      batch_size:     全局 batch size
      sharding:       JAX 分片方案（将数据分到各设备）
      skip_norm_stats: 是否跳过归一化
      shuffle:        是否打乱
      num_batches:    返回的 batch 数上限（None=无限）
      num_workers:    PyTorch DataLoader 的工作进程数（0=在主进程加载）
      seed:           随机种子
      framework:      目标框架（"jax" 或 "pytorch"）
    """
    # 第一步：创建原始（未变换的）数据集
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    # 第二步：应用变换流水线（重映射 → 数据变换 → 归一化 → 模型变换）
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # 第三步：配置分布式采样器和 local batch size
    # 在分布式训练中，每个进程只处理全局 batch 的一部分
    sampler = None
    if framework == "pytorch":
        # PyTorch DDP 分布式训练
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        # JAX 分布式：根据进程数量划分 batch
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")

    # 第四步：创建底层 Torch DataLoader
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,  # PyTorch 不需要 JAX 分片
        shuffle=(sampler is None and shuffle),  # 如果用了 sampler，就不在 DataLoader 层面 shuffle
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    # 第五步：包装为 DataLoaderImpl（增加 data_config() 方法和 Observation/Actions 解包）
    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """创建基于 RLDS 的数据加载器（用于 DROID 等大规模数据集）。

    与 Torch 数据加载器不同，RLDS 数据加载器的 batch 操作在数据集内部完成
    （DroidRldsDataset 自身负责批处理），所以 is_batched=True。

    注意：此加载器需要额外依赖，参见 examples/droid/README_train.md。

    参数：
      data_config:    数据配置
      action_horizon: 动作预测步数
      batch_size:     批量大小
      sharding:       JAX 分片方案
      skip_norm_stats: 是否跳过归一化
      shuffle:        是否打乱
      num_batches:    返回的 batch 数上限（None=无限）
      framework:      目标框架（目前仅支持 JAX）
    """
    if framework == "pytorch":
        raise NotImplementedError("RLDS 数据加载器暂不支持 PyTorch。")

    # 第一步：创建 RLDS 数据集（已返回 batch 数据）
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)

    # 第二步：应用变换流水线（因为数据源已 batch 化，设置 is_batched=True）
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    # 第三步：创建 RLDSDataLoader（处理分片和无限迭代）
    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    # 第四步：包装为 DataLoaderImpl
    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """基于 PyTorch DataLoader 的数据加载器实现。

    在 PyTorch 的 DataLoader 之上封装了：
      1. 无限循环迭代（数据集耗尽后自动重新开始）
      2. 可选的 batch 数上限
      3. JAX 分片支持（将 numpy/torch 数组转为 JAX 分片数组）

    对于 JAX 训练来说，PyTorch DataLoader 只在主进程加载数据，
    然后通过 jax.make_array_from_process_local_data 将数据广播到所有设备。
    """

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """初始化 TorchDataLoader。

        参数：
          dataset:          PyTorch Dataset 对象
          local_batch_size: 每个进程的本地 batch 大小
          sharding:         JAX 分片方案
          shuffle:          是否打乱数据
          sampler:          自定义采样器（分布式训练时使用 DistributedSampler）
          num_batches:      限制返回的 batch 数（None=无限）
          num_workers:      DataLoader 的工作进程数
          seed:             随机种子
          framework:        "jax" 或 "pytorch"
        """
        if jax.process_count() > 1:
            raise NotImplementedError("不支持多进程数据加载。")

        if len(dataset) < local_batch_size:
            raise ValueError(f"本地 batch 大小 ({local_batch_size}) 超过了数据集大小 ({len(dataset)})。")

        # ---- 分片方案 ----
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # JAX 默认使用数据并行分片（将所有设备视为一个 data-axis）
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        # ---- PyTorch DataLoader 配置 ----
        mp_context = None
        if num_workers > 0:
            # 多进程时使用 spawn 启动方式（避免 CUDA fork 问题）
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)

        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # 使用 sampler 时不在 DataLoader 层面 shuffle
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,  # 工作进程在 epoch 间保持存活
            collate_fn=_collate_fn,  # 自定义 batch 组装函数
            worker_init_fn=_worker_init_fn,  # 工作进程初始化（禁止 JAX 预占 GPU 显存）
            drop_last=True,  # 丢弃最后一个不完整的 batch
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        """暴露底层的 PyTorch DataLoader（供外部直接访问）。"""
        return self._data_loader

    def __iter__(self):
        """无限迭代产生 batch。

        行为控制：
          - 如果未设置 num_batches：永远迭代下去（用于主训练循环）
          - 如果设置了 num_batches：返回指定数量后停止
          - 数据集耗尽后自动重启（重新创建迭代器从头开始）
        """
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                # 检查是否达到上限
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    # 数据集已耗尽，跳出内层循环重新创建迭代器
                    break
                num_items += 1

                # 根据框架转换数据格式
                if self._sharding is not None:
                    # JAX 模式：将 numpy 数组转为分片 JAX 数组
                    # make_array_from_process_local_data 将主进程的数据
                    # 分片复制到各设备上
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    # PyTorch 模式：转为 torch 张量
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """PyTorch DataLoader 的自定义 batch 组装函数。

    PyTorch DataLoader 默认使用 default_collate，它要求所有元素都是 torch 张量。
    但我们的数据集可能包含 JAX 数组。此函数先将所有元素转为 numpy 数组，
    再用 np.stack 在第 0 维堆叠，形成 batch。

    参数：
      items: 一个 list，每个元素是一个由 __getitem__ 返回的 dict
             例如 [{"image": arr0, "action": act0}, {"image": arr1, "action": act1}, ...]

    返回：
      一个 dict，每个字段的值是堆叠后的 batch 数组 (B, ...)
    """
    # jax.tree.map(*items) 相当于对 items 中每个 dict 的对应键进行操作
    # lambda *xs 收集所有样本中同一键的值，然后堆叠
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """多进程数据加载工作进程的初始化函数。

    重要：在工作进程中禁止 JAX 预分配 GPU 显存。
    因为工作进程只做数据加载和预处理（CPU 操作），不需要 GPU 显存。
    如果不禁止，每个工作进程都会尝试独占 GPU，导致显存溢出（OOM）。

    注意：此函数在工作进程中调用时，JAX 已经完成了导入（import），
    因此无法通过此方式切换 JAX 后端（CPU/GPU/TPU）。
    """
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # 不预分配显存
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"  # 按需分配


class RLDSDataLoader:
    """RLDS 数据加载器的轻量封装。

    与 TorchDataLoader 不同，RLDS 数据集的 batch 操作已经在
    DroidRldsDataset 内部完成了（它返回已经是 batch 格式的数据）。
    此包装器主要负责：
      1. 处理 JAX 分片（将数据分布到各设备）
      2. 提供无限循环迭代
      3. 可选的 batch 数上限
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("不支持多进程数据加载。")

        if sharding is None:
            # 默认使用数据并行分片
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        """无限迭代产生 batch。

        与 TorchDataLoader.__iter__ 逻辑相同：
        - 支持 num_batches 上限
        - 数据集耗尽后自动重启
        - 将数据转为 JAX 分片数组
        """
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    # 数据集已耗尽，重启迭代
                    break
                num_items += 1
                # 将数据转为 JAX 分片数组，分布到各设备
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    """DataLoader 协议的最终实现 —— 统一包装器。

    将 TorchDataLoader 或 RLDSDataLoader 包装为 openpi 训练循环
    可用的标准接口，提供两个核心功能：
      1. data_config() → 返回数据配置（用于恢复训练时重建状态）
      2. __iter__() → 将 batch dict 解包为 (Observation, Actions) 元组
    """

    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        # 将底层加载器产生的 dict 拆分为 (Observation, Actions) 元组
        # Observation 是模型的标准输入格式，actions 是标签
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]     # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

"""
好问题，完整链路非常清楚，我给你画出来：

## 数据类型的完整转换链（JAX 训练模式）

```
FakeDataset.__getitem__()
   │  JAX arrays ← jax.random.uniform / jax.random.randint
   ▼
PyTorch DataLoader (call _collate_fn)
   │
   │  _collate_fn(items):  ← 你问的这里
   │    np.asarray(x)   → JAX → numpy ✅
   │    np.stack(...)   → batch
   │  Returns: numpy arrays
   ▼
TorchDataLoader.__iter__()
   │
   │  if sharding is not None:     ← JAX 模式（train.py 调用时）
   │    jax.make_array_from_process_local_data(sharding, x)  → numpy → JAX ✅
   │    Returns: JAX sharded arrays (分布在 GPU 上)
   │
   │  if sharding is None:         ← PyTorch 模式（train_pytorch.py 调用时）
   │    torch.as_tensor(x)         → numpy → torch tensor
   │    Returns: torch tensors
   ▼
DataLoaderImpl.__iter__()
   │  Observation.from_dict(batch) → 结构化 Observation
   │  yield (Observation, Actions)
   ▼
train_step()  ← JAX jit 编译的函数，接收 JAX arrays
```

## 一句话回答

**会。** `_collate_fn` 把 JAX → numpy 只是为了给 PyTorch DataLoader 做 batch 堆叠（`np.stack`）。然后在 `TorchDataLoader.__iter__()` 中：

- **JAX 训练** → `jax.make_array_from_process_local_data(sharding, x)` 把 numpy **转回 JAX sharded array**，并且直接分片到 GPU 上
- **PyTorch 训练** → `torch.as_tensor(x)` 把 numpy 转成 **torch tensor**

三个框架间的来回转换，**只是为了用 PyTorch 的 DataLoader 做多进程数据加载和 batch 组装**（PyTorch 的 DataLoader 在 `num_workers>0` 时效果最好）。真正训练时最终还是用各自框架的原生类型。

## 终极效果

所以你在 `train.py` 的 `train_step()` 里看到的 `batch` 是：
```
Observation(
    images={"base_0_rgb": jax.Array(2, 224, 224, 3) on GPU},  ← 已经是 JAX 分片数组
    state: jax.Array(2, 32) on GPU,
    ...
)
Actions: jax.Array(2, 50, 32) on GPU
```

数据已经躺在 GPU 显存里了，`jax.jit` 编译的训练函数直接拿过来就做前向传播。
"""








""" pi0模型的数据处理中有很多图像数据的掩码，但是lerobot数据集中有这些信息吗？我怎么感觉这里的掩码在实际训练推理中并没有什么用呢

● 好问题，让我深入看看 pi0 模型里的图像掩码是怎么处理的，以及 LeRobot 数据集是否真的有这些信息。

● Agent(Investigate image mask usage in pi0)
  ⎿  Backgrounded agent (↓ to manage · ctrl+o to expand)

  Thought for 1s (ctrl+o to expand)

● 让我也在关键文件里直接看一些核心代码，更全面地了解情况。

  Thought for 1s, read 2 files (ctrl+o to expand)

● Agent(Search LeRobot data config and mask handling)
  ⎿  Backgrounded agent (↓ to manage · ctrl+o to expand)

  Thought for 1s (ctrl+o to expand)

● 让我也在几个关键文件中直接查看。

  Thought for 39s, searched for 9 patterns, read 4 files (ctrl+o to expand)

● 好，现在我有了完整的画面。让我来个全面的回答。
     
  ---
  你的直觉是对的——但又不完全对
     
  你说得对：LeRobot 数据集本身确实没有 image_mask 字段。你在训练代码的 RepackTransform 里看不到任何 "image_mask" 的映射，是因为它就不来自 LeRobot 数据。

  但你说的"没什么用"——不完全对。让我给你画全貌。

  image_mask 从哪里来？

  两个来源，看场景：

  1. 训练时 → 由 DataTransform 中的平台输入变换 合成

  以你的 Tron2 配置为例，数据流是：

  LeRobotDataset (原始数据)
    │  {"observation.images.cam_high": ..., "observation.state": ..., "action": ...}
    ▼
  RepackTransform (重映射)
    │  {"image": {"cam_high": ...}, "state": ..., "actions": ...}  ← 没有 image_mask！
    ▼
  Tron2Inputs (数据变换第2层，你配置的)    ←  关键在这里！！！
    │  ★★★ 在这里合成 image_mask ★★★
    │  代码在 tron2_policy.py:140-157:
    │    image_masks["base_0_rgb"] = np.True_   // 主摄像头总是有效
    │    image_masks["left_wrist_0_rgb"] =  np.True_ if camera exists else np.False_  // 缺失的标记 False
    ▼
  Normalize → TokenizePrompt → PadStatesAndActions → Observation.from_dict()

  关键代码就在你的 tron2_policy.py:140-157：

  image_masks = {"base_0_rgb": np.True_}  # 基础图像始终有效
  for dest, source in extra_image_names.items():
      if source in in_images:
          images[dest] = in_images[source]
          image_masks[dest] = np.True_   # 有真实图像 → 有效
      else:
          images[dest] = np.zeros_like(base_image)  # 缺失 → 黑色填充
          image_masks[dest] = np.False_  # 标记为无效

  2. 推理时 → 由策略 Policy 中的 infer() 合成

  同样道理，aloha_policy.py, libero_policy.py, droid_policy.py 的 __call__ 里都在手动构建 image_mask。

  ---
  那 image_mask 到底有没有用？
  
  你的场景下大概率没用，但它在整个框架里有特定的设计用途。

  看 preprocess_observation() 的兜底逻辑（model.py:414-419）：

  if key not in observation.image_masks:
      out_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool)  # 没提供 → 默认全有效

  这意味着：如果所有摄像头都存在，image_mask 全部是 True，它在模型中确实什么也不干。 你删掉它，每个 batch 的 mask 全为 True，loss 不会有任何变化。

  那它为什么存在？

  为了一个具体的场景：不同机器人平台的摄像头数量不同。

  ┌────────┬──────────────┬──────────┬────────────────────────────────┐
  │  平台  │ 真实摄像头数 │ 模型期望 │            处理方式            │
  ├────────┼──────────────┼──────────┼────────────────────────────────┤
  │ ALOHA  │ 3            │ 3        │ 全部 True，mask 冗余           │
  ├────────┼──────────────┼──────────┼────────────────────────────────┤
  │ DROID  │ 2            │ 3        │ 第 3 个用黑图填充 + mask=False │
  ├────────┼──────────────┼──────────┼────────────────────────────────┤
  │ LIBERO │ 2            │ 3        │ 第 3 个用黑图填充 + mask=False │
  ├────────┼──────────────┼──────────┼────────────────────────────────┤
  │ Tron2  │ 2~4 可变     │ 3        │ 缺失的 mask=False              │
  └────────┴──────────────┴──────────┴────────────────────────────────┘

  核心机制在 embed_prefix()（pi0.py:436-444）：

  for name in obs.images:
      image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
      input_mask.append(
          einops.repeat(
              obs.image_masks[name],     # ← 这里用到了！
              "b -> b s",
              s=image_tokens.shape[1],
          )
      )

  当 image_mask=False 时，该摄像头产生的所有 token 的 input_mask 被设为 False，在 attention 计算中这些 token 完全被屏蔽（softmax 前加
  -inf），相当于这个摄像头"不存在"。

  这样设计的好处：
  - 你不需要因为某个平台少了一个摄像头就重新训练模型
  - 一个在 3 摄像头数据上训好的模型，拿给 2 摄像头的机器人用，只要把缺失的那个位置填黑图 + mask=False 就行
  - 训练时即使某个样本有一个摄像头坏了/掉了帧，也能继续参与训练，不用扔掉整个样本

  所以回到你的问题

  对于 LeRobot 数据集 + 所有摄像头都正常的场景：
  - ✅ 你说得对——image_mask 确实不起作用（全是 True）
  - ✅ 在 transform pipeline 中它是被"合成"出来的，不是 LeRobot 自带的
  - ❌ 但说"没什么用"不完全准确——它在其他平台（DROID、LIBERO）和摄像头偶尔断线的场景下是必要的

  如果你确定你的数据永远有全部摄像头，并且在 preprocess_observation() 中兜底默认就是全 True，那理论上是可以不传 image_mask
  的。但框架为了统一处理所有平台，强制要求这个字段存在。

✻ Waiting for 3 background agents to finish

● Agent "Search LeRobot data config and mask handling" finished · 37s

  Thought for 2s (ctrl+o to expand)

● 刚才的搜索任务完成了，结果与我的分析完全一致——简单总结一下补充发现：
     
  确认：LeRobot 数据集没有任何 image_mask 字段。

  image_mask 的完整生命周期是：

  1. 数据层（LeRobot） → 没有 image_mask
  2. 平台 Policy 层（你在 tron2_policy.py 里的 Tron2Inputs） → 合成 image_mask：
    - 检查每个摄像头是否存在 → 存在就 True，缺失填黑图设 False
  3. 模型前向（pi0.py:embed_prefix） → 用 image_masks[name] 把整个摄像头的 token 全部屏蔽或保留
  4. 兜底（model.py:preprocess_observation） → 如果某个摄像头根本没对应的 mask 键，自动全 True

  另外发现一个有趣的细节：π₀-FAST 模型对所有摄像头（包括填充的黑图）都设 mask=True，因为 FAST 架构中 padding 图像有特殊的处理逻辑，不能直接屏蔽
  token。这说明不同类型的模型对 mask 的处理方式不同。
              obs.image_masks[name],     # ← 这里用到了！
              "b -> b s",
              s=image_tokens.shape[1],
          )
      )

  当 image_mask=False 时，该摄像头产生的所有 token 的 input_mask 被设为 False，在 attention 计算中这些 token 完全被屏蔽（softmax 前加 -inf），相当于这个摄像头"不存在"。

  这样设计的好处：
  - 你不需要因为某个平台少了一个摄像头就重新训练模型
  - 一个在 3 摄像头数据上训好的模型，拿给 2 摄像头的机器人用，只要把缺失的那个位置填黑图 + mask=False 就行
  - 训练时即使某个样本有一个摄像头坏了/掉了帧，也能继续参与训练，不用扔掉整个样本

  所以回到你的问题

  对于 LeRobot 数据集 + 所有摄像头都正常的场景：
  - ✅ 你说得对——image_mask 确实不起作用（全是 True）
  - ✅ 在 transform pipeline 中它是被"合成"出来的，不是 LeRobot 自带的
  - ❌ 但说"没什么用"不完全准确——它在其他平台（DROID、LIBERO）和摄像头偶尔断线的场景下是必要的

  如果你确定你的数据永远有全部摄像头，并且在 preprocess_observation() 中兜底默认就是全 True，那理论上是可以不传 image_mask 的。但框架为了统一处理所有平台，强制要求这个字段存在。

✻ Waiting for 3 background agents to finish

● Agent "Search LeRobot data config and mask handling" finished · 37s

  Thought for 2s (ctrl+o to expand)

● 刚才的搜索任务完成了，结果与我的分析完全一致——简单总结一下补充发现：

  确认：LeRobot 数据集没有任何 image_mask 字段。

  image_mask 的完整生命周期是：

  Thought for 2s (ctrl+o to expand)

              obs.image_masks[name],     # ← 这里用到了！
              "b -> b s",
              s=image_tokens.shape[1],
          )
      )

  当 image_mask=False 时，该摄像头产生的所有 token 的 input_mask 被设为 False，在 attention 计算中这些 token 完全被屏蔽（softmax 前加 -inf），相当于这个摄像头"不存在"。

  这样设计的好处：
  - 你不需要因为某个平台少了一个摄像头就重新训练模型
  - 一个在 3 摄像头数据上训好的模型，拿给 2 摄像头的机器人用，只要把缺失的那个位置填黑图 + mask=False 就行
  - 训练时即使某个样本有一个摄像头坏了/掉了帧，也能继续参与训练，不用扔掉整个样本

  所以回到你的问题

  对于 LeRobot 数据集 + 所有摄像头都正常的场景：
  - ✅ 你说得对——image_mask 确实不起作用（全是 True）
  - ✅ 在 transform pipeline 中它是被"合成"出来的，不是 LeRobot 自带的
  - ❌ 但说"没什么用"不完全准确——它在其他平台（DROID、LIBERO）和摄像头偶尔断线的场景下是必要的

  如果你确定你的数据永远有全部摄像头，并且在 preprocess_observation() 中兜底默认就是全 True，那理论上是可以不传 image_mask 的。但框架为了统一处理所有平台，强制要求这个字段存在。

✻ Waiting for 3 background agents to finish

● Agent "Search LeRobot data config and mask handling" finished · 37s

  Thought for 2s (ctrl+o to expand)

● 刚才的搜索任务完成了，结果与我的分析完全一致——简单总结一下补充发现：

  确认：LeRobot 数据集没有任何 image_mask 字段。

  image_mask 的完整生命周期是：

  Thought for 2s (ctrl+o to expand)


  确认：LeRobot 数据集没有任何 image_mask 字段。

  image_mask 的完整生命周期是：

  1. 数据层（LeRobot） → 没有 image_mask
  2. 平台 Policy 层（你在 tron2_policy.py 里的 Tron2Inputs） → 合成 image_mask：
    - 检查每个摄像头是否存在 → 存在就 True，缺失填黑图设 False
  3. 模型前向（pi0.py:embed_prefix） → 用 image_masks[name] 把整个摄像头的 token 全部屏蔽或保留
  4. 兜底（model.py:preprocess_observation） → 如果某个摄像头根本没对应的 mask 键，自动全 True

  另外发现一个有趣的细节：π₀-FAST 模型对所有摄像头（包括填充的黑图）都设 mask=True，因为 FAST 架构中 padding 图像有特殊的处理逻辑，不能直接屏蔽
  token。这说明不同类型的模型对 mask 的处理方式不同。

  所以你之前的理解是对的——对于你 Tron2 这种所有摄像头都存在的场景，image_mask=True 是个恒等值，确实不起实际作用。




两个后台搜索任务都回来了，结果完全验证了之前的分析：

- **LeRobot 数据集没有 `image_mask` 字段** ✅
- **`image_mask` 只在 Policy 层的 `*Inputs` 中被合成** ✅
- **兜底层（`preprocess_observation`）会在缺失时自动补全 True** ✅

和你之前的直觉一致。


"""

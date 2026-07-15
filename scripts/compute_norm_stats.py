"""
计算指定配置的归一化统计量（均值和标准差）。

本脚本用于遍历数据集中的所有样本，累计计算各关键字段（如机器人状态 state 和动作 actions）
的均值（mean）和标准差（std），并将结果保存到配置指定的资产目录中。

这些统计量后续在训练/推理时用于对输入数据进行标准化（归一化），
确保模型各通道的数值范围相近，有助于训练的稳定性。
"""

"""
数据管线中 data_transforms 在 normalize 之前，normalize 使用的是数据集中的asset中的归一化量来归一化数据的。但是，aloha的数据管线中        
data_transforms =  
  _transforms.Group(                                                                                                                                           
              inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],                                                                                 
              outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],                                                                               
          )                                                                                                                                                    
  这里对数据进行了适配pi模型的操作，那之后再用数据集asset中的归一化数据进行归一化其不是不合理了，    
  return TransformedDataset(                                
          dataset,                                                                                                                                             
          [                                                                                                                                                    
              *data_config.repack_transforms.inputs,  # 第一步：键名重映射                                                                                     
              *data_config.data_transforms.inputs,  # 第二步：数据变换                                                                                         
              _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),  # 第三步：归一化                                                
              *data_config.model_transforms.inputs,  # 第四步：模型输入变换                                                                                    
          ],                                                                                                                                                   
      )我很疑惑，请你讲解。


好问题！我来把这个疑点彻底讲清楚。核心答案一句话就能说清，但背后涉及到一个你可能没注意到的关键文件。

---

## 一句话答案

**归一化统计量不是对原始数据算的，而是对 data_transforms 之后的数据算的。** 所以顺序完全合理。

---

## 详细解释

关键在 `compute_norm_stats.py` 中创建数据加载器的代码（第 35-43 行）：

```python
dataset = _data_loader.TransformedDataset(
    dataset,
    [
        *data_config.repack_transforms.inputs,   # ← 重排
        *data_config.data_transforms.inputs,     # ← data transforms (AlohaInputs, DeltaActions...)
        RemoveStrings(),                          # ← 去掉字符串字段
        # 注意：没有 Normalize ！
    ],
)
```

**计算归一化统计量的时候，并没有 `Normalize` 这步！** 它是在 **data_transforms 之后、Normalize 之前** 收集数据，算均值和标准差的。

然后在训练时，`transform_dataset`（`data_loader.py:440-448`）中：

```python
return TransformedDataset(dataset, [
    *data_config.repack_transforms.inputs,   # 1. 重排
    *data_config.data_transforms.inputs,     # 2. data transforms ← 跟算 stats 时完全一样
    _transforms.Normalize(norm_stats, ...),  # 3. 用第2步之后的数据算出的 stats 归一化 ← 一致！
    *data_config.model_transforms.inputs,    # 4. 模型变换
])
```

## 用 ALOHA 场景具体演示

以 `pi0_aloha_sim` 配置为例，一个数据样本的完整流水线是这样走的：

### 计算 norm stats 时
```
原始数据集样本:  state=[0.5, -1.2, ...], images=[C,H,W], action=[角度值...]
    │
    ▼ repack_transforms:  "action" → "actions",  "observation.state" → "state"
    │
    ▼ AlohaInputs(adapt_to_pi=True):
        ├─ 图像: [C,H,W] → [H,W,C]
        ├─ state: 关节翻转 [1,-1,-1,1,...] * state
        ├─ 夹爪: 线性位移 → 角度值
        └─ actions: 同样做逆变换
    │
    ▼ [此时 state 和 actions 已经是 pi 内部空间的数值了]
    │
    ▼ 扫整个数据集 → 算出 pi 内部空间的 mean 和 std
```

### 训练时
```
同样样本
    │
    ▼ repack_transforms  (同上)
    │
    ▼ AlohaInputs(adapt_to_pi=True)  (同上)
    │
    ▼ Normalize(mean_pi, std_pi)
    │  └─ 用刚才算好的 pi 内部空间的统计量归一化
    │
    ▼ model_transforms -> TokenizePrompt -> ResizeImages -> PadStatesAndActions
```

**所以 Normalize 用的 mean/std 本身就是在 data_transforms 之后的数据上算出来的，顺序完全一致，没有不合理的地方。**

---

## 你为什么会产生这个疑惑？

合理的推理链条。我猜你可能以为 norm stats 是数据集**自带的**（比如 LeRobot 数据集本身附带的属性），或者是 `compute_norm_stats.py` 直接在原始数据上计算的。

但实际上：

1. **数据集不提供 norm stats** — `compute_norm_stats.py` 是 openpi 的一个独立脚本，需要用户在训练前手动运行
2. **这个脚本在 `data_transforms` 之后算统计量** — 所以它感知到了坐标变换、delta 动作转换等所有变换
3. 归一化统计量保存为 `assets/{asset_id}/norm_stats.json`，里面存的是**变换后数据空间**的 mean/std

你可以想象 norm stats 文件里存的是什么：

```json
{
  "state": {
    "mean": [0.02, -0.15, ...],   // ← 这些是经过 AlohaInputs + DeltaActions 之后的 state 的均值
    "std":  [1.2, 0.8, ...]       //    ！！！
  },
  "actions": {
    "mean": [0.001, -0.003, ...],
    "std":  [0.05, 0.04, ...]
  }
}
```

它们已经是 pi 坐标系下的数值统计了，所以正好能在 `data_transforms` 之后直接使用。

**总结：norm stats 是在 data_transforms 之后、Normalize 之前的数据上计算的，所以先 data_transforms 后 Normalize 是完全自洽的。**

"""


import numpy as np
import tqdm  # 进度条库，用于显示遍历数据集的进度
import tyro  # 参数解析库，自动从命令行参数生成函数签名

# 导入 openpi 内部模块
import openpi.models.model as _model  # 模型基类定义
import openpi.shared.normalize as normalize  # 归一化工具（含 RunningStats 在线统计类）
import openpi.training.config as _config  # 训练配置系统
import openpi.training.data_loader as _data_loader  # 数据加载器工厂
import openpi.transforms as transforms  # 数据变换管线


class RemoveStrings(transforms.DataTransformFn):
    """
    数据变换：移除字典中所有字符串类型的字段。

    为什么要移除字符串？
    --------------------
    在本脚本中，我们只需要计算数值型字段（state、actions）的统计量。
    字符串字段（如语言指令 prompt、数据 ID 等）对均值和标准差没有意义，
    且 JAX 的数组运算不支持字符串类型，因此提前过滤掉，避免后续出错。
    """

    def __call__(self, x: dict) -> dict:
        # 保留所有值不是字符串子类型的键值对
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    """
    创建基于 PyTorch（torch.utils.data）的数据加载器。

    适用于 LeRobot 格式的数据集（如 LIBERO、ALOHA 等）。

    Args:
        data_config: 数据配置，包含数据集路径、变换配置等信息。
        action_horizon: 动作预测的时间窗口（未来多少步的动作）。
        batch_size: 每个训练批次的大小。
        model_config: 模型配置，用于确定输入/输出的形状等。
        num_workers: 数据加载时使用的子进程数（并行加载）。
        max_frames: 可选。限制最多使用的帧数（用于快速测试/调试）。

    Returns:
        (data_loader, num_batches): PyTorch 数据加载器 + 总批次数。
    """
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")

    # 1) 创建原始 PyTorch Dataset（从 HuggingFace Datasets / LeRobot 格式读取）
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)

    # 2) 在数据集上按顺序应用变换（transform chain）
    #    - repack_transforms: 重新组织/映射字段名，确保格式统一
    #    - data_transforms:   具体的数据增强/处理（如 DeltaActions 转换等）
    #    - RemoveStrings:     过滤掉所有字符串字段（JAX 不支持）
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,  # 输入阶段的字段重映射
            *data_config.data_transforms.inputs,  # 数据预处理变换（如归一化前的处理）
            RemoveStrings(),  # 移除字符串字段
        ],
    )

    # 3) 如果指定了 max_frames，且它小于全数据集大小，则只用前 max_frames 帧进行采样
    #    否则使用全量数据
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size  # 总批次数 = 限制帧数 ÷ batch_size
        shuffle = True  # 启用打乱（分批随机采样）
    else:
        num_batches = len(dataset) // batch_size  # 全量数据能分成多少批
        shuffle = False  # 不随机打乱（覆盖全部数据）

    # 4) 创建 PyTorch DataLoader 风格的加载器（支持多进程并行取数）
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,  # 控制总共输出多少个 batch（自定 epoch 长度）
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    """
    创建基于 RLDS（TensorFlow Datasets）的数据加载器。

    适用于大规模 RLDS 格式的数据集（如 Google DROID 数据集）。
    RLDS 使用 TFRecord 格式存储，支持流式读取，适合大规模数据。

    Args:
        data_config: 数据配置，包含 RLDS 数据目录等。
        action_horizon: 动作预测的时间窗口。
        batch_size: 每个训练批次的大小。
        max_frames: 可选。限制最多使用帧数。

    Returns:
        (data_loader, num_batches): RLDS 数据加载器 + 总批次数。
    """
    # 1) 创建 RLDS 格式的原始数据集（流式迭代器风格，不一次加载全部）
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)

    # 2) 应用变换链（与 create_torch_dataloader 相同）
    #    is_batched=True 表示输入已经是以 batch 为单位的，变换函数需按 batch 处理
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            RemoveStrings(),
        ],
        is_batched=True,  # RLDS 数据集已经是 batch 化的
    )

    # 3) 计算总批次数
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: len(dataset) 对于 DROID 是硬编码的（因为 RLDS 流式数据集无法预先知道确切长度）
        num_batches = len(dataset) // batch_size

    # 4) 创建 RLDS 专用的 DataLoader（流式读取，不支持多进程）
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(config_name: str, max_frames: int | None = None):
    """
    主函数：计算并保存指定配置的归一化统计量。

    执行流程：
    1. 加载指定名称的训练配置（模型、数据、优化器等）
    2. 根据数据格式（RLDS / PyTorch）创建对应的数据加载器
    3. 遍历所有数据，在线累加 state 和 actions 的统计量
    4. 计算最终的均值和标准差
    5. 保存到配置指定的资产目录

    Args:
        config_name: 配置名称。对应 config.py 中 _CONFIGS 字典里的键名。
                     e.g. "pi05_libero", "pi0_aloha_sim"
        max_frames: 可选。限制处理的最大帧数。不为 None 时可用于快速调试，
                    只采样部分数据计算统计量。
    """
    # 1) 根据名称获取训练配置（含模型、数据、优化器等所有设定）
    config = _config.get_config(config_name)

    # 2) 创建数据配置实例（进一步解析具体的数据集路径、变换配置等）
    data_config = config.data.create(config.assets_dirs, config.model)

    # 3) 根据数据格式选择不同的数据加载器
    if data_config.rlds_data_dir is not None:
        # 如果指定了 RLDS 数据目录 → 使用 RLDS 流式加载器
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        # 否则使用 PyTorch 风格加载器（LeRobot 格式）
        data_loader, num_batches = create_torch_dataloader(
            data_config,
            config.model.action_horizon,
            config.batch_size,
            config.model,
            config.num_workers,
            max_frames,
        )

    # 4) 准备统计量累积器
    #    我们只关心两个数值型字段：state（机器人状态）和 actions（机器人动作）
    keys = ["state", "actions"]

    # RunningStats 是一个在线累积均值和标准差的工具类。
    # 它使用 Welford 算法（单趟增量算法），避免一次性加载全部数据到内存。
    # 每次调用 update(batch_data) 时，它会用新一批数据更新内部计数、和、平方和。
    stats = {key: normalize.RunningStats() for key in keys}

    # 5) 遍历所有 batch，累积统计量
    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            # 将每个 key 对应的数据转换为 NumPy 数组，然后更新 RunningStats
            stats[key].update(np.asarray(batch[key]))

    # 6) 从 RunningStats 提取最终的统计量（mean, std, count）
    #    返回格式类似于: {"state": {"mean": ..., "std": ..., "count": ...}, "actions": {...}}
    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    # 7) 保存统计量到文件
    #    路径结构: {assets_dirs}/{repo_id}/norm_stats.json
    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)

    # 保存后的文件会被后续训练过程（train.py）自动加载，
    # 在数据预处理时将 state 和 actions 标准化到零均值单位方差的分布。


if __name__ == "__main__":
    # 使用 tyro 将 main 函数暴露为命令行接口。
    # tyro 会自动解析 sys.argv，并转换为函数的类型化参数。
    # 例如: python compute_norm_stats.py pi05_libero --max_frames 1000
    tyro.cli(main)
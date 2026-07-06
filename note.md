这是一个非常关键的问题。一条数据**只是一个片段（snippet），不是完整操作流程**。

## 一条数据长什么样？

以 `pi0_libero`（action_horizon=10）为例，**一条训练样本**经过所有变换后，最终进入模型的格式是：

```python
{
    # === 观测（当前时间步 t）===
    "image": {
        "base_0_rgb":        np.ndarray(224, 224, 3) uint8,   # 主摄像头画面
        "left_wrist_0_rgb":  np.ndarray(224, 224, 3) uint8,   # 腕部画面
        "right_wrist_0_rgb": np.ndarray(224, 224, 3) uint8,   # 零填充（LIBERO 无右腕）
    },
    "image_mask": {
        "base_0_rgb":        True,      # 真实图像
        "left_wrist_0_rgb":  True,      # 真实图像
        "right_wrist_0_rgb": False,     # 填充图像，模型忽略
    },
    "state": np.ndarray(8,),            # 当前时间步的机器人状态（关节位置等）

    # === 文本指令 ===
    "prompt": "pick up the red block",  # 还没分词

    # === 未来动作块（训练目标）===
    "actions": np.ndarray(10, 7),       # 未来 10 个时间步的动作
    #                               ↑         ↑
    #                        action_horizon  action_dim
}
```

**它只包含一个时间步的观测，以及接下来 10 步的动作用于训练目标。**

## 它是从完整流程中截取的

关键代码在 `create_torch_dataset`（`data_loader.py` 第 140-146 行）：

```python
dataset = lerobot_dataset.LeRobotDataset(
    data_config.repo_id,
    delta_timestamps={
        key: [t / dataset_meta.fps for t in range(action_horizon)]
        for key in data_config.action_sequence_keys
    },
)
```

`delta_timestamps` 告诉 LeRobot 数据集："给我当前帧之后的 N 个时间步的动作"。假设数据集是 10fps（每秒 10 帧），action_horizon=10：

```
完整操作流程（比如 200 帧，20 秒）： t=0,   t=1,   t=2,   ..., t=199
                                       ↓
样本 1: 观测 t=0  +  动作 [t=0,  t=1,  t=2,  ..., t=9]
样本 2: 观测 t=1  +  动作 [t=1,  t=2,  t=3,  ..., t=10]
样本 3: 观测 t=2  +  动作 [t=2,  t=3,  t=4,  ..., t=11]
...
样本 190: 观测 t=189 + 动作 [t=189, t=190, ..., t=198]
```

**LeRobot 数据集的底层存储结构**是这样的：

```
episode_0:
  frame_0:   {state: s0, action: a0, image: img0, ...}
  frame_1:   {state: s1, action: a1, image: img1, ...}
  frame_2:   {state: s2, action: a2, image: img2, ...}
  ...
  frame_199: {state: s199, action: a199, image: img199, ...}
```

每个 `__getitem__` 返回的不是"一集"，而是**一个帧加上未来 N 帧的动作为目标**。

## 这叫什么？——行为克隆（Behavior Cloning）

这正是模仿学习中最常见的方法——**行为克隆**。模型的任务是：

> **给定当前观测（图像 + 状态 + 文本），预测接下来一段时间的动作。**

训练时，每个样本都是一个 `(观测 → 动作块)` 的配对。模型学到的是"看到这个画面，就做出这个动作序列"的映射，而不是学习完整任务的长程规划。

```
观测 t     →  动作 [t, t+1, ..., t+9]
(图像+状态)    (模型要预测的目标)
```

这是基于一个假设：**机器人控制是马尔可夫（或近似马尔可夫）的**——当前观测足够决定接下来一小段时间的行动，不需要看整个历史。

## batch 的组成

一个 batch 包含 N 条这样的独立样本，每条来自**不同的时间步**，可能来自**不同的操作流程**：

```python
batch = {
    "image":        {"base_0_rgb": (B, 224, 224, 3), ...},
    "image_mask":   {"base_0_rgb": (B,), ...},
    "state":        (B, 8),
    "actions":      (B, 10, 7),   # B 条数据，每条 10 步动作
    "tokenized_prompt":    (B, L),
    "tokenized_prompt_mask": (B, L),
}
```

没有"序列维度"——batch 中的每条样本是独立的，不按时间顺序排列。模型在 batch 维度上并行计算，每条样本独立预测自己的动作块。









'''
这是一个**类型别名（Type Alias）**，定义了一个可以同时兼容 JAX 和 PyTorch 张量的统一类型。

```python
Array = jax.Array | torch.Tensor
```

### 含义

这行代码创建了一个名为 `Array` 的类型别名，等价于 `jax.Array | torch.Tensor`（Python 3.10+ 的联合类型语法），意思是：

> **"`Array` 类型要么是一个 JAX 数组，要么是一个 PyTorch 张量，二选一。"**

### 为什么需要这个？

openpi 同时支持两种后端：

| 后端        | 数组类型       | 典型用途                           |
| ----------- | -------------- | ---------------------------------- |
| **JAX**     | `jax.Array`    | 训练和部分推理（JAX 生态）         |
| **PyTorch** | `torch.Tensor` | PyTorch 训练 / 从 JAX 转换后的推理 |

但 `Observation`、`Actions` 等数据结构需要能容纳**两种后端的数据**。回到 `model.py` 的类型变量定义：

```python
ArrayT = TypeVar("ArrayT", bound=jax.Array | torch.Tensor | np.ndarray)
```

这里 `bound` 限制了 `ArrayT` 的取值范围，而 `array_typing.py` 中的 `Array = jax.Array | torch.Tensor` 就是具体的联合类型，用于类型注解中表明"这个位置可以接受 JAX 数组或 PyTorch 张量"。

### 使用示例

`model.py` 中的 `Observation` 使用了 `ArrayT` 泛型，所以同一个 `Observation` 类可以实例化为：

```python
# JAX 版本
obs_jax: Observation[jax.Array]  # images 是 dict[str, Float[jax.Array, ...]]

# PyTorch 版本
obs_torch: Observation[torch.Tensor]  # images 是 dict[str, Float[torch.Tensor, ...]]
```

而 `Actions` 的类型注解直接用了 `ArrayT`：

```python
Actions = at.Float[ArrayT, "*b ah ad"]
```

—— 这里的 `ArrayT` 最终被 `Array = jax.Array | torch.Tensor` 所约束，所以 `Actions` 可以是 JAX 数组也可以是 PyTorch 张量。

### 总结

就是一句大白话：**"这个变量可以是 JAX 的数组，也可以是 PyTorch 的张量，写代码时两种都接受。"** 它是 openpi 双后端设计的基础设施。
'''







## 实际大小：约 **10 GB**

从 GCS 上拉到的真实数据：

| Checkpoint                 | 文件数 | 大小        |
| -------------------------- | :----: | :---------: |
| `pi0_fast_droid` ✅ 你问的 | 21     | **10.1 GB** |
| `pi0_libero`               | 19     | 11.2 GB     |
| `pi0_base`                 | 33     | 11.2 GB     |
| `pi05_base`                | 29     | 11.6 GB     |
| `pi0_fast_base`            | 35     | 10.1 GB     |

所以 `pi0_fast_droid` 下载下来约 **10 GB**，占用磁盘 ~11 GB（算上文件系统的额外开销）。

---

## 这 10 GB 装了啥？

从文件数（21个）和大小来看，checkpoint 里包含：

```
checkpoints/pi0_fast_droid/
├── train_state/         ← 模型参数 + 优化器状态
│   ├── params           ← 模型权重本身（约 ~5.4 GB，bfloat16，2.7B 参数）
│   └── opt_state        ← AdamW 动量/方差（约 ~5.4 GB）
├── assets/              ← 归一化统计量、配置等（几 MB 到几十 MB）
└── metadata             ← 检查点元信息
```

模型一共约 **2.7B 参数**（PaliGemma 2B + SigLIP 400M + Action Expert 300M），每参数 2 字节（bfloat16）= ~5.4 GB。优化器状态（AdamW 的 momentum + variance 两倍于参数）= 另外 ~5.4 GB。加起来刚好 ~10.8 GB，和实际数据吻合。

---

## 下载耗时参考

| 带宽     | 预估用时  |
| :------: | :-------: |
| 100 Mbps | ~14 分钟  |
| 500 Mbps | ~3 分钟   |
| 1 Gbps   | ~1.5 分钟 |

实际会慢一些，因为 gsutil 作多线程下载且 GCS 通常限速。另外首次下载后缓存在 `~/.cache/openpi/`，以后不再重新下载。
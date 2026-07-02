"""
=============================================================
  openpi 数据变换（transforms）模块
=============================================================

本模块提供了一套用于机器人策略（policy）训练和推理的数据变换工具链。

核心概念：
  - DataDict（数据字典）：一个可能嵌套的字典结构，每个叶子节点是一个 numpy 数组。
    例如：{"image": {"cam_high": ..., "cam_low": ...}, "state": ..., "actions": ...}
  - DataTransformFn（数据变换函数）：接收一个 DataDict，返回一个变换后的 DataDict。
  - Group（变换组）：将变换分为"输入变换"和"输出变换"两组，方便管理。

工作流程：
  训练时：
    原始数据集 → RepackTransform（重排字段）→ 各种 Normalize（归一化）→
    ResizeImages（调整图像）→ TokenizePrompt（分词）→ 模型输入
  推理时：
    模型输出 → Unnormalize（反归一化）→ AbsoluteActions（转绝对动作）→ 最终控制信号
"""

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import re
from typing import Protocol, TypeAlias, TypeVar, runtime_checkable

import flax.traverse_util as traverse_util
import jax
import numpy as np
from openpi_client import image_tools

from openpi.models import tokenizer as _tokenizer  # 分词器（处理语言指令）
from openpi.shared import array_typing as at  # 数组类型工具
from openpi.shared import normalize as _normalize  # 归一化统计量

# DataDict（数据字典）是本模块中最核心的数据类型。
# 它是一个可能嵌套的字典，叶子节点是 numpy 数组（或 JAX 数组）。
# PyTree 是 JAX 社区的术语，指"可以由任意嵌套的字典/列表/元组构成的树状结构"。
DataDict: TypeAlias = at.PyTree

# NormStats（归一化统计量）存储用于归一化的统计数据（均值、标准差等）。
NormStats: TypeAlias = _normalize.NormStats


T = TypeVar("T")
S = TypeVar("S")


@runtime_checkable
class DataTransformFn(Protocol):
    """数据变换函数的协议（接口规范）。

    任何符合此协议的对象都可以作为一个数据变换步骤。这意味着你只需要实现
    一个 __call__ 方法，接收 DataDict 并返回 DataDict，就可以被纳入变换流水线。

    使用 @runtime_checkable 装饰器，意味着可以用 isinstance() 在运行时检查
    一个对象是否符合此协议。

    示例：
        class MyTransform:
            def __call__(self, data: DataDict) -> DataDict:
                data["state"] = data["state"] * 2.0
                return data
    """

    def __call__(self, data: DataDict) -> DataDict:
        """对数据应用变换。

        Args:
            data: 待变换的数据。它是一个可能嵌套的字典，包含"未批处理（unbatched）"的数据元素。
                  每个叶子节点预期是 numpy 数组。也可以使用 JAX 数组，但不推荐，
                  因为在数据加载器的子进程中可能导致额外的 GPU 内存使用。

        Returns:
            变换后的数据。可以是对输入的 data 进行原地修改后返回，也可以返回全新的数据结构。
        """


@dataclasses.dataclass(frozen=True)
class Group:
    """一组变换（transforms），分为输入变换和输出变换两组。

    在 openpi 中，我们将"数据预处理"分为两个阶段：
      1. 输入变换（inputs）——对模型输入数据进行的变换（如归一化、分词）
      2. 输出变换（outputs）——对模型输出数据进行的变换（如反归一化）

    frozen=True 表示这个类的实例创建后不可修改，保证了线程安全。

    使用 .push() 方法可以创建包含更多变换的新 Group，而不会修改原 Group。
    """

    # 应用于模型输入数据的变换序列。
    # 例如：归一化、调整图像尺寸、分词等
    inputs: Sequence[DataTransformFn] = ()

    # 应用于模型输出数据的变换序列。
    # 例如：反归一化、将增量动作（delta actions）转换为绝对动作（absolute actions）
    outputs: Sequence[DataTransformFn] = ()

    def push(self, *, inputs: Sequence[DataTransformFn] = (), outputs: Sequence[DataTransformFn] = ()) -> "Group":
        """将变换添加到当前组，返回一个新的 Group，原 Group 保持不变。

        这体现了"不可变对象"的设计模式：不修改自身，而是返回新对象。

        Args:
            inputs: 追加到"当前输入变换的末尾"。
                    因为输入变换是顺序执行的，新的变换加在后面。
            outputs: 追加到"当前输出变换的开头"。
                    注意和 inputs 相反！这是因为输出变换的执行顺序
                    也是正向的（从前往后），但 Group.push() 的设计约定
                    是 inputs 追加到末尾，outputs 追加到开头。
                    这样当你多次调用 push 时，最终 output 变换的"构建顺序"
                    和"执行顺序"是一致的。

        Returns:
            包含追加后的变换的新 Group。
        """
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """组合变换：按顺序依次应用多个变换。

    这是一个"组合模式（Composite Pattern）"的实现，将多个变换步骤
    组合成一个变换，外部看来就像一个单一变换一样。

    例如：compose([Normalize(...), ResizeImages(...)]) 就是一个 CompositeTransform。
    """

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """将多个变换组合成一个变换。

    这是 CompositeTransform 的便捷工厂函数。相当于：
        compose([A, B, C]) 等价于 CompositeTransform([A, B, C])
    执行效果：先 A，再 B，再 C。

    Args:
        transforms: 要组合的数据变换序列。

    Returns:
        一个单一的 DataTransformFn，内部顺序执行所有变换。
    """
    return CompositeTransform(transforms)


@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """重排变换：将输入的字典重新组织为新的字典结构。

    机器人数据集通常以特定格式存储（如 LeRobot 格式），但模型需要
    统一的输入格式。RepackTransform 负责将不同的数据格式映射到
    模型所期望的字段名称和结构。

    工作原理：
      1. 将输入字典"拍平（flatten）"，用"/"作为路径分隔符
         {"observation": {"state": [1,2]}} → {"observation/state": [1,2]}
      2. 根据 structure 中定义的映射关系，从拍平的字典中取值
      3. 将其组装成新的字典结构

    示例：
        structure = {
            "image": {
                "cam_high": "observation/images/top",    # 新键 cam_high ← 旧路径 observation/images/top
                "cam_low": "observation/images/bottom",   # 新键 cam_low  ← 旧路径 observation/images/bottom
            },
            "state": "observation/state",  # 新键 state ← 旧键 observation/state
            "actions": "action",           # 新键 actions ← 旧键 action
        }

    这样无论原始数据集字段名叫什么，最终模型收到的都是统一格式的输入。
    """

    # 一个字符串的 PyTree（嵌套字典），其结构与期望的输出结构一致。
    # 叶子节点的字符串值表示从输入中取值用的"拍平后的路径"。
    structure: at.PyTree[str]

    def __call__(self, data: DataDict) -> DataDict:
        # 步骤1：将嵌套字典拍平
        # {"obs": {"state": [1,2]}, "action": [0,0]} → {"obs/state": [1,2], "action": [0,0]}
        flat_item = flatten_dict(data)

        # 步骤2：根据 structure 中的路径映射，创建新字典
        # jax.tree.map 会遍历 self.structure 的每个叶子节点（那是一个字符串路径），
        # 用 lambda 从 flat_item 中取出对应的值。
        return jax.tree.map(lambda k: flat_item[k], self.structure)


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    """注入默认提示文本：如果数据中没有 prompt 字段，则使用预设的默认值。

    在机器人策略中，"prompt"（提示）通常是自然语言指令，例如"拿起红色方块"。
    有些数据集可能没有 prompt 字段（比如只在固定任务上操作的场景），
    这时可以使用这个变换注入一个默认提示。

    prompt: str | None
        - 如果为 None：什么都不做，数据原样返回。
        - 如果为字符串：仅当数据中不存在 "prompt" 键时，将其注入。
    """

    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = np.asarray(self.prompt)
        return data


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    """归一化变换：使用预计算的统计量对数据进行归一化。

    为什么要归一化？
      在训练神经网络时，不同维度的数据可能具有不同的尺度和范围（比如位置坐标在
      0~1 之间，而关节角度在 -180~180 之间）。如果不归一化，大尺度的特征会主导
      训练过程，导致模型难以收敛。

    支持的归一化方式：
      1. Z-score 归一化（标准分数）：
           x_norm = (x - mean) / (std + 1e-6)
         适用于数据近似正态分布的情况。1e-6 是为了防止除零。

      2. 分位数归一化（quantile normalization）：
           x_norm = (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
         适用于数据分布不均匀或有异常值的情况。使用 1% 和 99% 分位数
         代替最小/最大值，可以抵抗离群值的影响。将数据映射到 [-1, 1] 区间。
    """

    # 预计算的归一化统计量。这是一个 DataDict（嵌套字典），结构与输入数据对应。
    # 每个叶子节点是一个 NormStats 对象（包含 mean, std 或 q01, q99）。
    # 如果为 None，则不进行归一化。
    norm_stats: at.PyTree[NormStats] | None

    # 如果为 True，使用分位数归一化；否则使用标准 Z-score 归一化。
    use_quantiles: bool = False

    # 如果为 True，当 norm_stats 中有某个键在数据中不存在时，抛出错误。
    strict: bool = False

    def __post_init__(self):
        """在 __init__ 之后自动调用，用于验证参数的有效性。

        这里检查：如果既要求分位数归一化，又提供了 norm_stats，
        则确保 norm_stats 中包含了必要的分位数统计量（q01 和 q99）。
        """
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        return apply_tree(
            data,
            self.norm_stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )

    def _normalize(self, x, stats: NormStats):
        """Z-score 归一化（标准分数归一化）。

        公式: x_norm = (x - mean) / (std + 1e-6)

        注意：stats.mean 和 stats.std 的最后一个维度可能比 x 的最后一个维度大，
        （因为 pre-computed 时可能计算了更多维度的统计量），所以我们只取前面
        x.shape[-1] 个维度。1e-6 用于防止标准差为 0 时除零。
        """
        mean, std = stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]]
        return (x - mean) / (std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        """分位数归一化。

        公式: x_norm = (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0

        q01 = 第 1 百分位数，q99 = 第 99 百分位数。
        使用 1% 和 99% 分位数代替全局最小/最大值的好处：
          - 可以忽略极端异常值的影响
          - 将数据的"有效范围"映射到 [-1, 1] 区间
        """
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    """反归一化变换：将归一化后的数据恢复到原始的数值范围。

    这个变换主要用于推理阶段。模型输出的是归一化后的动作值，
    我们需要将它们"还原"为机器人可以执行的原始动作值。

    例如：
      归一化后的动作值：[-0.5, 0.3]
      → 反归一化后的值：[0.2, 0.8]（原始范围）

    Z-score 反归一化公式：
      x = x_norm * (std + 1e-6) + mean

    分位数反归一化公式：
      x = (x_norm + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
    """

    norm_stats: at.PyTree[NormStats] | None
    use_quantiles: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        # 确保所有 norm_stats 中的键都在数据中存在。
        # 因为推理时必须对所有模型输出做反归一化，漏掉任何一个都是危险的。
        return apply_tree(
            data,
            self.norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
            strict=True,
        )

    def _unnormalize(self, x, stats: NormStats):
        """Z-score 反归一化。

        因为模型输出的动作维度可能比预计算统计量的维度多（例如策略头可能
        预测了额外维度），所以这里用 pad_to_dim 将统计量扩展到匹配的维度。
        mean 缺少的维度用 0 填充，std 缺少的维度用 1 填充。
        """
        mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
        std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
        return x * (std + 1e-6) + mean

    def _unnormalize_quantile(self, x, stats: NormStats):
        """分位数反归一化。

        如果统计量的维度比 x 小（dim < x.shape[-1]），说明 x 有额外的维度
        没有对应的分位数统计信息。这时只对前 dim 个维度做反归一化，
        后面的维度保持不变。
        """
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01, stats.q99
        if (dim := q01.shape[-1]) < x.shape[-1]:
            return np.concatenate([(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]], axis=-1)
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    """图像尺寸调整变换：将所有相机图像调整为统一尺寸。

    为什么需要这个变换？
      不同的摄像头可能输出不同分辨率的图像，但神经网络要求输入具有固定的尺寸。
      这个变换将所有图像调整为指定的 height × width，并保持宽高比
      （不足的部分用黑色填充）。

    数据字典中的 "image" 键应该是一个子字典，包含各相机名称到图像数组的映射。
    例如：{"image": {"cam_high": (H1, W1, 3), "cam_low": (H2, W2, 3)}}
    变换后：{"image": {"cam_high": (height, width, 3), "cam_low": (height, width, 3)}}
    """

    height: int  # 目标高度
    width: int  # 目标宽度

    def __call__(self, data: DataDict) -> DataDict:
        # 对每个相机图像应用 resize_with_pad 变换
        data["image"] = {k: image_tools.resize_with_pad(v, self.height, self.width) for k, v in data["image"].items()}
        return data


@dataclasses.dataclass(frozen=True)
class SubsampleActions(DataTransformFn):
    """动作序列下采样变换：对动作序列进行降采样（每隔 stride 步取一个动作）。

    在机器人数据中，控制频率可能很高（比如 50Hz），但模型实际上不需要预测
    每一个时间步的动作，只需要每隔几帧预测一次就够了。这样既减少了计算量，
    也降低了动作预测的冗余性。

    例如，如果 stride=3，输入动作序列 [a0, a1, a2, a3, a4, a5, ...]
    输出变为 [a0, a3, a6, ...]
    """

    stride: int

    def __call__(self, data: DataDict) -> DataDict:
        data["actions"] = data["actions"][:: self.stride]
        return data


@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    """增量动作变换：将绝对动作（absolute actions）转换为增量动作（delta actions）。

    什么是增量动作？
      增量动作 = 当前时刻的目标位置 - 当前状态（机械臂关节位置）
      即：delta_action = target - current_state

    为什么要用增量动作？
      在很多机器人操控场景中，增量动作比绝对动作更稳定。因为增量动作只表示
      "相对于当前位置移动多少"，而不是"移动到哪个绝对位置"。这样即使状态
      估计有微小误差，增量动作的累积误差也更小。

    工作原理：
      对于掩码为 True 的维度：actions -= state（计算增量）
      对于掩码为 False 的维度：保持不变

    mask 参数：
      一个布尔值序列，指示哪些动作维度应该转换为增量形式。
      例如 mask=[True, True, False, False, True, True] 表示
      第 0、1、4、5 维度做增量转换，第 2、3 维度保持不变。
    """

    # 布尔掩码，指示哪些动作维度要转换为增量空间。
    # 长度可以小于实际的维度数（未指定的维度不处理）。
    # 如果为 None，则该变换不做任何操作。
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]

        # 关键计算：对于 mask=True 的维度，从动作中减去当前状态
        # np.where(mask, state, 0)：对 mask=True 的维度取 state 值，False 的维度取 0
        # np.expand_dims(..., axis=-2)：在倒数第二维扩展，以匹配 actions 的维度
        #   （actions 通常有额外的"时间步"维度或"动作候选"维度）
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteActions(DataTransformFn):
    """绝对动作变换：将增量动作（delta actions）恢复为绝对动作。

    这是 DeltaActions 的逆操作，通常用于推理阶段。
    公式：absolute_action = delta_action + state

    mask 参数说明与 DeltaActions 相同。
    """

    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] += np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    """提示文本分词变换：将自然语言指令转化为 token 序列。

    在 VLA（Vision-Language-Action，视觉-语言-动作）模型中，语言指令
    （如"拿起红色方块"）需要被编码为 token 序列才能输入给模型。

    这个变换使用 PaligemmaTokenizer（Gemma 系列模型的分词器）来处理文本。
    如果需要使用"离散状态输入"（discrete state input，π₀.₅ 的新特性），
    还会将机器人状态也通过分词器编码为离散 token。

    Args:
        tokenizer: Paligemma 分词器实例
        discrete_state_input: 是否将状态也作为离散 token 输入（π₀.₅ 模型使用）

    返回的字段：
        - tokenized_prompt: 分词后的 token 序列
        - tokenized_prompt_mask: token 的注意力掩码（哪些 token 是有效的）
    """
    
    '''
## 掩码的作用：区分"有效内容"和"填充"

`tokenized_prompt_mask` 是一个布尔数组，长度与 `tokenized_prompt` 相同，作用是**告诉模型哪些 token 是真正的文本内容，哪些是补齐用的 padding**。

### 为什么不直接用一个固定长度的 token 序列？

因为不同的自然语言指令**长度不同**：

```
"拿起红色方块"            → 5 个 token
"把左边的蓝色杯子放到托盘上" → 12 个 token
```

但模型需要**固定长度的输入**（由 `max_token_len` 决定）。所以短句会被 **padding**（填充）到固定长度：

```
假设 max_token_len = 16：

"拿起红色方块"  →  [45, 23, 67, 89, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
"把左边的蓝色杯子放到托盘上" → [77, 34, 56, 12, 90, 23, 45, 67, 11, 89, 34, 56, 0, 0, 0, 0]
```

这里 `0` 是 padding token。但模型怎么知道哪些 token 是真正的文本，哪些是 padding？

### 掩码发挥作用

```python
"拿起红色方块":
  tokenized_prompt:      [45, 23, 67, 89, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  tokenized_prompt_mask: [1,  1,  1,  1,  1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                              ↑ 有效内容 ↑        ↑     padding     ↑

"把左边的蓝色杯子放到托盘上":
  tokenized_prompt:      [77, 34, 56, 12, 90, 23, 45, 67, 11, 89, 34, 56, 0, 0, 0, 0]
  tokenized_prompt_mask: [1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 0, 0, 0, 0]
                              ↑       有效内容 (12个token)        ↑   ↑ padding ↑
```

**`mask[i] = True`** → token `i` 是有效的文本内容，参与 attention 计算
**`mask[i] = False`** → token `i` 是填充，模型必须忽略它（attention 时置零）

### 在模型内部的具体效果

在 Transformer 的 attention 机制中，这个 mask 会被加到 attention score 矩阵上：

```
attention_score[i][j] += mask[j] ? 0 : -inf
                                               ↑
                                    被 mask 的位置变成 -inf
                                    经过 softmax 后变为 0
                                    相当于这些 token "不存在"
```

所以当一个样本是短句时，模型**只看前几个有效 token**，后面的 padding 完全被屏蔽，不会影响计算结果，也不会产生梯度。

### 与 image_mask 的对应关系

值得注意的一个设计对称性：

| 模态     | 有效数据                       | 无效/缺失数据                                    |
| -------- | ------------------------------ | ------------------------------------------------ |
| **图像** | `image_mask = True`            | `image_mask = False` → 特征置零                  |
| **文本** | `tokenized_prompt_mask = True` | `tokenized_prompt_mask = False` → attention 屏蔽 |

两种 mask 本质上在做同一件事：**告诉模型"哪些输入是真的，哪些是占位符，请忽略占位符"**。    
    '''
    

    tokenizer: _tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        # 从数据中取出 prompt 并移除（不再需要原始文本）
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        # 如果是离散状态输入模式，从数据中取出 state
        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        # 如果 prompt 是 numpy 数组中的单个元素，提取出字符串
        if not isinstance(prompt, str):
            prompt = prompt.item()

        # 分词：将文本（和可选的状态向量）转换为 token IDs 和掩码
        tokens, token_masks = self.tokenizer.tokenize(prompt, state)
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}
    '''
你说得很对，你的直觉在**架构层**上是完全正确的。

## 你的理解没有错

Transformer 的 **attention 机制本身确实支持任意长度**：

```
           Q₁ Q₂ Q₃           Q₁ Q₂ Q₃ Q₄ Q₅
           ↓  ↓  ↓            ↓  ↓  ↓  ↓  ↓
Attn = [  ·  ·  ·  ]    vs    [  ·  ·  ·  ·  ·  ]
         ↑  ↑  ↑              ↑  ↑  ↑  ↑  ↑
         K₁ K₂ K₃             K₁ K₂ K₃ K₄ K₅
```

矩阵大小随序列长度变化，但这是**动态的**，没有理论限制。Gemma 模型也是用 RoPE（旋转位置编码），本身就支持长度外推。

那 `max_token_len` 是干什么的？它不是一个**架构限制**，而是一个**工程上的固定边界**。

---

## 为什么实际使用中需要固定一个上限？

### 1. JAX 要求静态形状（最根本的原因）

```python
# PyTorch 中常见的动态形状：
x = torch.randn(batch_size, seq_len, dim)  # seq_len 可以每步不同

# JAX 中 JIT 编译要求：
@jax.jit
def forward(x):         # ← 编译时 shapes 被冻结
    ...

# 如果 run 时 x 的形状变了 → 重新编译 → 慢几百倍
```

openpi 用 **JAX** 训练，JAX 的 JIT 编译器要求**所有张量形状在编译时确定**。你不能让 `seq_len` 在每次 forward 时变化——要么重新编译（极慢），要么固定到最大值。

### 2. Batch 打包需要对齐

即使单条推理时长度可变，**训练时 batch 内的所有样本必须长度一致**：

```
batch = [
    "拿起红色方块",              # 5 tokens
    "把左边的蓝色杯子放到托盘上",   # 12 tokens
    "按按钮",                    # 3 tokens
]
```

如果不对齐到同一个长度，`torch.stack()` / `jnp.stack()` 会报错。所以必须选一个 `max_token_len`，短的 padding，长的截断。

### 3. 预训练权重不是真正"无限"的

Paligemma 在预训练时有一个**最大上下文长度**（通常是 256 / 512 / 2048）。

> Gemma 2B 的原始上下文长度最大是 8192，但 Paligemma（视觉-语言版本）在训练时图像 patch 占了大部分 token budget，留给文本 token 的位置是**预先分配好**的。

如果 `max_token_len` 设置得太大：
- 无意义的 padding 太多，浪费显存和计算
- 超过预训练时的序列长度分布，模型可能没见过那么长的提示

---

## 所以 `max_token_len` 到底是什么？

回到 `BaseModelConfig` 的定义：

```python
class BaseModelConfig(abc.ABC):
    action_dim: int
    action_horizon: int
    max_token_len: int  # ← 这是"分配的上限"，不是"理论极限"
```

它决定了：
1. **Embedding 表的大小**（或者更准确地说，决定了 position embedding 支持的最大长度）
2. **Batch padding 的目标长度**
3. **JAX JIT 编译时的固定 shape**

它不是"架构上限"（模型不能处理超过这个长度的序列），而是**"效率/显存预算的上限"**（模型在这个长度以内高效工作，超过可能需要重新编译或 OOM）。

---

## 快速类比

|                | 理论能力             | 实际使用                                              |
| -------------- | -------------------- | ----------------------------------------------------- |
| **Llama 3**    | 支持任意长度（RoPE） | 训练时固定 8192，长了会重编译或退化                   |
| **Gemma**      | 支持任意长度（RoPE） | 预训练时固定 8192，推理可以外推但质量下降             |
| **openpi VLM** | 支持任意长度         | max_token_len = 256（因为指令本来就很短，没必要更大） |

**你的直觉是正确的**——Transformer 架构确实支持动态序列长度。`max_token_len` 不是架构限制，而是 **JAX 静态编译 + batch padding + 显存预算**共同决定的工程常数。
    '''
    

@dataclasses.dataclass(frozen=True)
class TokenizeFASTInputs(DataTransformFn):
    """FAST 模型输入分词变换。

    这是专门用于 π₀-FAST（自回归架构，Autoregressive）模型的分词变换。
    与普通的 TokenizePrompt 不同，FAST 模型需要额外的输出信息：

    1. 状态（state）和动作（actions）都会被编码为 token 序列
    2. 生成额外的注意力掩码（attention mask）和损失掩码（loss mask）

    返回的额外字段：
        - token_ar_mask: 自回归掩码，控制哪些 token 参与自回归预测
        - token_loss_mask: 损失掩码，控制哪些 token 参与损失计算
          （只有动作 token 的损失才被计算，语言部分的 token 不计入损失）
    """

    tokenizer: _tokenizer.FASTTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        state, actions = data["state"], data.get("actions")
        tokens, token_mask, ar_mask, loss_mask = self.tokenizer.tokenize(prompt, state, actions)
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
        }


@dataclasses.dataclass(frozen=True)
class ExtractFASTActions(DataTransformFn):
    """FAST 模型动作提取变换：将 FAST 模型输出的 token 转换为实际的动作值。

    FAST 模型输出的是"动作 token"（离散表示），而不是直接的动作值。
    这个变换使用 FAST 分词器的 extract_actions 方法，将 token 解码为
    连续的动作值。

    通常用于推理阶段，在模型输出后调用。

    Args:
        tokenizer: FAST 分词器（包含将 token 解码为动作的逻辑）
        action_horizon: 动作预测的步长（模型一次预测多少个未来动作）
        action_dim: 动作空间的维度（例如 7 维的机械臂关节控制）
    """

    tokenizer: _tokenizer.FASTTokenizer
    action_horizon: int  # 动作序列长度（时间步数）
    action_dim: int  # 每个动作的维度

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data
        # 注意：对于 FAST 模型，data["actions"] 中保存的是模型输出的 token，
        # 不是真正的动作值。所以这里先 pop 出来，解码后再放回去。
        tokens = data.pop("actions")
        actions = self.tokenizer.extract_actions(tokens.astype(np.int32), self.action_horizon, self.action_dim)
        return {
            **data,
            "actions": actions,
        }


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask(DataTransformFn):
    """从 LeRobot 数据集任务中提取提示文本。

    LeRobot 是一个常用的机器人数据集格式。在 LeRobot 数据集中，每个数据点
    包含一个 task_index（任务索引），对应一个任务描述字符串。

    这个变换的作用：根据 task_index 从任务映射表中查找对应的自然语言指令。

    适用场景：
      当你使用 LeRobot 格式的数据集，且数据集已经包含了任务描述时使用。

    Args:
        tasks: 任务索引到任务描述的映射。例如 {0: "pick up the red block", 1: "open the drawer"}
    """

    tasks: dict[int, str]

    def __call__(self, data: DataDict) -> DataDict:
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')

        task_index = int(data["task_index"])
        if (prompt := self.tasks.get(task_index)) is None:
            raise ValueError(f"{task_index=} not found in task mapping: {self.tasks}")

        return {**data, "prompt": prompt}


@dataclasses.dataclass(frozen=True)
class PadStatesAndActions(DataTransformFn):
    """状态和动作零填充：将状态和动作的维度扩展到模型的期望维度。

    为什么需要这个变换？
      不同的机器人可能有不同维度的状态/动作空间（例如 6 自由度和 7 自由度的
      机械臂），但模型通常固定了输入/输出维度（比如 model_action_dim=32）。
      这个变换将不足的维度用 0 填充，使其与模型维度匹配。

    典型场景：
      低维度的动作空间（如 7 维）被填充到模型的 32 维动作空间。
      多余的维度在训练时会被损失函数的掩码忽略掉。
    """

    model_action_dim: int  # 模型期望的动作维度

    def __call__(self, data: DataDict) -> DataDict:
        # 将 state 的最后一个维度填充到 model_action_dim
        data["state"] = pad_to_dim(data["state"], self.model_action_dim, axis=-1)
        if "actions" in data:
            data["actions"] = pad_to_dim(data["actions"], self.model_action_dim, axis=-1)
        return data


# =============================================================
#  工具函数（Utility Functions）
# =============================================================


def flatten_dict(tree: at.PyTree) -> dict:
    """将嵌套字典拍平成一个扁平的字典，使用 '/' 作为路径分隔符。

    例如：
        输入：{"observation": {"state": np.array([1,2]), "images": {"cam": np.array([3,4])}}}
        输出：{"observation/state": [1,2], "observation/images/cam": [3,4]}

    这样做的目的是方便进行路径匹配和键值操作。

    Args:
        tree: 可能嵌套的字典。

    Returns:
        拍平后的字典，键是用 '/' 连接的多级路径字符串。
    """
    return traverse_util.flatten_dict(tree, sep="/")


def unflatten_dict(tree: dict) -> at.PyTree:
    """将拍平后的字典恢复为嵌套的字典结构。

    flatten_dict 的逆操作。

    例如：
        输入：{"observation/state": [1,2], "observation/images/cam": [3,4]}
        输出：{"observation": {"state": [1,2], "images": {"cam": [3,4]}}}

    Args:
        tree: 键是用 '/' 分隔路径的扁平字典。

    Returns:
        恢复为嵌套结构后的字典。
    """
    return traverse_util.unflatten_dict(tree, sep="/")


def transform_dict(patterns: Mapping[str, str | None], tree: at.PyTree) -> at.PyTree:
    """使用模式匹配来转换嵌套字典的结构。

    这个函数提供了一种灵活的方式来重命名/删除字典中的键，支持正则表达式匹配
    和反向引用替换。

    工作原理：
        1. 将输入字典拍平为扁平键值对
        2. 对每个键，尝试匹配 patterns 中的正则表达式（按顺序）
        3. 匹配成功后，用替换字符串生成新键
        4. 如果替换值为 None，则删除该键值对
        5. 如果没有匹配到任何模式，保留原键

    重要规则：
        - patterns 中的顺序很重要！只有第一个匹配的模式会被使用。
        - 正则表达式必须匹配"整个"键（使用 fullmatch），而不是部分匹配。
        - 替换后的键不能冲突（不能有两个值映射到同一个新键）。

    Args:
        patterns: 一个映射，键是正则表达式模式（匹配旧键），
                  值是新键的替换字符串（支持 re.sub 的反向引用语法），
                  或 None（表示删除该键）。
        tree: 要转换的嵌套字典。

    Returns:
        转换后的嵌套字典。

    示例：
        假设输入数据为：
            {"observation.state": [1,2], "observation.action": [3,4]}

        patterns = {
            "observation/(.*)": r"obs_\1",  # 将 observation/xxx 重命名为 obs_xxx
            "unused_key": None,              # 删除 unused_key
        }

        结果：{"obs_state": [1,2], "obs_action": [3,4]}
    """
    data = flatten_dict(tree)

    # 编译正则表达式模式
    compiled = {re.compile(k): v for k, v in patterns.items()}

    output = {}
    for k in data:
        for pattern, repl in compiled.items():
            if pattern.fullmatch(k):
                # 如果 repl 不为 None，执行正则替换；否则表示删除此键
                new_k = pattern.sub(repl, k, count=1) if repl is not None else None
                break
        else:
            # 没有匹配到任何模式时，使用原始键
            new_k = k

        if new_k is not None:
            if new_k in output:
                raise ValueError(f"Key '{new_k}' already exists in output")
            output[new_k] = data[k]

    # 验证输出结构可以正确恢复为嵌套字典。
    # 例如：同时存在 "a/b" 和 "a" 是不允许的，因为 "a" 既是一个叶子又是一个节点。
    # 检查方法：排序后，如果前一个键是后一个键的前缀（后跟 '/'），则冲突。
    names = sorted(output)
    for i in range(len(names) - 1):
        name, next_name = names[i: i + 2]
        if next_name.startswith(name + "/"):
            raise ValueError(f"Leaf '{name}' aliases a node of '{next_name}'")

    return unflatten_dict(output)


def apply_tree(
    tree: at.PyTree[T], selector: at.PyTree[S], fn: Callable[[T, S], T], *, strict: bool = False
) -> at.PyTree[T]:
    """将函数应用到字典中那些在 selector 中有对应键的元素上。

    这是 Normalize 和 Unnormalize 的核心实现函数。它实现了"选择性地对
    字典中的某些路径应用变换"的功能。

    工作方式：
        1. 将 tree 和 selector 都拍平为扁平字典
        2. 对 tree 中的每个键：
           - 如果该键也存在于 selector 中，则调用 fn(该值, selector中的对应值)
           - 如果不存在，保留原值
        3. 将结果恢复为嵌套结构

    Args:
        tree: 要变换的数据字典。
        selector: 选择器字典（包含要与变换函数 fn 一起使用的参数）。
                  只有 tree 中与 selector 有相同路径的键才会被变换。
        fn: 变换函数，接收两个参数：
            - T: tree 中对应键的值
            - S: selector 中对应键的值（通常是 NormStats 等统计信息）
            返回变换后的 T 类型值。
        strict: 如果为 True，且 selector 中某个键在 tree 中不存在，抛出异常。

    Returns:
        变换后的数据字典，保持了原始的嵌套结构。
    """
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v: T) -> T:
        if k in selector:
            return fn(v, selector[k])
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    """将数组在指定维度上填充到目标长度。

    这个函数通常用于将低维度的状态或动作向量扩展到模型要求的维度。

    Args:
        x: 输入数组。
        target_dim: 目标维度大小。
        axis: 要在哪个轴上进行填充（默认最后一个轴）。
        value: 填充值（默认 0.0）。

    Returns:
        填充后的数组。如果 x 在指定轴上的尺寸已经 >= target_dim，则不做任何操作。

    示例：
        >>> pad_to_dim(np.array([1, 2, 3]), 5, axis=-1)
        array([1, 2, 3, 0, 0])  # 在后面补了两个 0

        >>> pad_to_dim(np.ones((3, 2)), 4, axis=0)
        array([[1., 1.],
               [1., 1.],
               [1., 1.],
               [0., 0.]])  # 在行方向上补了两行 0
    """
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    return x


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """根据维度规约生成布尔掩码（mask）。

    这个函数的主要用途是简化 DeltaActions 和 AbsoluteActions 中 mask 的创建。
    正数表示该维度为 True，负数表示该维度为 False。

    示例：
        >>> make_bool_mask(2, -2, 2)
        (True, True, False, False, True, True)
        # 解释：2 个 True → 2 个 False → 2 个 True

        >>> make_bool_mask(2, 0, 2)
        (True, True, True, True)
        # 解释：2 个 True → 0 个 = 跳过 → 2 个 True（等价于 4 个 True）

    Args:
        dims: 整数序列。正数 n 表示 n 个 True，负数 -n 表示 n 个 False。

    Returns:
        由 True/False 组成的元组。
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


def _assert_quantile_stats(norm_stats: at.PyTree[NormStats]) -> None:
    """验证分位数归一化统计量的完整性。

    检查所有 NormStats 对象是否都包含 q01 和 q99 统计量，
    如果缺失则抛出明确的错误信息。

    这是 Normalize 和 Unnormalize 的辅助函数，当 use_quantiles=True 时调用。

    Args:
        norm_stats: 要检查的归一化统计量树。

    Raises:
        ValueError: 如果任何 NormStats 缺少 q01 或 q99。
    """
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f"quantile stats must be provided if use_quantile_norm is True. Key {k} is missing q01 or q99."
            )

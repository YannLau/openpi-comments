"""
============================================================
  policy.py — 策略（Policy）模块

  本文件定义了模型推理的"策略"层。在机器人学习系统中，"策略"（Policy）
  负责接收观测数据（obs，即 observation 的缩写），调用模型生成动作（actions），
  并对输入/输出进行必要的变换（transform）。

  简单来说，本模块就是整个推理流程的"胶水代码"：
    输入传感器数据 -> 输入变换 -> 模型推理 -> 输出变换 -> 最终动作

  核心类：
    - Policy：        标准策略，支持 JAX 和 PyTorch 两种后端。
    - PolicyRecorder： 装饰器模式，记录每次推理的输入/输出到磁盘，用于调试。
============================================================
"""

from collections.abc import Sequence  # 类型提示：表示序列类型（list, tuple 等）
import logging  # 日志模块，用于记录信息
import pathlib  # 跨平台路径处理
import time  # 计时，用于性能测量
from typing import Any, TypeAlias  # 类型提示工具

import flax  # JAX 的神经网络库，提供参数管理、序列化等功能
import flax.traverse_util  # 用于展平/反展平嵌套字典（如 {"a": {"b": 1}} -> {"a/b": 1}）
import jax  # Google 的自动微分框架，支持 GPU/TPU 加速
import jax.numpy as jnp  # JAX 版本的 NumPy
import numpy as np  # 标准 NumPy 库
from openpi_client import base_policy as _base_policy  # openpi 客户端的基础策略接口
import torch  # PyTorch 深度学习框架
from typing_extensions import override  # 显式标记方法覆盖，增强可读性

from openpi import transforms as _transforms  # 数据变换模块（归一化、标记化等）
from openpi.models import model as _model  # 模型基类（BaseModel, Observation, Actions）
from openpi.shared import array_typing as at  # 数组类型别名
from openpi.shared import nnx_utils  # Flax NNX（JAX 神经网络扩展）的工具函数

# ============================================================================
# 类型别名
#
# BasePolicy 是 openpi_client 中定义的基础策略接口（抽象类），
# 我们将其作为类型别名引入，方便在类型提示中使用。
# TypeAlias 是 Python 3.10+ 中 TypeAlias 注解的 typing_extensions 版本。
# ============================================================================
BasePolicy: TypeAlias = _base_policy.BasePolicy


# ============================================================================
# 类：Policy
#
# Policy 是模型推理的核心包装器。它负责：
#   1. 接收原始的观测数据（例如相机图像、机器人关节角度等）
#   2. 应用输入变换（如归一化、标记化文本提示等）
#   3. 调用模型生成动作（支持 JAX 和 PyTorch 两种后端）
#   4. 应用输出变换（如反归一化、转换动作格式等）
#   5. 返回最终的预测结果
#
# 设计理念：
#   - 将 "模型本身" 与 "推理时的前后处理" 解耦。
#     这样模型只需要关心 "归一化后的张量 -> 动作张量"，
#     而 Policy 负责处理原始数据格式。
#   - 支持两种深度学习框架（JAX 和 PyTorch），
#     通过 is_pytorch 参数切换，对上层使用透明。
# ============================================================================
"""
好问题！这涉及到 π₀ **模型架构本身的推理方式**，我来解释一下。

## 核心原因：π₀ 是生成式模型，不是确定性模型

**π₀ 用的是流匹配（Flow Matching）**，不是直接输出一个确定动作。流匹配和扩散模型（Diffusion）是同类思路：

### 推理过程就像"从一团噪声中雕刻出动作"

```
训练阶段：    学的是"如何把噪声逐步变成正确动作"的逆过程
推理阶段：    从随机噪声开始，逐步去噪，最终得到动作
              ↓
             这需要随机数来生成那个"起点噪声"
```

### 打个比方

```
传统模型（确定性）：  你输入观测 → 模型直接输出唯一动作
                   就像给一张照片 → 直接输出"这个人是张三"

π₀（生成式）：       你输入观测 → 不直接输出动作，而是：
                    1. 先生成一张随机噪声图
                    2. 然后花 N 步把噪声"雕刻"成动作
                    就像给一张照片 → 先铺满随机颜料，再逐步画成肖像画
```

### 为什么动作需要随机采样？

机器人任务通常有**多个可行解**（多模态动作分布）：
- 抓杯子可以从左边抓，也可以从右边抓
- 关节可以用不同角度达到同一个末端位置

用随机采样而不是固定输出，能让模型保持这种**多模态性**，行为更自然灵活。

### JAX 的特殊性

**JAX 是函数式框架**——所有函数必须是纯函数（无副作用），包括随机数：

| 框架          | 随机数管理                                                                  |
| ------------- | --------------------------------------------------------------------------- |
| NumPy/PyTorch | 隐式全局状态：`np.random.randn()` 自动推进内部状态                          |
| **JAX**       | **显式管理**：`key, subkey = jax.random.split(key)`，你得手动传递和更新 key |

所以即使 PyTorch 版的 π₀ 底层也需要随机数去噪，但 PyTorch 可以用隐式全局状态，**不需要在 API 层面暴露**。而 JAX 强制你显式传递和更新 RNG key，这就是为什么你在代码里看到 `self._rng` 要来回 split 的原因。

### 另外：noise 参数是外部输入噪声

代码中的可选 `noise` 参数（第 83-88 行）允许用户**自定义噪声起点**（用于可控生成或复现结果），如果没传，模型内部仍会自己生成随机噪声——还是需要 RNG。

---

**一句话总结：随机数不是用于训练时的 Dropout 之类，而是 π₀ 的流匹配架构生成动作时，从噪声出发"雕刻"出动作所必需的。**
"""


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """初始化 Policy（策略）实例。

        这里会做几件事：
          1. 保存模型引用
          2. 将多个输入/输出变换函数组合为一个流水线
          3. 根据模型类型（JAX / PyTorch）进行不同的初始化：
             - JAX 模型：需要随机数生成器（rng），并对其采样函数做 JIT 编译（即时编译，加速推理）
             - PyTorch 模型：将模型移到指定设备（CPU/GPU），并切换到评估模式

        Args:
            model:           模型实例（BaseModel 的子类），用于生成动作。
            rng:             JAX 的随机数生成器密钥，仅对 JAX 模型有效。
                             如果为 None，会自动创建一个。
            transforms:      输入变换函数列表。推理前依次应用到观测数据上。
                             例如：归一化、标记化文本、调整图像大小等。
            output_transforms: 输出变换函数列表。推理后依次应用到输出数据上。
                               例如：反归一化，将模型输出的动作还原到真实范围。
            sample_kwargs:   传递给 model.sample_actions 的额外关键字参数。
                             例如可以在这里指定采样温度、噪声强度等。
            metadata:        额外元数据，存储为字典，可通过 .metadata 属性访问。
                             可用于存储策略的版本、训练配置等信息。
            pytorch_device:  PyTorch 模型运行的设备。可选值："cpu", "cuda:0", "cuda:1" 等。
                             仅当 is_pytorch=True 时有效。
            is_pytorch:      是否为 PyTorch 模型。
                             - True  → 使用 PyTorch 后端推理
                             - False → 使用 JAX 后端推理（默认）
        """
        # ========== 保存基本配置 ==========
        self._model = model

        # compose 将多个变换函数组合成一个流水线函数。
        # 例如，compose([norm, tokenize]) 会先做 norm，再做 tokenize。
        # 输入变换：观测数据 -> 模型可接受的格式
        self._input_transform = _transforms.compose(transforms)
        # 输出变换：模型输出 -> 机器人可执行的格式
        self._output_transform = _transforms.compose(output_transforms)

        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}

        # ========== 框架相关的初始化 ==========
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            # ---------- PyTorch 分支 ----------
            # 将模型移到指定设备（CPU / GPU），提高推理速度
            self._model = self._model.to(pytorch_device)
            # 切换到评估模式：
            #   - 关闭 Dropout、BatchNorm 的训练行为（因为这些只在训练时需要）
            #   - 确保每次推理结果确定（除非显式加了随机性）
            self._model.eval()
            # 直接引用模型的采样方法（PyTorch 不需要 JIT 编译）
            self._sample_actions = model.sample_actions
        else:
            # ---------- JAX 分支 ----------
            # module_jit 会对模型采样函数进行 JIT（Just-In-Time）编译。
            # JIT 编译会将 Python 函数编译为 XLA（Accelerated Linear Algebra）计算图，
            # 在 GPU/TPU 上可以获得显著的加速效果。
            # 并且 JIT 编译会自动处理 JAX 中的函数纯化和随机数管理。
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            # 如果没有提供随机数密钥，使用密钥 0 创建一个
            # JAX 使用显式的随机数状态（而不是隐式的全局种子），
            # 这是 JAX 函数式纯化设计的一部分。
            self._rng = rng or jax.random.key(0)

    @override  # 显式声明这个方法覆盖了父类 BasePolicy 的 infer 方法
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        """执行一次策略推理（从观测到动作的完整流程）。

        工作流程：
          1. 复制输入数据（因为变换可能会在原处修改数据）
          2. 应用输入变换（如归一化、标记化）
          3. 添加批次维度（模型要求输入有 batch 维度）
          4. 转换为相应框架的张量（JAX Array 或 PyTorch Tensor）
          5. 调用模型生成动作
          6. 移除批次维度，转换回 NumPy 数组
          7. 应用输出变换（如反归一化）
          8. 添加性能计时信息

        Args:
            obs:    观测数据字典。包含机器人传感器的原始数据，
                    例如相机图像（"image"）、关节角度（"state"）、
                    语言指令（"prompt"）等。
            noise:  可选的噪声输入。用于某些采样策略：
                    - 流匹配模型（π₀, π₀.₅）：从随机噪声逐步去噪得到动作
                    - 如果提供，会覆盖 sample_kwargs 中的 noise 参数

        Returns:
            dict: 包含以下键的结果字典：
                - "state":          原始状态输入（透传，不做处理）
                - "actions":        模型预测的动作
                - "policy_timing":  推理耗时统计（毫秒）
                - 以及其他输出变换添加的字段
        """
        # ====================================================================
        # 第 1 步：复制输入数据
        #
        # jax.tree.map(lambda x: x, obs) 会对字典/列表/元组等"树结构"进行
        # 逐叶子复制。这里的 lambda x: x 看似是恒等函数，但由于 JAX 数组是
        # 不可变的，这个复制主要是为了确保嵌套容器结构是新的（浅拷贝叶子，
        # 但深拷贝容器结构）。对于 NumPy 数组，这不会创建新的内存副本。
        #
        # 为什么要复制？因为变换函数可能会修改输入字典的某些字段（如添加新键），
        # 但我们不希望影响调用者传入的原始字典。
        # ====================================================================
        inputs = jax.tree.map(lambda x: x, obs)
        # 应用输入变换流水线，将原始观测数据转换为模型可接受的格式
        inputs = self._input_transform(inputs)

        # ====================================================================
        # 第 2 步：添加批次维度并转换为正确的张量类型
        #
        # 模型（无论是 JAX 还是 PyTorch）都期望输入带有批次维度（batch dimension），
        # 即使我们一次只推理一个样本。np.newaxis 在最前面添加一个长度为 1 的维度。
        # 例如：(3, 256, 256) -> (1, 3, 256, 256)
        # ====================================================================
        if not self._is_pytorch_model:
            # ---------- JAX 分支 ----------
            # jnp.asarray(x) 将输入转换为 JAX Array（如果还不是的话）
            # [np.newaxis, ...] 在最前面添加批次维度
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

            # JAX 使用函数式随机数生成：每次采样前都需要"分裂"（split）出新的密钥。
            # jax.random.split(self._rng) 返回 (new_rng, sample_key) 两个密钥，
            # 我们更新 self._rng 供下次使用，将 sample_key 传给采样函数。
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # ---------- PyTorch 分支 ----------
            # torch.from_numpy(np.array(x))：先确保是 NumPy 数组，再转为 PyTorch Tensor
            # .to(self._pytorch_device)：将张量移到正确的设备（CPU/GPU）
            # [None, ...]：添加批次维度
            inputs = jax.tree.map(
                lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...],
                inputs,
            )
            # PyTorch 没有 JAX 那样的随机数状态管理，这里直接存设备名，
            # 之后传给采样函数（采样函数内部会处理 PyTorch 的随机数生成器）
            sample_rng_or_pytorch_device = self._pytorch_device

        # ====================================================================
        # 第 3 步：准备 sample_kwargs（采样参数字典）
        #
        # _sample_kwargs 是在 __init__ 时传入的固定参数。
        # 这里复制一份（dict(self._sample_kwargs)），防止修改原始字典。
        # ====================================================================
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            # 用户传入了自定义噪声，用于控制采样过程。
            # 例如，在流匹配模型中，噪声是去噪过程的起点。
            if self._is_pytorch_model:
                noise = torch.from_numpy(noise).to(self._pytorch_device)
            else:
                noise = jnp.asarray(noise)

            # 如果噪声是 (action_horizon, action_dim) 的二维数组，
            # 说明没有批次维度，需要手动添加。
            # action_horizon：预测的动作序列长度（例如一次预测 16 步动作）
            # action_dim：每个动作的维度（例如 7 维关节角度）
            if noise.ndim == 2:
                noise = noise[None, ...]  # (H, D) -> (1, H, D)

            sample_kwargs["noise"] = noise

        # ====================================================================
        # 第 4 步：执行模型推理
        #
        # Observation.from_dict(inputs) 将字典转换为 Observation 对象。
        # Observation 是一个 dataclass，包含 precomputed（预计算特征）、
        # tokens（标记化文本）、mask（注意力掩码）、images（图像）等字段，
        # 是模型的标准化输入格式。
        # ====================================================================
        observation = _model.Observation.from_dict(inputs)

        # 开始计时，测量模型推理时间
        start_time = time.monotonic()

        # 调用模型采样动作：
        #   self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
        #   - 对于 JAX：sample_rng_or_pytorch_device 是随机数密钥，用于采样
        #   - 对于 PyTorch：sample_rng_or_pytorch_device 是设备名，告诉模型在哪个设备上运行
        #
        # 返回的动作形状通常是 (1, action_horizon, action_dim)，
        # 其中 1 是批次维度。
        outputs = {
            # "state" 透传原始的模型输入状态，不做处理
            "state": inputs["state"],
            # "actions" 是模型预测的动作序列
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }

        # 计算推理耗时（秒），后面会转换为毫秒
        model_time = time.monotonic() - start_time

        # ====================================================================
        # 第 5 步：后处理 — 移除批次维度，转回 NumPy 数组
        #
        # 模型输出带有批次维度 [0, ...]（第一个样本），
        # 而我们通常只需要单样本的结果，所以要移除这个维度。
        # ====================================================================
        if self._is_pytorch_model:
            # PyTorch 分支：
            #   x[0, ...]    — 取批次中的第一个（也是唯一一个）样本
            #   .detach()    — 从计算图中分离，不再跟踪梯度（推理时不需要梯度）
            #   .cpu()       — 从 GPU 移到 CPU（NumPy 无法直接操作 GPU 张量）
            #   np.asarray() — 转为 NumPy 数组
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            # JAX 分支：
            #   x[0, ...]    — 取批次中的第一个样本
            #   np.asarray() — JAX Array -> NumPy Array（触发设备到主机的传输）
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        # ====================================================================
        # 第 6 步：应用输出变换并添加性能指标
        #
        # 输出变换的典型用途：
        #   - Unnormalize：反归一化，将模型输出的归一化动作还原到真实范围
        #     （例如，模型输出范围 [-1, 1] -> 真实关节角度范围 [0, 180]）
        #   - DeltaActions -> AbsoluteActions：将增量动作转换为绝对位置
        # ====================================================================
        outputs = self._output_transform(outputs)

        # 添加性能计时信息，单位从秒转换为毫秒
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }

        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        """返回策略的元数据。

        元数据可以包含策略版本、训练配置、模型架构等信息，
        由调用者在创建 Policy 时通过 metadata 参数传入。
        """
        return self._metadata


# ============================================================================
# 类：PolicyRecorder
#
# 这个类使用"装饰器模式"（Decorator Pattern）包装另一个 Policy。
# 它的职责是：
#   1. 将每次推理的输入（obs）和输出（actions）记录到磁盘
#   2. 透明地将推理请求转发给被包装的策略
#
# 使用场景：
#   - 调试：在实际机器人上运行前，记录所有推理数据以便分析
#   - 数据收集：为后续训练收集高质量的"观测-动作"对
#   - 复现问题：当机器人行为异常时，记录的数据可以帮助定位问题
#
# 设计模式说明：
#   PolicyRecorder 和被包装的策略实现相同的接口（BasePolicy），
#   所以对调用者来说完全透明。
#   你可以把 PolicyRecorder 看作是策略的"日志代理"。
# ============================================================================
class PolicyRecorder(_base_policy.BasePolicy):
    """将策略的行为记录到磁盘。

    每次调用 infer() 时，除了执行原始推理外，
    还会将输入和输出保存为 .npy 文件。
    """

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        """初始化 PolicyRecorder。

        Args:
            policy:     要包装和记录的底层策略。
            record_dir: 记录文件的输出目录。如果目录不存在，会自动创建。
        """
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        # mkdir(parents=True) 会递归创建目录，等价于 mkdir -p
        # exist_ok=True 表示如果目录已存在，不会报错
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0  # 推理步数计数器

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        """执行推理并记录数据。

        先执行原始策略的推理（得到预测的动作），
        然后将输入和输出一起保存到文件。

        Args:
            obs: 观测数据字典。

        Returns:
            与原始策略返回相同的结果字典。
        """
        # 先进行实际的推理，获取结果
        results = self._policy.infer(obs)

        # ========== 记录数据到磁盘 ==========
        # 将输入和输出打包到一个字典中
        data = {"inputs": obs, "outputs": results}

        # flatten_dict 将嵌套字典展平，用 "/" 分隔键路径。
        # 例如：{"a": {"b": 1, "c": 2}} -> {"a/b": 1, "a/c": 2}
        # 这样做的好处是文件格式更简单，不需要处理嵌套结构。
        data = flax.traverse_util.flatten_dict(data, sep="/")

        # 生成输出文件路径：{record_dir}/step_{step}.npy
        # 每次推理创建一个独立的文件
        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        # 将数据保存为 NumPy 的 .npy 格式
        # np.asarray(data) 将字典转换为 NumPy 的 void 数组，
        # 这样可以用标准的 np.load 加载回来
        np.save(output_path, np.asarray(data))

        # 返回原始推理结果（对调用者透明）
        return results

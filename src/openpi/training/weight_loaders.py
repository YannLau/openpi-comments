# ============================================================
#  src/openpi/training/weight_loaders.py — 权重加载器
#
#  本模块定义了模型权重的加载策略，支持从不同来源加载
#  预训练权重，并与随机初始化的参数合并。
#
#  核心概念：
#    - WeightLoader 协议：所有加载器遵循的统一接口
#    - 部分加载：只加载匹配的权重，不匹配的键保留随机初始化值
#    - 合并策略：加载的权重优先，匹配 missing_regex 的缺失键
#      从参考参数中补充
#
#  支持的加载器类型：
#    1. NoOpWeightLoader — 空操作（全部随机初始化）
#    2. CheckpointWeightLoader — 从 openpi 训练的检查点恢复
#    3. PaliGemmaWeightLoader — 从官方 PaliGemma 预训练权重加载
# ============================================================

import dataclasses  # 用于创建冻结（不可变）数据类
import logging      # 日志
import re           # 正则表达式（用于匹配缺失的键名模式）
from typing import Protocol, runtime_checkable  # 类型协议与运行时检查

import flax.traverse_util  # Flax 树形结构遍历（flatten_dict / unflatten_dict）
import numpy as np         # 科学计算

import openpi.models.model as _model            # 模型定义（restore_params 等）
import openpi.shared.array_typing as at          # 数组类型标注
import openpi.shared.download as download        # 文件下载工具（支持 GCS / HTTP / 本地）

logger = logging.getLogger(__name__)  # 当前模块的日志器


@runtime_checkable
class WeightLoader(Protocol):
    """权重加载器协议 —— 所有加载器必须实现 load() 方法。

    核心设计：
      - load() 接收一个"参考参数结构"（params），返回一个与它结构完全相同的 dict
      - 返回的 dict 中，已加载的键使用加载的权重，未加载的键保留参考参数的随机初始值
      - 这使得我们可以"部分加载"：只加载预训练模型中的部分参数（如 PaliGemma 主干网络），
        而其他参数（如 action expert）保持随机初始化

    参数：
      params: 模型的参考参数结构。这是一个嵌套的 dict/tree，其中的 array-like 对象
              描述了参数的形状和 dtype。通常来自 jax.eval_shape 或 nnx.state()。

    返回：
      与 params 结构完全相同的参数 dict。已加载的参数使用加载的真实权重值，
      未加载的部分合并了 params 中的对应值。
    """

    def load(self, params: at.Params) -> at.Params:
        """加载模型权重。

        Args:
            params: 模型的参数结构（嵌套的 array-like 对象树）。

        Returns:
            加载后的参数。结构必须与 `params` 完全相同。
            如果只加载了部分参数，加载器必须将加载的参数与 `params` 合并。
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    """空操作权重加载器 —— 不加载任何外部权重。

    适用场景：
      - 从头开始训练（不使用预训练权重）
      - 调试或测试时快速启动
      - 所有参数均随机初始化

    load() 直接将传入的 params 原样返回，不做任何操作。
    这意味着模型的所有参数都将保持随机初始化状态，
    或从检查点恢复（由训练循环的其他部分处理）。
    """

    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """从 openpi 训练的检查点加载完整权重。

    适用于两类检查点目录：
      1. 本地训练的检查点：
         "./checkpoints/<config>/<exp>/<step>/params"
         使用本地训练产生的模型参数。

      2. 发布的预训练检查点：
         "gs://openpi-assets/checkpoints/<model>/params"
         从 Google Cloud Storage（GCS）下载官方发布的权重。

    加载策略：
      1. 从指定路径加载所有参数（以 numpy 数组格式）
      2. 使用 missing_regex=".*lora.*" 处理 LoRA 权重：
         - 如果检查点中没有 LoRA 权重（预训练模型通常没有），
           则保留随机初始化的 LoRA 权重
         - 如果检查点中有 LoRA 权重（微调后的模型），则使用加载的权重
    """

    params_path: str  # 检查点路径（本地路径或 GCS URL）

    def load(self, params: at.Params) -> at.Params:
        # maybe_download: 如果路径是 GCS URL，先下载到本地缓存
        # restore_params: 从检查点目录恢复参数，restore_type=np.ndarray 表示以 numpy 格式加载
        #（后续训练代码会将其转换为 JAX 数组并分片到设备）
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)

        # 与参考参数合并：对检查点中缺失的 LoRA 权重（匹配 .*lora.*），保留随机初始化值
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """从官方 PaliGemma 预训练检查点加载权重。

    PaliGemma 是 Google 发布的多模态视觉-语言模型（基于 SigLIP + Gemma），
    作为 π₀ 模型的视觉-语言主干网络。此加载器从官方 GCS 存储桶下载
    PaliGemma 权重，并与 π₀ 模型的参数结构合并。

    关键设计：
      - 只覆盖与 PaliGemma 名称匹配的权重
      - 保持所有额外权重不变（如 action expert、LoRA 适配器等）
      - missing_regex=".*" 表示参考参数中任意不存在的键都从 params 补充
        （因为 PaliGemma 权重只覆盖模型的一部分）

    注意：PaliGemma 官方权重以扁平化的 .npz 格式存储，需要先
    用 unflatten_dict 恢复为嵌套结构。
    """

    def load(self, params: at.Params) -> at.Params:
        # 从 Google Vertex AI 的公开存储桶下载 PaliGemma 224px 预训练权重
        # gs={"token": "anon"} 表示使用匿名访问（无需认证）
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )

        # 加载 .npz 文件（键为扁平化的路径字符串，如 "vit/encoder/block_0/mlp/kernel"）
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))

        # 将扁平化的键还原为嵌套字典结构
        # 例如 "vit/encoder/block_0/mlp/kernel" → {"vit": {"encoder": {"block_0": {"mlp": {"kernel": ...}}}}}
        # 然后用 ["PaliGemma"]["params"] 将其放到模型根命名空间下
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}

        # 与参考参数合并：所有在参考参数中存在但加载权重中不存在的键，从参考参数中补充
        # missing_regex=".*" 表示所有缺失键都补充（因为 PaliGemma 只覆盖模型的一部分）
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """将加载的权重与参考参数合并，生成完整的参数集。

    这是权重加载的核心算法，采用两阶段合并策略：

    阶段一 —— 加载优先：
      遍历加载的权重中的每个键，如果该键在参考参数中也存在，
      则使用加载的权重值（并转换为参考参数的目标 dtype）。

    阶段二 —— 缺失补充：
      遍历参考参数的所有键，找出满足以下条件的键：
        1. 匹配 missing_regex 模式
        2. 不在阶段一的结果中
      将这些缺失的键从参考参数中复制到结果中。

    这样做的原因：
      - 预训练权重通常只覆盖模型的一部分（如视觉编码器 + 语言模型）
      - 模型的其余部分（如 action expert、LoRA 权重）需要保留随机初始化
      - missing_regex 控制哪些"缺失"的部分需要保留随机值

    参数：
      loaded_params: 从外部加载的权重（可能只覆盖模型的一部分）
      params:        参考参数（包含模型所有参数的结构，未初始化的或随机初始化的）
      missing_regex: 正则表达式，匹配那些加载权重中缺失但需要从参考参数中补充的键

    返回：
      合并后的完整参数字典。

    示例：
      假设参考参数有键：["a/kernel", "b/kernel", "lora_a/kernel"]
      加载的权重有键：["a/kernel"]
      如果 missing_regex=".*lora.*"，则结果为：
        {"a/kernel": 加载值, "b/kernel": 参考值, "lora_a/kernel": 参考值}
      （注意：b/kernel 虽匹配 .* 但不匹配 .*lora.*，因此不会被补充）
    """
    # 将嵌套参数压平为扁平键（以 "/" 分隔），便于逐键操作
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # ---- 阶段一：加载的权重优先 ----
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            # 如果加载值的 dtype 与参考参数不同，转换为参考参数的 dtype
            # 例如加载的权重可能是 float32，但参考参数期望 bfloat16
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v
    # 释放加载权重的内存（不再需要）
    flat_loaded.clear()

    # ---- 阶段二：补充缺失的权重 ----
    # 编译正则表达式模式
    pattern = re.compile(missing_regex)

    # 遍历参考参数的所有键，找出既是 missing_regex 匹配的、又不在 result 中的键
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]  # 保留参考参数中的值（通常为随机初始化值）

    # 将扁平结果恢复为嵌套字典结构
    return flax.traverse_util.unflatten_dict(result, sep="/")

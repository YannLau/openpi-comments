# =============================================================================
# 训练工具模块
#
# 本模块定义了 VLA（视觉-语言-动作）模型训练过程中最核心的数据结构
# 和日志辅助函数：
#
#   1. TrainState —— 训练状态的"总管家"，包含了训练所需的一切
#      （模型参数、优化器状态、学习率规划器、EMA 参数等）
#   2. tree_to_info / array_tree_to_info —— 将 JAX 的树结构
#      （PyTree）转换为人类可读的字符串，用于日志记录
#
# 【为什么需要 TrainState？】
# 传统深度学习框架（如 PyTorch）中，模型参数是"可变"的——你可以直接
# 修改 model.weight 的值。但在 JAX 中，一切都是"函数式"的：
# 参数是不可变的数组，更新参数意味着"返回一个新版本"，而不是修改旧版本。
# 因此 JAX 需要一个显式的数据结构来"打包"所有需要维护的状态，
# 这就是 TrainState 的职责。
#
# 这个概念最早来自 Flax 的 train_state 模块，本项目的 TrainState
# 在 Flax 的基础上增加了 EMA（指数移动平均）支持。
# =============================================================================

# ---------------------------------------------------------------------------
# 导入标准库
# ---------------------------------------------------------------------------
from collections.abc import Callable  # Callable 类型：任何可调用对象（函数、方法、类等）
from typing import Any  # Any 类型：可以是任何值（Python 中一切皆对象，Any 就是"随便啥都行"）

# ---------------------------------------------------------------------------
# 导入第三方库 —— JAX 生态
# ---------------------------------------------------------------------------
from flax import nnx  # Flax NNX：JAX 的新一代神经网络库（"Neural Network X"）
# NNX 引入了"变量"（Variable）的概念，让你在编程时感觉像 PyTorch 一样
# 用 .param 访问参数，但在底层仍然是纯函数式的 JAX 计算。
from flax import struct  # Flax 的数据类：类似于 Python 的 dataclasses，但专门针对 JAX 设计
import jax  # JAX 核心库
import optax  # JAX 的优化器库（类似于 PyTorch 的 torch.optim）

# ---------------------------------------------------------------------------
# 导入项目内部模块
# ---------------------------------------------------------------------------
from openpi.models import model as _model  # 模型定义：BaseModel（基础模型类）
from openpi.shared import array_typing as at  # 类型标注工具（提供带 shape/dtype 信息的类型注解）


# =============================================================================
# 数据类：TrainState（训练状态）
#
# 【作用】
# 这是训练过程中"所有需要持久化维护的东西"的集合体。
# 想象你在做一顿大餐——TrainState 就像你的"料理备忘录"：
# 记录了当前做到哪一步了（step）、配料还剩多少（params）、
# 火候怎么调（opt_state）、用什么手法做（tx）、
# 以及有没有备用的秘方（ema_params）。
#
# 使用 @struct.dataclass 而不是 Python 原生的 @dataclasses.dataclass，
# 是因为 Flax 的版本能更好地与 JAX 的 PyTree 机制整合。
#
# 【PyTree 是什么？】
# PyTree 是 JAX 中的一个核心概念。简单说，它就是"可以嵌套的、叶子节点是
# JAX 数组的树状结构"。比如：
#
#   {"encoder": {"w": array(...), "b": array(...)},
#    "decoder": {"w": array(...), "b": array(...)}}
#
# 就是一个 PyTree。JAX 可以自动"遍历"这棵树，把变换（如梯度裁剪、参数更新）
# 应用到所有叶子节点上。
#
# 【什么是 EMA？】
# EMA = Exponential Moving Average（指数移动平均）
# 训练时，模型参数的每一次更新都会有"噪声"。EMA 就像给参数加了低通滤波器：
#   ema_params = 0.999 * ema_params + 0.001 * current_params
# 这样得到的参数比当前的"瞬时参数"更平滑，在推理时通常表现更好。
# 你可以在很多顶级模型（如 YOLO、Diffusion 模型）中看到这个技巧。
#
# 【字段说明】
#   step      训练步数计数器（第几千步了）
#   params    模型的所有可学习参数（权重、偏置等），以 nnx.State 形式保存
#   model_def 模型的"蓝图"——即网络结构定义。它可以让你从参数重建出完整模型
#   opt_state 优化器的内部状态（如 Adam 的动量项 m_t 和 v_t）
#   tx        优化器变换函数（如 Adam、SGD 等）——注意它不是"数据节点"，不参与树结构
#   ema_decay EMA 衰减系数（一般取 0.999 或 0.9999），None 表示不使用 EMA
#   ema_params EMA 平滑后的参数副本，与 params 结构完全一致
# =============================================================================
@at.typecheck  # 装饰器：启用本项目中自定义的运行时类型检查（详见 array_typing 模块）
@struct.dataclass  # Flax 的数据类装饰器，自动生成 __init__、__eq__ 等方法
class TrainState:
    # ----- 核心训练状态 -----

    step: at.Int[at.ArrayLike, ""]  # 当前训练步数（标量）
    # at.Int[at.ArrayLike, ""] 的含义：
    #   - at.Int:      整数类型
    #   - ArrayLike:   可以是 JAX 数组、numpy 数组或纯 Python 数字
    #   - "":          空字符串表示"标量"（0 维数组）
    # 这一步数在分布式训练中会在所有设备间同步，确保每个设备都知道"现在跑到第几步了"

    params: nnx.State  # 模型参数（nnx.State 是 NNX 中管理变量状态的容器）
    # nnx.State 是一个嵌套的字典结构，包含模型所有 Layer 的权重和偏置。
    # 例如：
    #   {
    #     "encoder": {"kernel": array(shape=[768, 3072]), "bias": array(shape=[3072])},
    #     "decoder": {"kernel": array(shape=[3072, 768]), "bias": array(shape=[768])}
    #   }

    model_def: nnx.GraphDef[_model.BaseModel]  # 模型定义（"蓝图"）
    # GraphDef 是 NNX 中"去掉参数后的模型结构"——它只包含网络的计算图结构，
    # 不包含实际权重。你可以这样理解：
    #   - params 是"食材"（权重值）
    #   - model_def 是"菜谱"（网络结构）
    # 两者结合就能"烹饪"出完整的模型。
    #
    # 使用泛型 [_model.BaseModel] 表示这个 GraphDef 是从 BaseModel 派生出来的，
    # 类型系统可以据此推断出精确的类型。

    opt_state: optax.OptState  # 优化器状态（Optimizer State）
    # 像 Adam 这类优化器会为每个参数维护额外的统计数据：
    #   - 动量（momentum）：记录历史梯度的指数加权平均
    #   - 二阶矩（variance）：记录历史梯度平方的指数加权平均
    # 这些在恢复训练时必须和参数一起保存，否则优化器会"失忆"，导致训练不稳定。
    # OptState 的类型会根据具体使用的优化器（Adam、SGD 等）而变化。

    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    # 优化器变换函数（Gradient Transformation）
    # optax 的核心理念：优化器 = 一系列"变换"的组合（链式调用）
    # 例如 AdamW 可以被拆解为：
    #   scale_by_adam() + add_decayed_weights() + scale(-lr)
    #
    # 【重点：pytree_node=False】
    # 这个设置告诉 JAX："tx 是一个普通的 Python 对象（函数链），
    # 不是 JAX 数组，请不要把它当作 PyTree 的节点来处理。"
    # 当你调用 jax.tree_util.tree_map() 时，tx 会被跳过。
    # 这是正确的，因为你绝不会对一个优化器函数做梯度计算或参数更新。
    #
    # 可以这样理解：params 和 opt_state 是"数据"，需要被 JAX 处理；
    # 而 tx 是"代码逻辑"，不需要被 JAX 处理。

    # ----- EMA（指数移动平均）参数 -----

    ema_decay: float | None = struct.field(pytree_node=False)
    # EMA 衰减系数，通常取 0.999 或 0.9999
    #   - 0.999：每步更新，新参数贡献 0.1% 权重，历史贡献 99.9%
    #   - 越大，平滑效果越强，但跟踪当前参数的速度越慢
    #   - None：不使用 EMA
    #
    # 同样标记为 pytree_node=False，因为这是一个超参数（标量），不是可微张量。

    ema_params: nnx.State | None = None
    # EMA 平滑后的参数副本
    # 结构完全等同于 params，但值是经过指数移动平均"平滑"后的版本。
    # 训练过程中，每次更新 params 后，也会更新 ema_params：
    #   ema_params = ema_decay * ema_params + (1 - ema_decay) * params
    # 推理（部署到机器人）时如果使用 ema_params，结果往往更稳定可靠。


# =============================================================================
# 函数：tree_to_info（将 PyTree 转换为可读字符串）
#
# 【作用】
# 将一个任意的 JAX PyTree（嵌套的字典/列表/元组结构，叶子节点是数组）
# 转换为人类可读的多行字符串，用于日志记录和调试。
#
# 例如，一个这样的 PyTree：
#   {"encoder": {"w": array([1, 2, 3]), "b": array([0.5])}}
#
# 经过 tree_to_info 会变成：
#   encoder.w: 1, 2, 3
#   encoder.b: 0.5
#
# 【参数】
#   tree       : 任意 JAX PyTree
#   interp_func: 将叶子节点的值转换为字符串的函数
#                默认是 str()，即直接打印数值
#
# 【返回值】
#   多行字符串，每行格式为 "路径: 值"
# =============================================================================
@at.typecheck
def tree_to_info(tree: at.PyTree, interp_func: Callable[[Any], str] = str) -> str:
    """将一个 PyTree 展开为人类可读的字符串，方便日志记录。
    可选地传入 interp_func 来把叶子节点的值转换成更有意义的字符串。
    """
    # ---- 第一步：展平 PyTree ----
    #
    # tree_flatten_with_path 会将嵌套的 PyTree 展平为扁平的（路径, 值）对列表。
    # 路径本身也是 PyTree，但每个"路径节点"代表树中的一级键或索引。
    #
    # 输入：{"encoder": {"w": array([1, 2, 3]), "b": array([0.5])}}
    # 输出：
    #   [((DictKey("encoder"), DictKey("w")), array([1, 2, 3])),
    #    ((DictKey("encoder"), DictKey("b")), array([0.5]))]
    #          ↑ 路径是一个元组，从根节点到叶子的每一级键
    #
    # tree_flatten_with_path 返回了 (flat_tree, tree_def) 两个值，
    # 但我们只需要 flat_tree，所以用 _ 忽略 tree_def。
    tree, _ = jax.tree_util.tree_flatten_with_path(tree)

    # ---- 第二步：将路径转换为美观的字符串 ----
    #
    # keystr(path) 会把路径元组转换成点分隔的字符串：
    #   (DictKey("encoder"), DictKey("w")) → "encoder.w"
    #   (DictKey("encoder"), DictKey("b")) → "encoder.b"
    #
    # interp_func(value) 把值转为字符串，默认就用 str()
    return "\n".join(f"{jax.tree_util.keystr(path)}: {interp_func(value)}" for path, value in tree)


# =============================================================================
# 函数：array_tree_to_info（将数组 PyTree 转换为可读字符串）
#
# 【作用】
# tree_to_info 的特化版本，专门用于"所有叶子节点都是 JAX 数组"的 PyTree。
# 打印时不是输出数组的具体数值（那会刷屏），而是输出每个数组的形状和数据类型。
#
# 例如，输入参数字典：
#   {"encoder.w": array(shape=[768, 3072], dtype=float32),
#    "encoder.b": array(shape=[3072], dtype=float32)}
#
# 输出：
#   encoder.w: (768, 3072)@float32
#   encoder.b: (3072,)@float32
#
# 这在训练开始时打印模型结构非常有用——你可以一目了然地看到
# 每个参数的名字、形状和精度，而不被具体数值淹没。
# =============================================================================
@at.typecheck
def array_tree_to_info(tree: at.PyTree) -> str:
    """将一个由数组组成的 PyTree 转换为人类可读的日志字符串。"""
    return tree_to_info(
        tree,
        # 自定义解释函数：输出形状（shape）和数据类型（dtype）
        # 格式示例: (768, 3072)@float32
        lambda x: f"{x.shape}@{x.dtype}",
    )

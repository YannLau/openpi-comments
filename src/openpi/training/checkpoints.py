# =============================================================================
# 检查点（Checkpoint）管理模块
#
# 本模块负责机器人视觉-语言-动作（VLA）模型训练过程中的：
#   1. 检查点目录初始化（新建 / 覆盖 / 恢复）
#   2. 训练状态的保存（保存模型参数、优化器状态、归一化统计量）
#   3. 训练状态的恢复（从磁盘加载，继续训练）
#   4. 归一化统计量的独立加载（用于推理/部署）
#
# 【什么是检查点？】
# 训练一个深度学习模型通常需要数小时到数天。检查点就是在训练过程中
# "拍一张快照"，把当前模型的所有权重、优化器状态（动量等）存到磁盘上。
# 这样如果机器崩溃了，或者你想中途停下来看看效果，都可以从最近的快照
# 恢复继续训练，而不必从头开始。
#
# 本模块使用 Google 的 orbax 库来管理检查点。orbax 是 JAX 生态系统中
# 专门为高性能 ML 训练设计的检查点框架，支持异步保存、并行恢复等特性。
# =============================================================================

# ---------------------------------------------------------------------------
# 导入标准库
# ---------------------------------------------------------------------------
from __future__ import annotations  # 允许在类型注解中使用字符串形式的类名（如 "TrainState"），避免循环导入问题

import asyncio  # 异步 I/O 库，用于非阻塞地执行耗时操作
import concurrent.futures as futures  # 线程池/进程池相关，用于并行执行任务
import dataclasses  # 数据类（@dataclass），Python 3.7 引入的轻量级结构体定义方式
import logging  # 日志模块，替代 print() 进行结构化输出
from typing import (
    Protocol,
)  # 类型系统中的"协议"——类似接口（interface），定义"只要实现了 __call__ 方法就可以当作回调使用"

# ---------------------------------------------------------------------------
# 导入第三方库 —— orbax（检查点）和 JAX
# ---------------------------------------------------------------------------
from etils import epath  # Google 的路径库，增强 pathlib，支持 GCS（Google Cloud Storage）等云存储路径
import jax  # JAX 核心库：高性能数值计算，支持 GPU/TPU 自动微分
import orbax.checkpoint as ocp  # orbax 检查点库：负责模型参数的序列化/反序列化
import orbax.checkpoint.future as future  # orbax 的 Future 类型，用于异步保存的同步原语

# ---------------------------------------------------------------------------
# 导入项目内部模块
# ---------------------------------------------------------------------------
from openpi.shared import array_typing as at  # 类型标注工具，提供 Params 等类型别名；训练/推理时被禁用（避免性能开销）
import openpi.shared.normalize as _normalize  # 归一化（标准化）工具：计算和加载数据集的均值/方差等统计量
import openpi.training.data_loader as _data_loader  # 数据加载器：负责从磁盘/云端读取训练数据
import openpi.training.utils as training_utils  # 训练工具函数：定义了 TrainState（训练状态数据结构）等


# =============================================================================
# 函数：initialize_checkpoint_dir （初始化检查点目录）
#
# 【作用】
# 准备一个目录来存放训练过程中保存的检查点文件。这个函数会处理三种场景：
#   - 目录不存在 → 新建
#   - 目录已存在且用户要求覆盖（--overwrite）→ 删除后重建
#   - 目录已存在且用户要求恢复（--resume）→ 尝试继续训练
#
# 【参数】
#   checkpoint_dir : 检查点保存路径（本地路径或 GCS 路径）
#   keep_period    : 每隔多少步保留一次检查点。如果为 None，那么只保留最新的
#   overwrite      : 如果目录已存在，是否强制覆盖（删除旧数据）
#   resume         : 如果目录已存在，是否尝试从中恢复训练
#
# 【返回值】
#   (CheckpointManager, bool) —— 检查点管理器对象，以及"是否正在恢复训练"的标记
# =============================================================================
def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str, *, keep_period: int | None, overwrite: bool, resume: bool
) -> tuple[ocp.CheckpointManager, bool]:
    # ---- 第一步：处理目录存在的三种情况 ----
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()  # 将字符串/相对路径转为绝对路径
    resuming = False  # 默认不是恢复模式

    if checkpoint_dir.exists():
        # 场景 A：用户要求覆盖 —— 删除旧目录，重新创建（彻底重来）
        if overwrite:
            checkpoint_dir.rmtree()  # 递归删除整个目录树
            checkpoint_dir.mkdir(parents=True, exist_ok=True)  # 重新创建目录（parents=True 会自动创建中间目录）
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")

        # 场景 B：用户要求恢复 —— 标记为恢复模式
        elif resume:
            resuming = True  # 标记为"正在恢复"，后面会尝试加载已有的检查点

        # 场景 C：目录存在但用户既没要求覆盖也没要求恢复 —— 报错退出，避免意外覆盖
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    # 如果上面删除了旧目录重建，这里 dir 已经存在了，mkdir 不会报错（exist_ok=True）
    # 如果是新路径，这里创建目录
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ---- 第二步：创建 CheckpointManager（检查点管理器）----
    #
    # CheckpointManager 是 orbax 的核心类，负责协调检查点的保存和恢复。
    # 它可以管理多种"项"（items），每种项有自己的保存/恢复策略：
    #
    #   "assets"（资产）    → 使用 CallbackHandler（自定义回调处理器）
    #                          保存归一化统计量等非模型参数数据
    #   "train_state"      → 使用 PyTreeCheckpointHandler（JAX 树结构处理器）
    #                         保存完整的训练状态（优化器状态、学习率调度器等）
    #   "params"（参数）    → 使用 PyTreeCheckpointHandler
    #                         单独保存模型参数（可用于推理，不需要完整训练状态）
    #
    # 【为什么要把参数单独存一份？】
    # 推理（部署到机器人上）时只需要模型参数，不需要优化器状态。
    # 单独存一份 params 让推理代码只需加载几 MB 的参数文件，而不用管整个训练状态。
    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": CallbackHandler(),  # 自定义回调句柄，见下方定义
            "train_state": ocp.PyTreeCheckpointHandler(),  # JAX PyTree 句柄，可以保存任意嵌套的 JAX 数组结构
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,  # 磁盘上最多保留 1 个完整的检查点（节省空间）
            keep_period=keep_period,  # 每隔 N 步额外保留一份（不被 max_to_keep 清理）
            create=False,  # 不自动创建目录（我们已经手动创建了）
            async_options=ocp.AsyncOptions(timeout_secs=7200),  # 异步操作超时：2 小时（深层网络保存可能很慢）
        ),
    )

    # ---- 第三步：处理"目录存在但没有有效检查点"的特殊情况 ----
    #
    # 场景：用户训练到第 5 步，还没保存检查点就中断了。目录里是空的，
    # 此时如果 resume=True，代码会尝试恢复训练，但目录里什么都没有，
    # 恢复操作会失败报错。
    #
    # 解决方案：检查管理器里的所有步骤号。如果为空或者只有第 0 步（初始化步骤，
    # 通常不包含有效参数），就取消恢复模式，当作从头开始训练。
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False  # 取消恢复标记，让训练从第 0 步重新开始

    return mngr, resuming


# =============================================================================
# 函数：save_state （保存训练状态到检查点）
#
# 【作用】
# 将当前训练步的完整状态保存到磁盘，包括：
#   1. 模型参数（权重）
#   2. 优化器状态（如 Adam 的动量、二阶矩估计）
#   3. 归一化统计量（数据集的均值、标准差等）
#
# 关联阅读：save_state 如何被调用？详见训练主循环中的周期性保存逻辑。
#
# 【参数】
#   checkpoint_manager : 检查点管理器（上面初始化好的）
#   state              : 当前的训练状态（TrainState），包含模型参数、优化器状态等
#   data_loader        : 数据加载器，从中获取数据配置（用于获取归一化统计量）
#   step               : 当前的训练步数（第几千步/第几万步）
# =============================================================================
def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
):
    # ---- 第一步：保存"资产"（归一化统计量等附属数据）----
    #
    # 这里定义了一个嵌套函数 save_assets，它会在适当的时机被调用（由 CallbackHandler 调用）。
    # 【什么是归一化统计量？】
    # 训练时我们会把数据（图像像素值、机器人关节角度等）减去均值再除以标准差，
    # 让数据分布更稳定，训练更快。这些均值和标准差就是"归一化统计量"。
    # 推理时必须使用训练时相同的统计量，否则模型行为会不正确。
    def save_assets(directory: epath.Path):                                                             #！！！！！这里其实实现了，Callback接口，因为是鸭子类型。
        data_config = data_loader.data_config()  # 获取数据配置（包含数据集路径、变换规则等）
        norm_stats = data_config.norm_stats  # 归一化统计量（均值、标准差等字典）
        if norm_stats is not None and data_config.asset_id is not None:
            # 将归一化统计量保存到 assets/{asset_id}/ 目录下
            # asset_id 通常是一个标识符，如 "aloha_sim" 或 "libero"
            _normalize.save(directory / data_config.asset_id, norm_stats)

    # ---- 第二步：分离训练参数和推理参数 ----
    #
    # 训练状态中包含 EMA（指数移动平均）参数 —— 这是对模型参数做平滑后的版本，
    # 在推理时通常比原始参数效果更好。
    #
    # _split_params 会把 EMA 参数（如果有）从训练状态中剥离出来，单独保存为 params 项。
    # 如果不使用 EMA，那就把当前参数直接保存。
    #
    # at.disable_typechecking() 暂时关闭类型检查，因为 JAX 的 PyTree 结构在运行时会
    # 发生变化，静态类型标注无法准确描述，关闭可以避免警告/错误。
    with at.disable_typechecking():
        train_state, params = _split_params(state)

    # ---- 第三步：组织保存项并写入磁盘 ----
    items = {
        "assets": save_assets,  # 资产：保存归一化统计量的回调函数
        "train_state": train_state,  # 训练状态：优化器状态、步数等
        "params": {"params": params},  # 模型参数（可独立用于推理）
    }
    checkpoint_manager.save(step, items)  # 异步保存到磁盘（不会阻塞训练继续执行太多）


# =============================================================================
# 函数：restore_state （从检查点恢复训练状态）
#
# 【作用】
# 从之前保存的检查点中恢复训练状态，这样可以从中断的地方继续训练，
# 而不是从头开始。
#
# 【参数】
#   checkpoint_manager : 检查点管理器
#   state              : 当前训练状态（刚初始化好，还不包含实际训练出来的权重）
#                        我们用它来"搭骨架"——告诉 orbax 从哪个结构（shape/dtype）恢复
#   data_loader        : 数据加载器（此函数中未使用，保留参数是为了接口统一）
#   step               : 要恢复到第几步。如果为 None，则恢复到最新的检查点
#
# 【返回值】
#   恢复后的训练状态（TrainState），可以被训练循环直接使用
# =============================================================================
def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
) -> training_utils.TrainState:
    # 注意：此函数目前没有使用 data_loader，但保留参数是为了保持与 save_state 对称的接口
    # 将来可能会需要 data_loader 来恢复数据加载器的状态（如数据集迭代位置）
    del data_loader  # 显式标记未使用，避免 linter 警告

    with at.disable_typechecking():
        # 分离参数：和保存时一样的逻辑，保持结构一致
        train_state, params = _split_params(state)

        # 从磁盘恢复：
        # orbax 会读取之前保存的 "train_state" 和 "params" 两项，
        # 然后把数据填入我们提供的骨架（train_state 和 params 的结构）
        restored = checkpoint_manager.restore(
            step,  # 要恢复的步数（None = 最新步）
            items={
                "train_state": train_state,
                "params": {"params": params},
            },
        )

    # 恢复完成后，将分离的参数合并回完整的训练状态
    # _merge_params 是 _split_params 的逆操作
    return _merge_params(restored["train_state"], restored["params"])


# =============================================================================
# 函数：load_norm_stats （加载归一化统计量）
#
# 【作用】
# 从检查点目录中独立加载归一化统计量。这个函数通常用于推理/部署场景：
# 当你只想用训练好的模型做预测（而不是训练），你只需要模型参数 + 归一化统计量，
# 不需要完整的训练状态。
#
# 【参数】
#   assets_dir : 资产目录路径（通常是检查点目录）
#   asset_id   : 资产标识符（如 "aloha_sim"），对应保存时的名称
#
# 【返回值】
#   归一化统计量字典，Key 是数据字段名（如 "state", "action"），
#   Value 是 NormStats 对象，包含 mean（均值）和 std（标准差）
# =============================================================================
def load_norm_stats(assets_dir: epath.Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    norm_stats_dir = epath.Path(assets_dir) / asset_id  # 拼接路径：assets_dir/asset_id
    norm_stats = _normalize.load(norm_stats_dir)  # 从该目录加载统计量
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


# =============================================================================
# Callback 协议类型（Protocol）
#
# Python 的 Protocol（协议）定义了"一组方法签名"。
# 任何实现了 __call__(directory) 方法的对象都可以被视为 Callback。
# 这与 Java/C# 的"接口"概念类似，但 Python 采用的是"鸭子类型"（duck typing）：
# "如果它走路像鸭子、叫起来像鸭子，那它就是鸭子"——只要对象有需要的方法就行。
# =============================================================================
class Callback(Protocol):
    """回调函数协议：接收一个路径参数，执行任意操作（通常是将数据保存到该路径下）。"""

    def __call__(self, directory: epath.Path) -> None: ...


# =============================================================================
# 类：CallbackHandler （回调处理器）
#
# 【作用】
# 这是 orbax 框架的一个自定义处理器，支持以异步方式调用任意回调函数。
# orbax 原生支持保存/恢复 JAX 的 PyTree 结构（模型参数），
# 但对于"保存归一化统计量"这种非标准数据，就需要自定义处理器。
#
# 继承自 ocp.AsyncCheckpointHandler，这意味着它的保存操作是异步的——
# 不会阻塞训练循环的主线程。
#
# 【为什么需要这个类？】
# orbax 的 CheckpointManager.save() 会自动决定何时调用 save() 方法。
# 当我们保存 "assets" 项时，我们不是直接传数据，而是传一个函数（save_assets），
# CallbackHandler 负责在适当的时机调用这个函数。
# =============================================================================
class CallbackHandler(ocp.AsyncCheckpointHandler):
    """一个用于异步调用任意函数的检查点处理器。仅用于保存，不支持恢复。"""

    def save(self, directory: epath.Path, args: CallbackSave):
        """同步保存：只在主进程（process_index == 0）上执行回调函数。

        在多 GPU/多 TPU 训练中，有多个进程同时运行。如果每个进程都保存一次
        归一化统计量，就会造成重复写入。因此只让进程 0（主进程）执行保存。
        """
        if jax.process_index() == 0:  # 只在主进程中执行（分布式训练时避免重复）
            args.callback(directory)  # 调用用户传入的回调函数（如 save_assets）

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        """异步保存：把同步的 save() 包装到线程中，返回 Future 让 orbax 可以等待完成。

        asyncio.to_thread 会把 save() 放到一个单独的线程中执行，这样主线程可以
        继续做其他事情（如准备下一批训练数据）。
        """
        return [
            future.CommitFutureAwaitingContractedSignals(
                asyncio.to_thread(self.save, directory, args)  # 在线程池中运行 save()
            )
        ]

    def restore(self, *args, **kwargs):
        """不支持恢复：归一化统计量的恢复由 load_norm_stats 函数独立处理。

        这里显式抛出 NotImplementedError，防止意外调用。
        """
        raise NotImplementedError("CallbackHandler does not support restore")


"""
Agent 的搜索结果和我的分析完全一致。简要总结：

**这四个类 (`Callback`, `CallbackHandler`, `CallbackSave`, `CallbackRestore`) 在整个代码库中没有任何外部引用**——它们完全私有于 `checkpoints.py`。

整个项目的其他文件只 import 了 `checkpoints` 的四个**函数**：

| 文件                                   | import 了什么                                              |
| -------------------------------------- | ---------------------------------------------------------- |
| `scripts/train.py`                     | `initialize_checkpoint_dir`, `restore_state`, `save_state` |
| `src/openpi/policies/policy_config.py` | `load_norm_stats`                                          |

从来没有人直接 import `CallbackHandler`、`CallbackSave` 等。

但这不意味着它们是死代码——`CallbackHandler` 和 `CallbackSave` 通过 **orbax 的注册机制** (`@ocp.args.register_with_handler`) 在运行时被框架间接使用，所以你看到"零引用"是正常的：

```python
# orbax 在 checkpoint_manager.save(step, items) 时，
# 内部会查找 "assets" 对应的 handler → CallbackHandler
# 再查找 CallbackHandler 注册的 save args 类型 → CallbackSave
# 然后自动包装参数并调用 handler.save()
item_handlers={"assets": CallbackHandler()}
```

`CallbackRestore` 严格来说是死代码——`CallbackHandler.restore()` 直接抛 `NotImplementedError`，这行注册只是为了满足 orbax "每个 handler 必须同时注册 save 和 restore" 的接口契约。
"""




"""
 # 类：CallbackHandler （回调处理器）                                                                                                                                                                                                  
  #                                                                                                                                                                                                                                     
  # 【作用】                                                                                                                                                                                                                            
  # 这是 orbax 框架的一个自定义处理器，支持以异步方式调用任意回调函数。                                                                                                                                                                 
  # orbax 原生支持保存/恢复 JAX 的 PyTree 结构（模型参数），                                                                                                                                                                            
  # 但对于"保存归一化统计量"这种非标准数据，就需要自定义处理器。  
  # 
如何理解，我也没看到pi0模型去自定义处理器去保存归一化统计量这种非标准数据啊 ?

好问题——这句话确实写得容易让人困惑。让我澄清一下：

## `CallbackHandler` **就是** pi0 的自定义处理器

你找不到其他自定义处理器，**因为 `CallbackHandler` 就是 pi0 定义的那唯一一个**，它就在 `checkpoints.py` 第 302 行。文档里的"就需要自定义处理器"指的就是它自己。

## 为什么需要这个自定义处理器？

核心原因在于 **orbax 的数据类型限制**：

| 数据类型                   | orbax 原生支持？             | 存储示例                           |
| -------------------------- | ---------------------------- | ---------------------------------- |
| JAX 数组 (权重参数)        | ✅ `PyTreeCheckpointHandler` | `state.params`, `state.ema_params` |
| Python 字典                | ✅ `PyTreeCheckpointHandler` | 优化器状态                         |
| **任意的 Python 回调函数** | ❌ **不支持**                | 无标准序列化器                     |

归一化统计量本身虽然是字典，**可以通过 `PyTreeCheckpointHandler` 保存**。但这里的设计意图不同：

```python
# checkpoints.py 第 172-178 行
def save_assets(directory: epath.Path):
    data_config = data_loader.data_config()     # 1. 从 data_loader 获取配置
    norm_stats = data_config.norm_stats         # 2. 从配置中取统计量
    if norm_stats is not None and data_config.asset_id is not None:
        _normalize.save(directory / data_config.asset_id, norm_stats)  # 3. 保存
```

注意 `save_assets` 是一个**闭包**——它捕获了 `data_loader` 外部变量。这种"运行时闭包函数"：
- ❌ `PyTreeCheckpointHandler` 不认识它（它只认 JAX 数组）
- ❌ `pickle` 不一定能序列化它（依赖捕获的环境）
- ✅ **`CallbackHandler`** 的方式是："我不序列化你的函数，我只在**保存的时刻调用它**"

## orbax 的处理流程

```
checkpoint_manager.save(step, items)
  │
  ├── "train_state" → PyTreeCheckpointHandler  → 序列化 JAX 数组到磁盘
  ├── "params"      → PyTreeCheckpointHandler  → 序列化 JAX 数组到磁盘
  └── "assets"      → CallbackHandler           → 调用 save_assets(directory)，让它自己去写文件
```

`CallbackHandler` 本质上是一个 **"执行器"而非"序列化器"**——它不关心你要存什么、怎么存，只是在合适的时机调用你给它的函数，由那个函数自己去处理文件 I/O。

## 一句话总结

不是"pi0 需要自定义处理器来保存归一化统计量"，而是 **"pi0 选择用回调函数的方式保存归一化统计量，所以写了 `CallbackHandler` 来执行这个回调"，仅此一处，别无分店。**
"""





"""
# 一句话核心

**`checkpoint_manager.save(step, items)` 需要给每个 item key 分配一个 handler 来处理保存。`train_state` 和 `params` 存的是 JAX 数组，
用 orbax 自带的 `PyTreeCheckpointHandler`。`assets` 存的是一个函数（`save_assets`），JAX 不认识函数，
没内置 handler 能处理它——所以自己写了个 `CallbackHandler`，它的 save 逻辑就是一句话：调用这个函数。**

```python
# 第 305-312 行  CallbackHandler.save():
def save(self, directory, args):
    args.callback(directory)  # ← 全部逻辑就这一行：执行你给我的函数
```

而 `save_assets` 函数做的事情（第 172-178 行）：
```python
def save_assets(directory):
    norm_stats = data_loader.data_config().norm_stats  # 从 data_loader 取统计量
    _normalize.save(directory / asset_id, norm_stats)   # 写到 assets/ 目录下
```

**所以整个 `CallbackHandler` + `CallbackSave` 这几十行代码，本质上就是在 orbax 的框架里开了一个"执行函数"的通道：不序列化任何东西，只是到点了调用你塞进来的函数去干活。**
"""





"""
对，你说得完全正确。完整链路是这样的：

```
save_assets(directory: epath.Path) -> None
         ↑ 符合 Callback 协议（第 280 行）
         │   def __call__(self, directory: epath.Path) -> None
```

```python
# 第 348 行  CallbackSave 把它包一层
CallbackSave(callback=save_assets)

# 第 305 行  CallbackHandler.save() 把它解开执行
args.callback(directory)

# 等价于直接调用
save_assets(directory)
```

**所以整个链条就是：`save_assets`（一个普通的嵌套函数）→ 满足 `Callback` 协议 → 被 `CallbackSave` 包装 → 
orbax 内部把它传给 `CallbackHandler.save()` → 最后 `args.callback(directory)` 还原调用。**

你可以把 `CallbackSave` 看作一个"包装盒"，它存在的唯一目的就是让 orbax 框架能统一处理各种 item 的参数：无论是 `PyTreeCheckpointHandler` 的数组参数，
还是 `CallbackHandler` 的函数参数，都包装成统一的 `CheckpointArgs` 子类。
"""


# =============================================================================
# 数据类：CallbackSave （回调保存参数）
#
# 【作用】
# 这个类描述了在保存时需要传递给 CallbackHandler 的数据 ——
# 即一个回调函数（如 save_assets）。
#
# @ocp.args.register_with_handler 告诉 orbax："当保存 'assets' 类型的数据时，
# 使用 CallbackHandler 来处理，参数类型是 CallbackSave。"
#
# @dataclasses.dataclass 让 Python 自动生成 __init__()、__repr__() 等方法
# =============================================================================
@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    """用于保存的回调参数，包装了一个 Callback 函数。"""

    callback: Callback  # 要执行的回调函数


# =============================================================================
# 数据类：CallbackRestore （回调恢复参数）
#
# 【作用】
# 注册恢复参数类型（当前为空），用于保证 orbax 框架的接口完整性。
# 虽然 CallbackHandler 不支持恢复，但在 orbax 的架构中，
# 每个处理器都需要同时注册 save 和 restore 两种参数类型。
# =============================================================================
@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs):
    """用于恢复的回调参数（当前为空，因为 CallbackHandler 不支持恢复）。"""

    ...


# =============================================================================
# 内部函数：_split_params （分离参数）
#
# 【作用】
# 将完整的训练状态拆分为"训练状态"和"推理参数"两部分。
# 这样做的目的是：
#   1. 训练状态包含了优化器参数（动量、自适应学习率历史等）—— 体积大，但用于继续训练
#   2. 推理参数只包含模型权重 —— 体积小，适合部署
#
# 【什么是 EMA 参数？】
# EMA = Exponential Moving Average（指数移动平均）
# 在训练过程中，不仅保存当前参数，还在内存中维护一个"参数的平均值"。
# 这个平均值像是一个"慢速跟踪"的版本，通常比最新的参数更稳定、泛化能力更好。
# 推理时使用 EMA 参数往往能得到更好的效果。
#
# 【参数】
#   state : 完整的训练状态
#
# 【返回值】
#   (train_state, params) 元组：
#     train_state : 不含推理参数/EMA 参数的训练状态
#     params      : 用于推理的模型参数
# =============================================================================
def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    """
    分离策略：
    - 如果存在 EMA 参数（state.ema_params is not None）：
      使用 EMA 参数作为推理参数，将训练状态中的 EMA 参数清空
    - 如果不存在 EMA 参数：
      使用当前参数作为推理参数，将训练状态中的当前参数清空
    """
    if state.ema_params is not None:
        # 使用 EMA 平滑后的参数（推理效果更好）
        params = state.ema_params
        # 创建新的训练状态副本，但把 ema_params 设为 None（已单独取出来了）
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        # 不使用 EMA，直接用当前最新参数
        params = state.params
        # 创建新的训练状态副本，但把 params 设为空字典（已单独取出来了）
        train_state = dataclasses.replace(state, params={})
    return train_state, params


# =============================================================================
# 内部函数：_merge_params （合并参数）
#
# 【作用】
# _split_params 的逆操作。将分离的"训练状态"和"推理参数"重新合并为一个
# 完整的训练状态对象。
#
# 当从检查点恢复时，_split_params 把参数分离出来让 orbax 去填充数据，
# orbax 填充完成后，再调用 _merge_params 把两部分合并回去。
# =============================================================================
def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    """合并参数到训练状态。

    判断逻辑：
    - 如果训练状态中已经有 params 字段（非空）→ 说明拆分时使用 EMA，因此恢复时也回到 ema_params
    - 如果训练状态中 params 字段为空 → 说明拆分时用的是当前参数，恢复时回到 params
    """
    if train_state.params:
        # 训练状态中已有参数，说明拆分时用的是 EMA → 合并回 ema_params
        return dataclasses.replace(train_state, ema_params=params["params"])
    # 拆分时用的是当前参数 → 合并回 params
    return dataclasses.replace(train_state, params=params["params"])

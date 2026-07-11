"""
============================================================
  serve_policy.py — 策略推理服务器启动脚本

  这个脚本是整个 openpi 系统的"上线入口"。
  它的职责是：
    1. 加载一个训练好的策略（Policy）
    2. 启动一个 WebSocket 服务器
    3. 持续接收机器人发来的观测数据，返回预测动作

  工作流程（从命令行到机器人推理）：
    终端命令 → 解析参数 → 加载模型 → 启动 WebSocket 服务器
                           ↓
                    等待机器人连接...
                           ↓
                    收到观测（obs）→ 推理（infer）→ 返回动作

  典型用法：
    # 使用默认策略（ALOHA 仿真环境）
    uv run scripts/serve_policy.py

    # 使用指定检查点
    uv run scripts/serve_policy.py \
      policy:checkpoint \
      policy.config=pi05_libero \
      policy.dir=checkpoints/pi05_libero/my_exp/20000

  WebSocket 协议细节：
    服务器使用 msgpack（MessagePack，一种高效的二进制序列化格式）
    进行数据传输，比 JSON 更紧凑、更快，适合实时机器人控制。
============================================================
"""

import dataclasses  # 数据类装饰器，用于定义配置结构体
import enum  # 枚举类型，用于表示有限的选项集合
import logging  # 日志模块，在控制台输出运行状态
import socket  # 网络工具，用于获取主机名和 IP 地址

import tyro  # 参数解析库（比 argparse 更好用，支持 dataclass 自动解析）

from openpi.policies import policy as _policy  # Policy 类和 PolicyRecorder 装饰器
from openpi.policies import policy_config as _policy_config  # 策略构造工厂函数
from openpi.serving import websocket_policy_server  # WebSocket 策略服务器
from openpi.training import config as _config  # 训练配置管理（按名称获取配置）


# ============================================================================
# 枚举：EnvMode（环境模式）
#
# 枚举（Enum）是一种类型安全的常量定义方式。
# 在这里，我们列出支持的机器人平台/仿真环境。
# 每个环境都有自己独特的观测空间（摄像头数量、关节数量等）和动作空间。
#
# 为什么用枚举而不是字符串？
#   - 类型安全：写错了 IDE 会报错，不会等到运行时才发现
#   - 自动补全：IDE 会提示可选值
#   - 统一管理：所有可用环境一目了然
# ============================================================================
class EnvMode(enum.Enum):
    """支持的机器人平台和仿真环境。"""

    ALOHA = "aloha"  # 真实的 ALOHA 机器人（双臂协作平台）
    ALOHA_SIM = "aloha_sim"  # ALOHA 仿真环境（MuJoCo 模拟器）
    DROID = "droid"  # DROID 数据集中的机器人平台
    LIBERO = "libero"  # LIBERO 仿真基准测试套件


# ============================================================================
# 数据类：Checkpoint（检查点参数）
#
# 这个类定义了"从特定检查点加载策略"所需的参数。
# 与 Default 类配合使用，形成参数解析的两种模式。
#
# tyro 库利用 Python 的 dataclass 来定义命令行参数结构。
# Checkpoint 和 Default 都有 name() 方法（由 @dataclass 自动生成），
# tyro 用它们来区分用户想要哪种模式。
#
# 例如：
#   python serve_policy.py policy:checkpoint --policy.config=pi0_aloha_sim ...
#     → 解析为 Checkpoint(config="pi0_aloha_sim", dir="...")
#   python serve_policy.py
#     → 解析为 Default()（默认值）
# ============================================================================
@dataclasses.dataclass
class Checkpoint:
    """从训练好的检查点加载策略。"""

    # 训练配置名称（对应 config.py 中 _CONFIGS 列表里的配置名）
    # 例如："pi0_aloha_sim"、"pi05_libero"、"pi05_droid"
    config: str

    # 检查点目录路径
    # 可以是本地路径或 GCS 远程路径
    # 例如："checkpoints/pi0_aloha_sim/exp/10000" 或 "gs://bucket/checkpoint"
    dir: str


# ============================================================================
# 数据类：Default（默认策略）
#
# 这是一个"空壳"类，没有任何字段。
# 它的存在只是为了表示"我不想指定检查点，用系统默认的就行"。
# 这是一种常见的"标签"（Tag）模式——用类型来区分不同的行为。
# ============================================================================
@dataclasses.dataclass
class Default:
    """使用当前环境的默认策略。"""


# ============================================================================
# 数据类：Args（主参数）
#
# 这是脚本的顶层参数结构，包含所有可配置项。
# tyro 会从命令行参数自动填充这个类的字段。
#
# 字段类型提示中的联合类型（如 Checkpoint | Default）让 tyro 实现了
# "子命令"（subcommand）的效果。
#
# 命令行用法示例：
#   # 完全默认（ALOHA 仿真环境）
#   uv run scripts/serve_policy.py
#
#   # 指定环境和默认提示
#   uv run scripts/serve_policy.py --env=libero --default_prompt="pick up the cube"
#
#   # 带自定义检查点
#   uv run scripts/serve_policy.py --policy:checkpoint --policy.config=pi05_libero ...
#
#   # 启用记录模式
#   uv run scripts/serve_policy.py --record
#
#   # 指定端口
#   uv run scripts/serve_policy.py --port=8080
# ============================================================================
@dataclasses.dataclass
class Args:
    """serve_policy 脚本的命令行参数。"""

    # 目标环境。仅在使用默认策略时需要指定。
    # 如果使用自定义检查点（policy:checkpoint），环境由检查点的配置决定。
    env: EnvMode = EnvMode.ALOHA_SIM

    # 默认文本提示（语言指令）。
    # 当输入数据中缺少 "prompt" 字段时使用这个值。
    # 例如："pick up the cube"、"place the object in the bin"
    # 如果为 None 且输入也没有 prompt，部分模型可能无法正常工作。
    default_prompt: str | None = None

    # WebSocket 服务器监听端口。默认 8000。
    port: int = 8000

    # 是否记录策略的推理过程（输入+输出）。
    # 启用后，每次 infer() 的输入输出都会被保存到磁盘，
    # 用于调试和性能分析。
    # 记录文件保存在当前目录的 policy_records/ 文件夹中。
    record: bool = False

    # 策略加载方式。支持两种子模式：
    #   - Checkpoint: 从已知路径加载特定检查点
    #   - Default:    使用环境的默认检查点（从 GCS 拉取）
    # 默认为 Default（自动使用默认策略）。
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# ============================================================================
# 全局映射表：DEFAULT_CHECKPOINT
#
# 这是一个"环境 → 默认检查点"的查找表。
# 每个环境都有一个默认的预训练检查点，存储在 Google Cloud Storage (GCS) 上。
#
# 为什么不把这些写死在代码里？
#   - 用户可以随时更新到新版本，而不用改脚本
#   - 不同环境使用不同的基础模型（π₀ vs π₀.₅），这样才能获得最佳效果
#
# GCS 路径说明：
#   "gs://openpi-assets/checkpoints/..." 是公开的预训练模型仓库。
#   openpi 团队会定期发布新版本的预训练模型。
#   首次使用时，download.maybe_download() 会自动下载并缓存到本地。
# ============================================================================
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    # ALOHA 真实机器人 — 使用 π₀.₅（改进版流匹配模型）
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    # ALOHA 仿真环境 — 使用 π₀（标准流匹配模型）
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    # DROID 机器人平台 — 使用 π₀.₅
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    # LIBERO 仿真基准 — 使用 π₀.₅
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


# ============================================================================
# 函数：create_default_policy
#
# 根据环境类型创建默认策略。
# 这个函数是"快速上手"场景的核心——用户只需要指定环境，
# 系统会自动选择最优的预训练模型和检查点。
# ============================================================================
def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """为指定环境创建一个默认策略。

    流程：
      1. 在 DEFAULT_CHECKPOINT 中查找环境对应的检查点信息
      2. 利用 policy_config.create_trained_policy() 加载模型

    Args:
        env:            目标环境（ALOHA、ALOHA_SIM、DROID、LIBERO）。
        default_prompt: 可选默认文本指令。

    Returns:
        配置好的 Policy 对象，可直接调用 infer()。

    Raises:
        ValueError: 如果环境中 DEFAULT_CHECKPOINT 中没有对应的检查点配置。
    """
    # ":=" 是海象运算符（walrus operator），
    # 在 if 条件中同时完成赋值和判断。
    # 相当于：
    #   checkpoint = DEFAULT_CHECKPOINT.get(env)
    #   if checkpoint is not None:
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config),  # 按名称获取训练配置
            checkpoint.dir,  # 检查点路径（GCS 或本地）
            default_prompt=default_prompt,
        )
    raise ValueError(f"Unsupported environment mode: {env}")


# ============================================================================
# 函数：create_policy
#
# 根据命令行参数创建策略。
# 支持两种模式：
#   - Checkpoint 模式：用户指定了检查点路径和配置名
#   - Default 模式：使用预定义的默认检查点
#
# Python 3.10 的 match-case（模式匹配）让这种分支逻辑非常清晰。
# 它比传统的 if-elif-else 更具可读性，特别是当分支条件涉及类型判断时。
# ============================================================================
def create_policy(args: Args) -> _policy.Policy:
    """根据命令行参数创建策略。"""
    match args.policy:
        # 情况 1：用户提供了自定义检查点（policy:checkpoint 子命令）
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config),
                args.policy.dir,
                default_prompt=args.default_prompt,
            )
        # 情况 2：用户没有指定检查点，使用默认策略
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


# ============================================================================
# 函数：main（程序主入口）
#
# 这是脚本的核心流程编排函数。
# 它按顺序执行以下几个步骤：
#   1. 创建策略（加载模型和变换流水线）
#   2. 可选：用 PolicyRecorder 包装策略用于调试记录
#   3. 创建 WebSocket 服务器
#   4. 启动服务器循环（等待并处理客户端连接）
# ============================================================================
def main(args: Args) -> None:
    """脚本主函数。

    Args:
        args: 解析后的命令行参数。
    """
    # ========================================================================
    # 第 1 步：创建策略
    #
    # 这一步会触发：
    #   - GCS 文件下载（如果是首次使用默认检查点）
    #   - 模型权重加载（JAX 或 PyTorch）
    #   - 归一化统计信息加载
    #   - 变换流水线组装
    #
    # 整个过程可能需要几秒到几分钟（取决于网络速度和模型大小）。
    # ========================================================================
    policy = create_policy(args)

    # 保存策略元数据，稍后传给服务器
    # 元数据包含配置信息，客户端可以用来了解策略的版本和能力
    policy_metadata = policy.metadata

    # ========================================================================
    # 第 2 步：可选 — 启用推理记录
    #
    # PolicyRecorder 是一个"装饰器"（Decorator），它包裹在原始 policy 外面。
    # 每次 infer() 被调用时，它会：
    #   1. 正常执行推理
    #   2. 把输入和输出保存到磁盘（policy_records/ 目录）
    #   3. 返回结果
    #
    # 相当于给推理过程加了一个"黑匣子"（类似飞机的飞行记录仪）。
    # 开启后需要注意：
    #   - 每次推理都会写磁盘，可能会降低性能
    #   - 长时间运行会产生大量文件，需要定期清理
    # ========================================================================
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    # ========================================================================
    # 第 3 步：获取本机网络信息并启动 WebSocket 服务器
    #
    # socket.gethostname() 获取本机的主机名（如 "robot-pc"）
    # socket.gethostbyname() 将主机名解析为 IP 地址（如 "192.168.1.100"）
    #
    # 这些信息会打印到日志，方便用户确认服务器地址和端口。
    # ========================================================================
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    # ========================================================================
    # 第 4 步：启动 WebSocket 服务器
    #
    # WebsocketPolicyServer 是一个基于 asyncio 的 WebSocket 服务器。
    #
    # 配置说明：
    #   - host="0.0.0.0"   → 监听所有网络接口
    #                        这样局域网内的机器人客户端都能连接
    #   - port=args.port   → 监听指定的端口
    #
    # serve_forever() 是一个阻塞调用，意味着：
    #   - 服务器启动后，这个函数不会返回
    #   - 服务器会持续运行，直到被 Ctrl+C 中断或收到终止信号
    #
    # 服务器的工作循环：
    #   1. 等待 WebSocket 连接
    #   2. 接收 msgpack 编码的观测数据
    #   3. 调用 policy.infer(obs) 进行推理
    #   4. 返回 msgpack 编码的动作结果
    #   5. 等待下一次请求
    #
    # 消息格式（msgpack）：
    #   - 客户端 → 服务器：观测字典（图像、状态、提示）
    #   - 服务器 → 客户端：动作字典（预测的动作 + 时间戳）
    # ========================================================================
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",  # 绑定到所有网络接口
        port=args.port,
        metadata=policy_metadata,  # 元数据，客户端可以通过 /metadata 端点获取
    )
    server.serve_forever()  # 永不停机的服务器循环


# ============================================================================
# 脚本入口点
#
# 当这个文件作为主程序运行时（python serve_policy.py），
# Python 会将 __name__ 变量设置为 "__main__"。
#
# tyro.cli(Args) 会：
#   1. 解析命令行参数（sys.argv）
#   2. 根据 Args 的类型注解自动生成 --help 文档
#   3. 创建 Args 实例
#   4. 将其传给 main() 函数
#
# logging.basicConfig(level=logging.INFO) 配置日志输出：
#   - level=logging.INFO：显示 INFO 级别及以上的日志（INFO, WARNING, ERROR）
#   - force=True：强制覆盖任何已有的日志配置
#   这样用户就能在控制台看到模型的加载进度等信息。
# ============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))

"""
**你的猜测完全正确。** 看一下两边的数据格式对比就清楚了：

## LeRobot 数据集（训练时）

LeRobot 的字段命名是它自己的约定：

```python
# config.py:634-639  LeRobotAlohaDataConfig.repack_transforms
RepackTransform({
    "images": {"cam_high": "observation.images.top"},   # ← LeRobot: "observation.images.top"
    "state": "observation.state",                        # ← LeRobot: "observation.state"
    "actions": "action",                                 # ← LeRobot: "action"（单数）
})
```

需要 `repack` 把 LeRobot 的名字翻译成统一格式。

## 推理客户端（推理时）

看 `make_aloha_example()`：

```python
# aloha_policy.py:12-21
return {
    "state": np.ones((14,)),
    "images": {"cam_high": ..., "cam_low": ..., ...},  # ← 已经是统一命名
    "prompt": "do something",
}
```

**客户端直接按 `AlohaInputs.__call__()` 期望的格式发送数据，字段名就是统一的 `"images"`、`"state"`、`"prompt"`。** 不需要任何重命名。

## 核心原因

| 方向 | 数据来源             | 键名格式                                                      | 需要 repack？  |
| ---- | -------------------- | ------------------------------------------------------------- | -------------- |
| 训练 | LeRobot 数据集       | `"observation.images.top"`, `"observation.state"`, `"action"` | ✅ 必须 repack |
| 推理 | 机器人客户端直接发送 | `"images.cam_high"`, `"state"`, `"actions"`                   | ❌ 不需要      |

**推理时假设客户端已经知道 openpi 的统一格式，按这个格式传数据。这就相当于约定了"你就按这个结构给我"，不需要中间的翻译层。** 
而 `create_trained_policy` 那个 `repack_transforms` 参数只是为特殊场景留的"后门"——
如果你的客户端就是非要用 LeRobot 格式传，你可以自己传一个 repack 进去。 """

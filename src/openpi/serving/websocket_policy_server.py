"""
WebSocket 策略推理服务器

本模块实现了 WebsocketPolicyServer —— 一个通过 WebSocket 协议提供策略推理服务的"模型服务器"。
它是 openpi 分布式推理架构中的服务端组件，与客户端 （WebsocketClientPolicy） 配合使用。

架构设计：
  这是典型的"远程推理"（Remote Inference）模式 —— 将计算密集的模型推理放在
  服务器上，机器人控制端只做轻量级的传感器读取和动作执行：

  ┌── 机器人工控机（CPU） ──┐        WebSocket         ┌── GPU 推理服务器 ────────┐
  │                         │    (msgpack 序列化)      │                          │
  │  Runtime 主循环         │ ◄───────────────────────► │  WebsocketPolicyServer   │
  │    │                    │    obs (含图像/状态)      │    │                     │
  │    ▼                    │ ────────────────────────► │    ▼                    │
  │  PolicyAgent            │                           │  Policy.infer(obs)      │
  │    │                    │    action (含动作)        │    │                     │
  │    ▼                    │ ◄──────────────────────── │    ▼                     │
  │  ActionChunkBroker      │                           │  模型前向传播            │
  │    │                    │                           │  (JAX / PyTorch)        │
  │    ▼                    │                           │                          │
  │  WebsocketClientPolicy  │                           │                          │
  └─────────────────────────┘                           └──────────────────────────┘

为什么要用 WebSocket 而不是 HTTP/REST？
  1. 双向通信：HTTP 是"请求-响应"模式，每次都需要客户端发起请求。
     WebSocket 建立连接后，两端都可以随时发送数据。
  2. 低延迟：WebSocket 在建立连接后，头部开销只有 2 字节（HTTP 请求头几百字节）。
     对于 50Hz 的高频控制（每 20ms 一次推理），这个差异很显著。
  3. 持久连接：无需为每次推理重新建立 TCP 连接，减少握手时间。
  4. 全双工：服务器可以主动推送数据（如状态更新、参数变化通知）。

协议设计（消息交换顺序）：
  步骤    客户端                             服务器
  ─────────────────────────────────────────────────────
  1       连接 WebSocket                  ← 接受连接
  2       ← 接收元数据消息                   发送元数据（策略信息、复位位姿等）
  3       发送观测（obs）                 → 接收观测，执行推理
  4       ← 接收动作结果（action）           发送推理结果
  5       重复步骤 3-4（推理循环）         → 持续服务
  6       断开连接                         ← 检测断开，清理

与客户端的关系：
  服务端对应客户端：openpi_client/websocket_client_policy.py
  通信格式：openpi_client/msgpack_numpy.py —— 支持 NumPy 数组的 msgpack 序列化

运行方式：
  # 方式 1：作为脚本直接运行（通过 serve_policy.py）
  uv run scripts/serve_policy.py pi0_aloha_sim --policy.dir=/path/to/checkpoint

  # 方式 2：编程方式调用
  server = WebsocketPolicyServer(policy=my_policy, host="0.0.0.0", port=8000)
  server.serve_forever()  # 永久运行
"""

import asyncio  # Python 异步 I/O 框架 —— 用于处理多个 WebSocket 连接
import http  # HTTP 状态码和协议相关常量
import logging  # 日志系统 —— 输出服务器运行状态
import time  # 时间库 —— 用于性能计时（推理耗时统计）
import traceback  # 异常回溯 —— 用于将服务器端的错误详细信息发送给客户端

# ── openpi 策略接口与序列化 ──
from openpi_client import base_policy as _base_policy  # 策略基类接口（infer + reset）
from openpi_client import msgpack_numpy  # 支持 NumPy 数组的 msgpack 序列化

# ── WebSocket 服务器库 ──
import websockets.asyncio.server as _server  # WebSocket 异步服务器
import websockets.frames  # WebSocket 帧类型（如 CloseCode 枚举）

# 当前模块的日志记录器
# 使用 __name__（即 "openpi.serving.websocket_policy_server"）作为日志器名称，
# 方便在日志配置中按模块过滤日志级别。
logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """WebSocket 策略推理服务器。

    这个类将任何 BasePolicy 实现暴露为一个网络服务，使远程客户端可以
    通过 WebSocket 连接调用 policy.infer() 进行推理。

    它本身不做任何模型推理 —— 它只是把收到的观测通过 msgpack 反序列化，
    传给传入的 policy 对象，然后把推理结果序列化后返回给客户端。

    服务器的核心循环（每个连接）：
      ┌───────────────────────────────────────────────────────────────┐
      │  _handler(websocket)                                          │
      │    │                                                           │
      │    ├─ 1. 发送元数据（连接建立后第一条消息）                      │
      │    │    包含：reset_pose（复位位姿）、模型名称等                   │
      │    │                                                           │
      │    ├─ 2. 进入推理循环                                          │
      │    │    ┌───────────────────────────────────────────┐          │
      │    │    │ while True:                                │          │
      │    │    │   接收客户端消息 → unpackb(obs)           │          │
      │    │    │   policy.infer(obs) → action              │          │
      │    │    │   添加推理耗时（server_timing）            │          │
      │    │    │   发送结果给客户端 packb(action)          │          │
      │    │    └───────────────────────────────────────────┘          │
      │    │                                                           │
      │    └─ 3. 错误处理或连接关闭                                      │
      └───────────────────────────────────────────────────────────────┘

    关键设计决策：
      - 同步推理（非并行）：每个连接的处理是顺序的（接收 → 推理 → 发送 → 接收 →
        ...）。这是因为推理是计算密集型操作，并行推理多个请求没有意义（GPU 只能
        串行执行）。如果需要同时服务多个机器人，每个连接有自己的 handler 协程，
        推理请求会按到达顺序排队，由 asyncio 事件循环调度。
      - 无连接池：每个客户端连接独立处理。服务器不维护全局状态，因此连接之间
        不会相互干扰（除了共享 GPU 计算的排队等待）。
      - 健康检查端点：支持 HTTP GET /healthz 用于 Kubernetes/Docker 等容器编排
        平台的存活探针（liveness probe）。
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        """初始化 WebSocket 策略服务器。

        这个构造函数只保存参数，不启动网络服务。
        实际的网络监听在 serve_forever() 或 run() 中启动。

        Args:
            policy:  要提供服务的策略对象（BasePolicy 的任何实现）。
                     这是服务器的核心 —— 所有客户端请求最终都会调用
                     这个 policy.infer() 方法来执行推理。

                     可以是：
                       - 本地加载的模型（JAX 或 PyTorch，通过 policy_config 创建）
                       - PolicyRecorder（带记录功能的策略装饰器）
                       - 任何其他 BasePolicy 实现

            host:    服务器绑定的主机地址。
                     - "0.0.0.0"：监听所有网络接口（允许外部连接）
                     - "127.0.0.1"：仅监听本地回环接口（仅本机可访问）
                     - "192.168.1.100"：仅监听特定网卡

            port:    服务器监听的 TCP 端口。
                     - None：由操作系统自动分配一个空闲端口（适用于测试）
                     - 8000：openpi 策略服务器的惯例端口

            metadata: 服务器元数据字典（可选）。
                      这个字典会在每个客户端连接建立时，作为第一条消息
                      发送给客户端。通常包含：
                        {
                            "reset_pose": [0, -0.96, 1.16, 0, -0.3, 0],
                                                          # 机器人的初始关节位姿
                            "policy_name": "pi0_aloha",   # 模型名称
                            "action_dim": 14,              # 动作空间维度
                            "action_horizon": 25,          # 动作块大小
                            ...
                        }
                      客户端通过 get_server_metadata() 获取这些信息。
        """
        self._policy = policy  # 推理策略（服务器的"大脑"）
        self._host = host  # 监听地址
        self._port = port  # 监听端口
        self._metadata = metadata or {}  # 发送给客户端的元数据（默认为空字典）

        # 设置 websockets 库自身的日志级别
        # websockets 库默认会输出大量的调试信息（每个连接建立/断开），
        # 设为 INFO 级别来减少噪音，只显示重要事件。
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        """启动服务器并永久运行（同步阻塞入口）。

        这是最常用的启动方式 —— 一个阻塞调用，适合作为主程序的最后一步。
        它在内部调用 asyncio.run(self.run()) 来创建异步事件循环并运行服务器。

        为什么不直接做成 async 函数？
          因为这个方法的设计目的是"启动后永远不返回"——它是一个长期运行的服务。
          使用同步包装可以让调用者不需要处理 asyncio 事件循环的创建和管理。

        启动流程：
          1. 创建 asyncio 事件循环（asyncio.run）
          2. 创建 WebSocket 服务器（websockets.serve）
          3. 开始监听端口
          4. 接受客户端连接
          5. 为每个连接创建 handler 协程
          6. 永久运行下去（直到被 Ctrl+C 或 SIGTERM 终止）

        典型用法：
          server = WebsocketPolicyServer(policy=my_policy)
          server.serve_forever()  # ← 阻塞在这里，直到进程被杀死
        """
        # asyncio.run() 是 Python 3.7+ 中运行异步主函数的推荐方式。
        # 它会：
        #   1. 创建一个新的事件循环
        #   2. 运行传入的协程（self.run()）
        #   3. 关闭事件循环
        # 由于 self.run() 中的 server.serve_forever() 永远不会返回，
        # 这个 asyncio.run() 实际上会一直运行直到被中断（如 KeyboardInterrupt）。
        asyncio.run(self.run())

    async def run(self):
        """异步启动 WebSocket 服务器并永久运行。

        这个是实际的服务器启动逻辑（async 版本）。
        使用 async with 确保服务器在退出时正确关闭和清理资源。

        websockets.asyncio.server.serve() 的参数说明：
          - self._handler: 处理新连接的协程函数。
            每个新客户端连接都会启动一个独立的 _handler 协程实例。
            这是 WebSocket 服务器的"请求处理函数"——相当于 HTTP 中的视图函数。

          - compression=None: 禁用 WebSocket 压缩。
            原因：我们传输的数据主要是 NumPy 数组（已经高度紧凑的二进制格式），
            WebSocket 的 per-message 压缩算法（deflate）不仅压缩效果有限，
            反而会增加 CPU 开销和延迟。禁用压缩可以减少每步推理的延迟。

          - max_size=None: 不限制消息大小。
            原因：观测数据可能包含多张高分辨率摄像头图像（每张 ~150KB 甚至更大），
            一个完整的观测字典可能达到数 MB。默认的 WebSocket 消息大小限制（1MB）
            可能会被超过。设为 None 表示接受任意大小的消息。
            注意：这可能导致内存被耗尽（如果客户端恶意发送超大消息）。
            在生产环境中，应根据实际数据大小设置一个合理的上限。

          - process_request=_health_check: HTTP 请求预处理钩子。
            WebSocket 服务器在握手之前，客户端会发送 HTTP 升级请求。
            这个钩子允许我们在握手前拦截 HTTP 请求。
            我们用它来实现健康检查端点（HTTP GET /healthz）。

        关于 async with 语法：
          async with _server.serve(...) as server:
            这会在进入时创建并启动 WebSocket 服务器，在退出时（比如异常或
            Ctrl+C）自动关闭服务器并清理所有连接。

        Args:
            server: websockets 库创建的 WebSocket 服务器对象。
                    await server.serve_forever() 让服务器永久运行。
        """
        # ── 创建并启动 WebSocket 服务器 ──
        # 使用 async with 确保资源正确释放
        async with _server.serve(
            self._handler,  # 新连接处理器（协程）
            self._host,  # 绑定的主机地址
            self._port,  # 绑定的端口（None = 自动分配）
            compression=None,  # 禁用消息压缩（减少延迟）
            max_size=None,  # 不限消息大小（支持大图像传输）
            process_request=_health_check,  # HTTP 请求预处理（健康检查）
        ) as server:
            # ── 永久运行 ──
            # server.serve_forever() 会阻塞当前协程，直到服务器被关闭。
            # 通常不会返回，除非：
            #   - 收到关闭信号（SIGTERM、SIGINT）
            #   - 调用 server.close()
            #   - 发生未捕获的异常
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        """处理单个 WebSocket 连接的完整生命周期。

        这个方法是一个 asyncio 协程，为每个新建立的 WebSocket 连接自动调用。
        如果有多个客户端同时连接，每个连接会启动一个独立的 _handler 协程实例，
        由 asyncio 事件循环并发调度。

        连接生命周期：
          建立（connect）
            │
            ▼
          发送元数据 ← 第一条消息，告知客户端策略配置信息
            │
            ▼
          ┌─ 推理循环（loop）──────────────┐
          │  等待 obs → 反序列化 → 推理    │
          │  → 添加计时 → 序列化 → 发送    │
          └────────────────────────────────┘
            │
            ▼
          断开（close）或异常（error）

        WebSocket 通信的"双通道"特性：
          注意 WebSocket 是全双工的 —— 服务器可以在任何时候发送数据，
          不需要等待客户端请求。但在这个实现中，我们遵循严格的一问一答模式：
          等待客户端发送观测 → 回复动作。这是为了与同步客户端兼容。

        Args:
            websocket: WebSocket 连接对象。代表与一个客户端的双向通信通道。
                       它提供了 send() 和 recv() 方法。
                       参数类型为 _server.ServerConnection（websockets 库的
                       服务器端连接表示）。
        """
        # ── 记录新连接日志 ──
        # remote_address 是客户端的 IP 地址和端口，方便排查连接来源。
        logger.info(f"Connection from {websocket.remote_address} opened")

        # ── 准备 msgpack 序列化器 ──
        # Packer 是一个可复用的序列化器，支持 NumPy 数组的序列化。
        # 为每个连接创建独立的 Packer 实例是线程安全的（asyncio 单线程模型）。
        packer = msgpack_numpy.Packer()

        # ── 第一步：发送元数据 ──
        # 这是连接建立后的第一条消息（也是唯一一条非推理消息）。
        # 元数据包含：
        #   - reset_pose：机器人的起始关节位姿（用于 episode 开始前的归位）
        #   - 模型信息（名称、版本等）
        #   - 策略配置（action_dim、action_horizon 等）
        #
        # 客户端在连接后会先收到这条消息，然后才开始发送推理请求。
        # 这允许客户端在开始推理前，根据元数据配置环境（如设置复位位姿）。
        #
        # 数据格式：msgpack 编码的字典（包含 NumPy 数组如果需要的话）。
        await websocket.send(packer.pack(self._metadata))

        # ── 推理循环计时器（用于计算总耗时统计）──
        # prev_total_time 记录上一次推理循环的"完整往返时间"（包括接收obs、
        # 推理、发送action的全过程）。
        # 第一次循环时 prev_total_time 为 None，不会添加到结果中。
        # 从第二次开始，每次都会包含上一次的完整耗时，让客户端可以了解
        # 服务器端真正的"接收+推理+发送"总时间。
        prev_total_time = None

        # ── 第二步：进入推理循环 ──
        # 这是服务器的核心 —— 持续接收客户端的推理请求，返回推理结果。
        # 循环会一直运行，直到客户端断开连接或发生错误。
        while True:
            try:
                # ============================================================
                # 2.1 接收并反序列化观测数据
                # ============================================================
                # 记录整个循环的起始时间（包含接收、推理、发送三个阶段）
                start_time = time.monotonic()

                # await websocket.recv() 是异步等待 —— 当前协程会暂停，
                # 直到有数据到达。这使得事件循环可以在等待期间处理其他
                # 连接的事件（如果有多个客户端同时连接的话）。
                #
                # msgpack_numpy.unpackb 将二进制数据反序列化为 Python 字典，
                # 自动恢复其中的 NumPy 数组。
                #
                # 收到的是序列化后的观测字典，典型结构：
                #   {
                #       "images": {
                #           "cam_high": np.ndarray [3, 224, 224],  # 摄像头图像
                #           "cam_low": np.ndarray [3, 224, 224],
                #       },
                #       "state": np.ndarray [14],  # 关节状态（位置/速度）
                #       "prompt": "pick up the cube",  # 语言指令（可选）
                #   }
                obs = msgpack_numpy.unpackb(await websocket.recv())

                # ============================================================
                # 2.2 执行策略推理
                # ============================================================
                # 记录推理开始时间，精确测量推理耗时
                infer_time = time.monotonic()

                # 调用底层策略的 infer 方法 —— 这是整个系统的核心步骤。
                # self._policy.infer(obs) 会执行模型的前向传播：
                #   - JAX 模型：在 GPU 上执行 JIT 编译后的计算图
                #   - PyTorch 模型：在 GPU 上执行 torch.no_grad() 推理
                #   - 远程策略（如果 PolicyRecorder 包装了另一个远程策略）：
                #     递归调用远程策略服务器
                #
                # 这是同步（阻塞）调用 —— 虽然我们在 async 函数中，
                # 但 policy.infer() 本身是同步的，会阻塞当前协程。
                # 在推理期间，其他连接的 handler 协程仍然可以运行
                # （由 asyncio 事件循环调度）。
                #
                # 返回的动作字典典型结构：
                #   {
                #       "actions": np.ndarray [action_horizon, 14],
                #                                             # 预测的动作块
                #       "state": np.ndarray [...],            # 对应的状态
                #   }
                action = self._policy.infer(obs)

                # 计算推理耗时（毫秒）
                infer_time = time.monotonic() - infer_time

                # ============================================================
                # 2.3 添加性能计时信息
                # ============================================================
                # 在返回结果中附加服务器端的性能统计，帮助客户端了解
                # 推理服务的延迟状况。
                #
                # server_timing 字段包含：
                #   - infer_ms：本次推理的模型前向传播时间（毫秒）
                #     不包括网络传输和序列化开销。
                #   - prev_total_ms：上一次循环的完整处理时间
                #     （接收 + 反序列化 + 推理 + 序列化 + 发送）
                #     让客户端可以了解服务器的实际处理延迟。
                #
                # 这些信息对于调试性能问题和优化系统非常有用。
                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,  # 转换为毫秒
                }
                if prev_total_time is not None:
                    # 由于我们需要包含本次的 send 时间，但 send 还没执行，
                    # 所以只能记录上一次的总耗时（prev_total_ms）。
                    # 这样仍能让客户端知道"上次请求在服务器上花了多久"。
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                # ============================================================
                # 2.4 序列化并发送推理结果
                # ============================================================
                # packer.pack(action) 将动作字典序列化为 msgpack 二进制格式，
                # 自动处理其中的 NumPy 数组。
                #
                # await websocket.send() 异步发送 —— 不会阻塞事件循环。
                # 发送完成后，本轮推理循环结束，等待下一个观测。
                await websocket.send(packer.pack(action))

                # 记录本次循环总耗时（用于下一次循环的 prev_total_time）
                prev_total_time = time.monotonic() - start_time

            # ================================================================
            # 2.5 错误处理
            # ================================================================

            # ── 正常断开 ──
            # 客户端优雅关闭了连接（或网络断开导致连接丢失）。
            # websockets 库将底层 TCP 断开包装为 ConnectionClosed 异常。
            # 这是我们预期的正常退出路径 —— 记录日志后退出 handler 协程。
            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break  # 退出 while 循环，结束 handler

            # ── 异常错误 ──
            # 推理过程中发生未预料的异常（如模型错误、OOM、数据格式错误等）。
            # 处理策略：
            #   1. 将完整的异常回溯（traceback）发送给客户端 —— 方便调试
            #   2. 关闭 WebSocket 连接，使用 INTERNAL_ERROR 状态码
            #   3. 重新抛出异常 —— 让上层（asyncio 事件循环）知道发生了错误
            #
            # 注意：这里没有捕获 KeyboardInterrupt 或 asyncio.CancelledError，
            # 这些会自然传播，导致服务器正常关闭。
            except Exception:
                # 将异常的完整堆栈信息发送给客户端。
                # traceback.format_exc() 返回当前异常的完整回溯字符串。
                # 客户端会在 WebsocketClientPolicy.infer() 中检查
                # 响应是否为字符串（而非二进制），如果是则抛出 RuntimeError。
                #
                # 这样做的目的：即使服务器崩溃，客户端也能知道发生了什么错误，
                # 而不是"连接突然断开，原因不明"。
                await websocket.send(traceback.format_exc())

                # 使用 INTERNAL_ERROR 状态码关闭连接。
                # websockets.frames.CloseCode.INTERNAL_ERROR = 1011
                # 这是一个标准的 WebSocket 关闭码，表示服务器发生了意外错误。
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )

                # 重新抛出异常。
                # 如果不重新抛出，错误会被静默吞掉，asyncio 事件循环不会知道
                # 连接处理失败了。重新抛出后，asyncio 会记录错误日志并继续
                # 处理其他连接。
                raise


def _health_check(
    connection: _server.ServerConnection, request: _server.Request
) -> _server.Response | None:
    """HTTP 请求预处理钩子 —— 实现健康检查端点。

    这个函数作为 process_request 参数传递给 websockets.serve()。
    在 WebSocket 握手（HTTP 升级）之前，每个 HTTP 请求都会先经过这个函数。
    我们可以在这里拦截 HTTP 请求，实现健康检查等 HTTP 端点。

    健康检查（Health Check）：
      在容器化部署（Docker / Kubernetes）中，编排系统需要定期检查
      服务是否存活。这就是"存活探针"（Liveness Probe）—— 一个简单的
      HTTP GET 请求，如果返回 200 OK，则认为服务健康。

      当服务器部署在 Kubernetes 中时，可以配置：
        livenessProbe:
          httpGet:
            path: /healthz
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10

    为什么放在 WebSocket 服务器中？
      - 不需要额外的 HTTP 服务器（如 FastAPI/Flask）
      - 零额外依赖
      - 健康检查功能与主服务生命周期一致

    工作原理：
      1. 客户端（健康检查器）发送 HTTP GET /healthz 到 WebSocket 端口
      2. 这个函数拦截到 HTTP 请求，检查 request.path
      3. 如果路径是 "/healthz"，直接返回 HTTP 200 OK 响应
      4. 如果不是健康检查路径，返回 None，让 websockets 库正常处理
         （WebSocket 握手升级）

    Args:
        connection: WebSocket 连接对象（实际此时还是 HTTP 连接）。
        request:    客户端的 HTTP 请求对象，包含 method、path、headers 等。
                    我们只检查 request.path。

    Returns:
        - Response: 如果请求路径是 "/healthz"，返回 HTTP 200 响应。
          这告诉健康检查器"服务正常运行"。
        - None: 对于所有其他路径，返回 None 表示"继续正常处理"。
          websockets 库会将请求升级为 WebSocket 连接。
          WebSocket 客户端连接的就是这种情况。
    """
    # ── 检查请求路径 ──
    if request.path == "/healthz":
        # 返回 HTTP 200 OK 响应，响应体为 "OK\n"
        # connection.respond() 是 websockets 库提供的方法，
        # 用于直接向客户端发送 HTTP 响应，阻止后续的 WebSocket 升级。
        return connection.respond(http.HTTPStatus.OK, "OK\n")

    # ── 非健康检查请求 → 继续正常处理（WebSocket 握手） ──
    # 返回 None 表示"我不处理这个请求，让 websockets 库自己处理"。
    # 对于正常的 WebSocket 客户端连接，这会触发 WebSocket 握手，
    # 然后 websockets 库会启动对应的 _handler 协程。
    return None


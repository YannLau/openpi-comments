"""
WebSocket 策略客户端模块

本模块定义了 WebsocketClientPolicy 类——通过 WebSocket 协议连接远程策略服务器
进行推理的客户端实现。

它有什么用？
  在 openpi 系统中，策略推理（模型前向传播）通常在 GPU 服务器上运行，
  而机器人控制代码在另一台机器（或同一台机器的不同进程）上运行。
  WebsocketClientPolicy 就是连接这两者的"网络桥梁"：

    机器人控制端（CPU）                    推理服务器端（GPU）
  ┌──────────────────────────┐          ┌──────────────────────────┐
  │  Runtime 主循环           │          │  WebsocketPolicyServer    │
  │    │                      │          │    │                      │
  │    ▼                      │  WebSocket  │    ▼                    │
  │  PolicyAgent               │ ────────→ │  Policy.infer(obs)      │
  │    │                      │  obs       │    │                    │
  │    ▼                      │ ←──────── │    ▼                    │
  │  ActionChunkBroker        │  action    │  模型前向传播            │
  │    │                      │          │                          │
  │    ▼                      │          │                          │
  │  WebsocketClientPolicy ◀──┘          │  WebsocketPolicyServer   │
  └──────────────────────────┘          └──────────────────────────┘

与本地推理（src/openpi/policies/policy.py）的区别：
  - 本地推理：模型直接加载在当前进程中，推理在当前进程完成
  - 远程推理（本类）：模型在服务器上，通过网络请求推理结果

  远程推理的好处：
    1. GPU 共享：多台机器人可以共享一个 GPU 服务器
    2. 资源隔离：模型加载和推理不影响机器人控制进程的实时性
    3. 热更新：可以在不重启机器人控制端的情况下更新服务端模型
    4. 语言无关：服务端可以是任何语言（Python/C++/Rust），客户端无感知

类设计：
  它实现 BasePolicy 接口（infer + reset），因此可以像本地策略一样被使用。
  Runtime → PolicyAgent → ActionChunkBroker → WebsocketClientPolicy
  对调用者来说，它和本地加载的 Policy 对象没有区别。

服务发现：
  构造函数会阻塞等待服务器就绪（_wait_for_server），通过循环尝试连接
  直到成功。这使得机器人控制端可以先启动，等待推理服务器上线。

  ⚠️ 注意：如果服务器始终未启动，客户端会无限等待（每 5 秒重试一次）。
"""

import logging  # 日志系统，用于输出连接状态信息
import time  # 时间库，用于重试间隔和计时
from typing import Dict, Optional, Tuple  # 类型提示

from typing_extensions import override  # 类型提示：显式标记方法覆盖
import websockets.sync.client  # WebSocket 同步客户端（用于与服务端通信）

from openpi_client import base_policy as _base_policy  # 基础策略接口
from openpi_client import msgpack_numpy  # 支持 NumPy 数组的 msgpack 序列化


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """WebSocket 策略客户端 —— 通过远程连接调用策略推理。

    这个类将"网络通信"封装成"策略调用"，对上层透明。
    调用者只需调用 infer(obs)，底层自动完成：
      序列化 → 发送 → 等待 → 接收 → 反序列化

    和直接调用远端函数的区别：
      这里不是 RPC（远程过程调用），而是通过 WebSocket 发送和接收数据。
      客户端和服务端之间通过 msgpack 格式交换序列化后的字典数据。
      你可以理解为"在网络上传输字典"——只是字典的值可以是 NumPy 数组。

    对应服务端实现：src/openpi/serving/websocket_policy_server.py
    """

    def __init__(self, host: str = "0.0.0.0", port: Optional[int] = None, api_key: Optional[str] = None) -> None:
        """初始化 WebSocket 客户端并等待服务器就绪。

        构造过程：
          1. 构建 WebSocket URI（支持 ws:// 前缀或裸 host）
          2. 准备序列化工具（msgpack + NumPy 支持）
          3. 阻塞等待服务器连接（_wait_for_server）
          4. 连接成功后接收服务器元数据（如模型信息、重置位姿等）

        Args:
            host:    服务器主机地址。
                     - 可以是不带协议的地址："localhost" 或 "192.168.1.100"
                     - 也可以是完整 WebSocket URI："ws://example.com:8000"
                     - 默认 "0.0.0.0" 表示本地连接
            port:    服务器端口（如 8000）。
                     如果 host 已经包含端口（如 "ws://host:8000"），这里可留 None。
            api_key: 可选的认证密钥。如果服务器要求身份验证，
                     通过 HTTP Header "Authorization: Api-Key <key>" 发送。
                     为 None 时不发送认证头。

        连接成功后，服务端会立即发送一条元数据消息（msgpack 格式），
        包含策略的配置信息，例如：
            {
                "reset_pose": [0, -1.5, 1.5, 0, 0, 0],  # 起始复位位姿
                "model_name": "pi0_aloha_sim",
                ...
            }
        这些元数据可以通过 get_server_metadata() 获取。
        """
        # ====================================================================
        # 构建 WebSocket URI
        #
        # 处理两种输入格式：
        #   1. 完整 URI："ws://192.168.1.100:8000/ws" → 直接使用
        #   2. 裸地址："192.168.1.100" → 拼接为 "ws://192.168.1.100"
        # 如果提供了 port，追加 ":port"
        # ====================================================================
        if host.startswith("ws"):
            self._uri = host  # 已经是完整 URI，直接使用
        else:
            self._uri = f"ws://{host}"  # 拼接 ws:// 前缀
        if port is not None:
            self._uri += f":{port}"  # 追加端口

        # 准备 msgpack 序列化器（支持 ndarray 的自动打包/解包）
        self._packer = msgpack_numpy.Packer()

        # 可选的 API 密钥（用于服务器认证）
        self._api_key = api_key

        # 连接服务器并获取元数据
        # _wait_for_server 会阻塞直到连接成功
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        """返回服务端发送的元数据。

        元数据包含服务器加载的策略配置信息，例如：
        - reset_pose：机器人复位位姿（用于 episode 开始前的归位动作）
        - 模型名称/版本信息
        - 其他服务端配置

        这些元数据在连接建立时由服务端主动发送（第一条消息），
        客户端可以通过此方法获取，用于了解服务端状态。

        Returns:
            dict: 服务端元数据字典。
        """
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        """等待 WebSocket 服务器就绪并建立连接。

        这个方法会：
          1. 循环尝试连接指定的 URI
          2. 如果连接被拒绝，等待 5 秒后重试
          3. 连接成功后，接收服务端发送的元数据（msgpack 格式）
          4. 返回 (连接对象, 元数据字典)

        为什么是同步阻塞而不是异步？
          因为 WebsocketClientPolicy 通常在 Runtime 主循环中使用，
          Runtime 是同步的（每步推理都要等待结果）。异步不会带来
          好处，反而增加了复杂性。

        为什么需要等待重试？
          在机器人部署场景中，启动顺序通常是：
            1. 启动机器人控制端（本类）
            2. 启动推理服务端（可能需要加载大模型，耗时较长）
          客户端需要在服务端就绪前等待，而不是直接报错退出。

        Returns:
            Tuple[ClientConnection, Dict]: (WebSocket 连接, 服务端元数据)

        Raises:
            实际上不会抛出异常——它会无限重试直到连接成功。
            如果过程中发生非 ConnectionRefusedError 的其他错误，
            则会抛出对应的异常（如网络不可达、DNS 解析失败等）。
        """
        logging.info(f"Waiting for server at {self._uri}...")

        # 循环重试直到连接成功
        while True:
            try:
                # 准备 HTTP 头（如果需要认证）
                headers = (
                    {"Authorization": f"Api-Key {self._api_key}"}
                    if self._api_key
                    else None
                )

                # 尝试建立 WebSocket 连接
                # compression=None: 禁用压缩（NumPy 数组已经是二进制，压缩效果有限）
                # max_size=None: 不限制消息大小（图像数据可能很大）
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                )

                # 连接成功后，服务端会发送第一条消息（元数据）
                # 消息格式是 msgpack 编码的字典
                metadata = msgpack_numpy.unpackb(conn.recv())

                # 返回连接对象和元数据
                return conn, metadata

            except ConnectionRefusedError:
                # 服务器尚未启动或端口未开放
                # 打印日志并等待 5 秒后重试
                logging.info("Still waiting for server...")
                time.sleep(5)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        """执行远程策略推理：发送观测 → 接收动作。

        这是 BasePolicy.infer() 的实现。它通过网络将观测数据发送到
        推理服务器，等待服务器返回推理结果。

        工作流程：
          1. 序列化（pack）：将包含 NumPy 数组的观测字典转为 bytes
          2. 发送：通过 WebSocket 将数据发送到服务器
          3. 等待：阻塞等待服务器推理完成并返回结果
          4. 接收：从 WebSocket 接收响应数据
          5. 反序列化（unpack）：将 bytes 还原为包含 NumPy 数组的动作字典
          6. 返回：将动作字典返回给调用者

        性能考虑：
          - 序列化/反序列化大图像（~150KB）在毫秒级
          - 网络延迟取决于客户端和服务器的距离（局域网 <1ms，跨机器 ~1-5ms）
          - 推理时间取决于模型大小和 GPU（通常 10-50ms）
          - 大部分时间花在服务器推理上，网络开销相对较小

        错误处理：
          - 如果服务器返回字符串（而非二进制），说明服务端发生了错误，
            会抛出 RuntimeError 异常。
          - 如果连接断开，websockets 库会抛出 ConnectionClosed 异常，
            由上层调用者处理（Runtime 主循环会终止）。

        Args:
            obs: 观测字典。包含图像（ndarray）、状态（ndarray）等。
                 字典的键值类型需要能被 msgpack + pack_array 序列化。
                 支持的数据类型：ndarray、Python 原生类型、嵌套字典/列表。

        Returns:
            dict: 动作字典。包含策略预测的动作（ndarray）及服务端返回的
                  其他信息（如推理耗时统计、状态信息等）。

        Raises:
            RuntimeError: 服务端返回错误信息（字符串消息）时触发。
        """
        # 步骤 1：序列化观测数据
        # msgpack_numpy.Packer 会自动处理 ndarray 的编码
        data = self._packer.pack(obs)

        # 步骤 2：发送序列化后的数据到服务器
        self._ws.send(data)

        # 步骤 3：接收服务器响应
        response = self._ws.recv()

        # 步骤 4：检查是否有错误
        # 正常响应是 bytes（msgpack 二进制格式）
        # 如果服务器返回 str，说明发生了错误
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")

        # 步骤 5：反序列化并返回结果
        # msgpack_numpy.unpackb 会自动将编码的 ndarray 还原
        return msgpack_numpy.unpackb(response)

    @override
    def reset(self) -> None:
        """重置策略状态。

        对于远程策略客户端来说，"重置"意味着清空服务端可能维护的
        内部状态（如 action chunk cache、RNN hidden state 等）。

        但是，当前这个实现中 reset() 什么也不做。
        原因如下：

        1. 服务端的状态管理：
           服务端（WebsocketPolicyServer）为每个 WebSocket 连接维护一个
           独立的状态。当客户端调用 reset() 时，需要在服务端也触发重置。
           目前这个版本的协议没有设计"重置"消息——reset 信息没有发送到服务端。

        2. 为什么还能工作？
           对于大多数策略（如 π₀），推理是无状态的——每次 infer(obs) 都是
           独立的，不依赖之前的状态。只有 ActionChunkBroker 这样的包装器
           需要维护缓存状态，而它是在客户端实现的，由 PolicyAgent 管理。

        3. 未来改进：
           如果服务端策略需要维护状态（如 RNN 模型、在线适应算法），
           需要扩展协议：客户端发送一个 "reset" 信号，服务端清空其状态。
        """
        pass

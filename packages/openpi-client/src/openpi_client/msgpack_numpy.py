"""
NumPy 数组的 msgpack 序列化扩展模块

本模块将 msgpack 序列化库扩展为支持 NumPy 数组（ndarray）的序列化和反序列化。
它是 openpi 中 WebSocket 通信的底层数据格式基础。

为什么使用 msgpack？
  msgpack 是一种高效的二进制序列化格式，类似于 JSON 但更紧凑、更快速。
  在需要通过网络传输大量数据（尤其是图像和状态数组）时，选择 msgpack 有
  以下几个关键考量：

  1. 安全性（Security）
     相比之下，Python 的 pickle/dill 等格式允许任意代码执行——
     反序列化 pickle 数据等于执行攻击者的代码。msgpack 是纯数据格式，
     不存在远程代码执行（RCE）风险。

  2. 跨语言支持（Cross-language）
     msgpack 在几乎所有编程语言中都有实现。这对于机器人系统特别重要：
     Python 训练，C++/Rust 部署的混合架构很常见。

  3. 无需预定义模式（Schema-free）
     与 protobuf、flatbuffers 等需要先定义 .proto 文件的序列化格式不同，
     msgpack 不需要模式定义。这在动态类型语言（Python、JavaScript）中
     更加灵活——发送方和接收方不需要事先约定数据结构。

  4. 性能（Performance）
     二进制格式，比 JSON 更快更小。
     原作者测试发现：对于大数组的序列化，msgpack 比 pickle 快约 4 倍。

本模块的来源：
  代码改编自 https://github.com/lebedov/msgpack-numpy 库。
  不直接使用该库的原因是：它在处理对象数组（dtype=object）时会回退到
  不安全的 pickle 序列化。本实现直接拒绝不支持的 dtype，避免安全隐患。

数据流（WebSocket 通信场景）：
  客户端（仿真/机器人）                    服务端（策略推理）
  ┌─────────────────────┐                ┌─────────────────────┐
  │ obs = {              │                │                      │
  │   "images": ndarray, │   packb()     │ unpackb()             │
  │   "state": ndarray,  │ ────────────→ │ obs = {               │
  │   "prompt": str      │   WebSocket   │   "images": ndarray,  │
  │ }                    │               │   "state": ndarray,   │
  │                      │               │   "prompt": str       │
  │ action = {           │   unpackb()   │ }                     │
  │   "actions": ndarray │ ←──────────── │                      │
  │ }                    │               │ action = policy.infer│
  └─────────────────────┘                └─────────────────────┘

为什么不用 JSON？
  JSON 不支持二进制数据（需要 base64 编码，体积增大 33%），
  且 NumPy 数组在 JSON 中只能展开为列表，丢失形状和 dtype 信息。

序列化格式说明：
  本模块将 ndarray 编码为一个包含以下字段的字典：
    {
        "__ndarray__": True,          # 标记：这是一个 NumPy 数组
        "data": <bytes>,              # 原始二进制数据（tobytes()）
        "dtype": "<f8",              # 数据类型字符串（如 float64、uint8）
        "shape": (3, 224, 224),       # 数组形状
    }

  这样设计的好处：
    - 自描述：接收方从字典中就能知道数组的形状和类型，无需额外信息
    - 高效：数据以原始二进制形式传输，没有任何转换开销
    - 兼容：普通的 msgpack 解码器也能识别这是一个字典（只是不认识 __ndarray__
      标记），不会崩溃，只是效率稍低
"""

import functools  # functools.partial：固定函数的部分参数，生成新的可调用对象

import msgpack  # msgpack 序列化库（核心）
import numpy as np  # NumPy 数组库


def pack_array(obj):
    """将 NumPy 数组编码为 msgpack 可序列化的字典格式（序列化钩子函数）。

    这个函数作为 msgpack.Packer(default=pack_array) 的 default 回调。
    当 msgpack 遇到它不知道如何序列化的类型时，会调用这个函数。

    工作逻辑：
      如果 obj 是 ndarray → 编码为自定义字典格式（含标记、数据、类型、形状）
      如果 obj 是 np.generic（标量）→ 编码为标量格式
      否则 → 返回原值（让 msgpack 自己处理，或抛出异常）

    Args:
        obj: 要序列化的对象。msgpack 会在遇到无法处理的类型时调用此函数。

    Returns:
        可被 msgpack 序列化的对象（通常是字典）。

    Raises:
        ValueError: 如果数组的 dtype 是 void（"V"）、object（"O"）或 complex（"c"）。
                    这些类型无法安全或高效地序列化：
                    - "V"（void）：原始字节，缺乏类型信息，反序列化时无法恢复
                    - "O"（object）：可能包含任意 Python 对象，需 pickle → 不安全
                    - "c"（complex）：msgpack-numpy 的原始库支持复杂数但效率低

    序列化示例：
        >>> arr = np.array([[1, 2], [3, 4]], dtype=np.float32)
        >>> pack_array(arr)
        {
            b'__ndarray__': True,
            b'data': b'...',           # arr.tobytes() 的结果
            b'dtype': '<f4',            # little-endian float32
            b'shape': (2, 2),
        }

        >>> pack_array(np.float64(3.14))
        {
            b'__npgeneric__': True,
            b'data': 3.14,
            b'dtype': '<f8',
        }

    注意：
      - 字典的键使用 bytes（b'...'）而不是 str——这是为了与 msgpack 的
        二进制头兼容，减少序列化后的体积。
      - data 字段存储的是原始内存布局（tobytes()），不包含任何元数据。
        形状和类型信息单独保存在 dtype 和 shape 字段中。
    """
    # ── 安全检查：拒绝不安全的 dtype ──
    #   "V" = void（没有类型信息的原始字节）
    #   "O" = object（可能包含任意 Python 对象 → 需要 pickle → 不安全）
    #   "c" = complex（复数，msgpack 原生不支持）
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    # ── ndarray：多维数组 ──
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,       # 标记：这是一个 ndarray
            b"data": obj.tobytes(),     # 数组的原始二进制数据
            b"dtype": obj.dtype.str,    # 数据类型（如 "<f8" = float64）
            b"shape": obj.shape,        # 形状元组（如 (224, 224, 3)）
        }

    # ── np.generic：NumPy 标量（如 np.float64(3.14)）──
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,     # 标记：这是一个 NumPy 标量
            b"data": obj.item(),        # 转换为 Python 原生类型（如 float）
            b"dtype": obj.dtype.str,    # 数据类型
        }

    # ── 其他类型：不做处理，交给 msgpack 自己处理或抛出异常 ──
    return obj


def unpack_array(obj):
    """将 msgpack 解码后的字典还原为 NumPy 数组（反序列化钩子函数）。

    这个函数作为 msgpack.Unpacker(object_hook=unpack_array) 的 object_hook 回调。
    每当 msgpack 解码出一个字典时，都会调用这个函数检查是否需要还原为 ndarray。

    Args:
        obj: msgpack 解码后的字典。可能包含 __ndarray__ 或 __npgeneric__ 标记。

    Returns:
        如果 obj 包含 __ndarray__ 标记 → 还原为 np.ndarray
        如果 obj 包含 __npgeneric__ 标记 → 还原为 np.generic（标量）
        否则 → 原样返回字典

    反序列化示例：
        >>> data = {
        ...     b'__ndarray__': True,
        ...     b'data': b'...',
        ...     b'dtype': '<f4',
        ...     b'shape': (2, 2),
        ... }
        >>> unpack_array(data)
        array([[1., 2.],
               [3., 4.]], dtype=float32)

    实现细节：
      - np.ndarray(buffer=..., dtype=..., shape=...) 直接从内存缓冲区
        创建数组，零拷贝开销。
      - data 字段是 bytes 对象，np.ndarray 的 buffer 接口要求 buffer 是
        可读的类字节对象。tobytes() 返回的就是 bytes，完美匹配。
      - dtype.str 是类型字符串（如 '<f8'），np.dtype() 可以解析它。
        这保证了跨平台兼容性（明确指定字节序 '<' 或 '>'）。
    """
    # ── 还原 ndarray ──
    if b"__ndarray__" in obj:
        # 直接从二进制缓冲区创建数组（零拷贝构造）
        return np.ndarray(
            buffer=obj[b"data"],          # 原始二进制数据
            dtype=np.dtype(obj[b"dtype"]),  # 数据类型（如 np.float64）
            shape=obj[b"shape"],           # 形状
        )

    # ── 还原 np.generic（标量）──
    if b"__npgeneric__" in obj:
        # 使用 dtype 的 type() 方法从 Python 原生值创建 NumPy 标量
        # 例如 np.float64(3.14)
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    # ── 普通字典：不做处理 ──
    return obj


# ============================================================================
# 便捷接口
#
# 使用 functools.partial 固定 msgpack 序列化/反序列化函数的 default 和
# object_hook 参数，创建四个开箱即用的接口：
#
#   Packer  /  packb    —— 序列化（对象 → bytes）
#   Unpacker / unpackb  —— 反序列化（bytes → 对象）
#
# 接口对比：
#   Packer    = 流式 Packer 类（适用于大数据的增量编码）
#   packb     = 一次性打包函数（适用于小数据的快速编码）
#   Unpacker  = 流式 Unpacker 类（适用于大数据的增量解码）
#   unpackb   = 一次性解包函数（适用于小数据的快速解码）
#
# 使用示例：
#   from openpi_client import msgpack_numpy
#
#   # 序列化
#   data = {"image": np.zeros((224, 224, 3), dtype=np.uint8)}
#   packed = msgpack_numpy.packb(data)  # -> bytes
#
#   # 反序列化
#   restored = msgpack_numpy.unpackb(packed)  # -> dict with ndarray
#
#   # 流式写入（适合大文件/网络流）
#   packer = msgpack_numpy.Packer()
#   with open("data.msgpack", "wb") as f:
#       f.write(packer.pack(data1))
#       f.write(packer.pack(data2))
# ============================================================================

# packb 和 Packer：序列化入口
#   default=pack_array 告诉 msgpack：遇到未知类型时调用 pack_array 处理
Packer = functools.partial(msgpack.Packer, default=pack_array)
packb = functools.partial(msgpack.packb, default=pack_array)

# unpackb 和 Unpacker：反序列化入口
#   object_hook=unpack_array 告诉 msgpack：解码每个字典时调用 unpack_array 检查
Unpacker = functools.partial(msgpack.Unpacker, object_hook=unpack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)

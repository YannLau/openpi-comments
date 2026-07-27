"""
Fake Client —— 向 pi05_tron_single_data_lora 策略服务器发送虚假观测并打印推理结果。

【用途】
在没有真实机器人的情况下，测试策略服务器是否正常运行，以及观察输入/输出格式。
- 伪造 state (16维)、image (3个摄像头 3x224x224)、prompt
- 使用 openpi_client 的 WebsocketClientPolicy 连接服务器
- 打印服务器元数据、输入数据结构、输出数据结构

【用法】
1. 先启动策略服务器（使用 pi05_tron_single_data_lora 配置）：
       uv run scripts/serve_policy.py \
           policy:checkpoint \
           policy.config=pi05_tron_single_data_lora \
           policy.dir=/path/to/checkpoint

2. 再运行本脚本：
       uv run python examples/tron2/fake_client.py
       或者指定 host/port：
       uv run python examples/tron2/fake_client.py --host 192.168.1.100 --port 8000

【Key 命名说明】
本脚本使用 "image"（单数）作为图像字典的键名，以匹配 Tron2Inputs 变换的期望。
如果你发现服务器报 KeyError，可以尝试改为 "images"（复数）—— 将
make_fake_observation() 中的 "image" 改为 "images" 即可。
这取决于你的 serve_policy 调用链中是否有额外的 repack 变换做重命名。
"""

import argparse
import time
import numpy as np
from typing import Any

from openpi_client import websocket_client_policy


# ============================================================================
# 工具函数：漂亮打印字典结构（数组显示 shape，小数据直接打印）
# ============================================================================

def _format_value(val: Any, indent: int = 0, max_elements: int = 8) -> str:
    """格式化一个值：大数组显示 shape/dtype，小数组显示具体值。

    Args:
        val: 要格式化的值。
        indent: 当前缩进层级（用于递归）。
        max_elements: 超过此数量的元素不再逐项显示，改用 shape 描述。

    Returns:
        格式化后的字符串。
    """
    pad = "  " * indent

    if isinstance(val, np.ndarray):
        # ── numpy 数组 ──
        n = val.size
        if n > max_elements or val.ndim > 1:
            desc = f"np.ndarray(shape={list(val.shape)}, dtype={val.dtype})"
            if np.issubdtype(val.dtype, np.floating):
                desc += f"  range=[{val.min():.4f}, {val.max():.4f}]"
            elif np.issubdtype(val.dtype, np.integer):
                desc += f"  range=[{val.min()}, {val.max()}]"
            return f"{pad}{desc}"
        else:
            # 小的一维数组 → 打印具体值
            return f"{pad}np.ndarray({np.array2string(val, precision=4, separator=', ')}, dtype={val.dtype})"

    elif isinstance(val, dict):
        # ── 嵌套字典 ──
        items = []
        for k, v in val.items():
            items.append(f"{pad}  {k}: {_format_value(v, indent + 1)}")
        inner = "\n".join(items)
        return f"{pad}{{\n{inner}\n{pad}}}"

    elif isinstance(val, list):
        n = len(val)
        if n > max_elements:
            return f"{pad}list[{n}] (first {max_elements}: {val[:max_elements]})"
        else:
            return f"{pad}[{', '.join(repr(v) for v in val)}]"

    elif isinstance(val, str):
        # 长字符串截断
        if len(val) > 80:
            return f"{pad}\"{val[:77]}...\""
        return f"{pad}\"{val}\""

    elif isinstance(val, (bool, np.bool_)):
        return f"{pad}{bool(val)}"

    elif val is None:
        return f"{pad}None"

    else:
        return f"{pad}{repr(val)}"


def pretty_print_dict(d: dict, title: str = "") -> None:
    """以可读格式打印字典结构。

    Args:
        d: 要打印的字典。
        title: 可选的标题（会以颜色高亮显示）。
    """
    if title:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")

    for key, val in d.items():
        print(f"  {key}: {_format_value(val)}")
    print()


def make_fake_observation() -> dict:
    """创建一条与 Tron2 真实观测格式相同的假数据。

    模拟真实机器人发送给策略服务器的观测结构。
    注意服务器端 Tron2Inputs 变换会处理图像从 [C,H,W] → [H,W,C]，
    所以我们发送 [C,H,W] 格式（与真实客户端一致）。

    Returns:
        dict: 包含以下字段：
            - image: dict[str, np.ndarray] 三个摄像头 [3, 224, 224] uint8
            - state: np.ndarray [16] float64 关节状态
            - prompt: str 任务指示
    """
    # ── 伪造关节状态 ──
    # 结构：[左臂7, 左夹爪1, 右臂7, 右夹爪1] = 16 维
    # 模拟一个接近零位的随机姿态（范围 -0.5 ~ 0.5 弧度，夹爪 0~1）
    state = np.zeros(16, dtype=np.float64)
    state[0:7] = np.random.uniform(-0.5, 0.5, size=7)   # 左臂关节
    state[7] = np.random.uniform(0.0, 1.0)               # 左夹爪
    state[8:15] = np.random.uniform(-0.5, 0.5, size=7)   # 右臂关节
    state[15] = np.random.uniform(0.0, 1.0)               # 右夹爪

    # ── 伪造图像 ──
    # 3 个摄像头，[C, H, W] = [3, 224, 224]，uint8
    # 用随机噪声模拟图像（真实场景中这里是 realsense 相机数据）
    fake_rgb = np.random.randint(0, 256, size=(3, 224, 224), dtype=np.uint8)

    images = {
        "cam_high": fake_rgb,
        "cam_left_wrist": fake_rgb,
        "cam_right_wrist": fake_rgb,
    }

    # ── 注意键名 ──
    # Tron2Inputs 变换内部使用 data["image"]（单数），不是 data["images"]（复数）
    # 这与真实客户端（real_env.py / pi_client.py）使用的 "images" 键不同。
    # 参考 make_aloha_example() 也使用 "image"（单数）。
    obs = {
        "image": images,
        "state": state,
        "prompt": "Put the banana on the plate.",
    }
    return obs


# ============================================================================
# 主程序
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fake Tron2 client — send fake observations to a policy server and inspect the response."
    )
    parser.add_argument("--host", default="127.0.1.1", help="Policy server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Policy server port (default: 8000)")
    parser.add_argument("--rounds", type=int, default=3, help="Number of inference rounds (default: 3)")
    args = parser.parse_args()

    # ================================================================
    # 第 1 步：连接策略服务器
    # ================================================================
    print(f"Connecting to policy server at {args.host}:{args.port} ...")
    ws_client = websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )

    # ── 打印服务器元数据 ──
    # 服务器在连接建立后发送的第一条消息，包含策略配置信息
    metadata = ws_client.get_server_metadata()
    pretty_print_dict(metadata, title="Server Metadata (from handshake)")

    # ================================================================
    # 第 2 步：多轮推理测试
    # ================================================================
    for t in range(args.rounds):
        print(f"\n{'#' * 40}")
        print(f"  Round {t + 1} / {args.rounds}")
        print(f"{'#' * 40}")

        # ── 生成虚假观测 ──
        obs = make_fake_observation()

        # ── 打印输入结构 ──
        pretty_print_dict(obs, title="Input (fake observation)")

        # ── 推理 ──
        ts = time.time()
        result = ws_client.infer(obs)
        te = time.time()

        print(f"  Inference time: {te - ts:.3f}s ({1000*(te-ts):.1f}ms)")

        # ── 打印输出结构 ──
        pretty_print_dict(result, title="Output (policy result)")

        # ── 额外的动作信息 ──
        if "actions" in result:
            actions = result["actions"]
            if isinstance(actions, np.ndarray):
                print(f"  >>> actions shape: {actions.shape}")
                print(f"  >>> actions range: [{actions.min():.4f}, {actions.max():.4f}]")
                print(f"  >>> first action (left arm): {actions[0][:8]}")
                print(f"  >>> first action (right arm): {actions[0][8:]}")
                print(f"  >>> last action (left arm):  {actions[-1][:8]}")
                print(f"  >>> last action (right arm):  {actions[-1][8:]}")
        print()

        # 短暂间隔，避免过于密集
        time.sleep(0.1)

    print("Done. All rounds completed successfully.")


if __name__ == "__main__":
    main()

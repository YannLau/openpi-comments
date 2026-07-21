"""使用 LeRobotDataset 加载 Tron2 数据集并打印一条样本。"""

import os

os.environ["HF_LEROBOT_HOME"] = "/home/punk/yann_repo/tron2/example"

# --- LeRobot（Hugging Face 上的机器人数据集格式）---
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset



REPO = "lerobot_2026-07-20_09-31-04"

# 创建 LeRobot 数据集实例
# delta_timestamps 指定每个样本需要返回哪些时间步的动作
# 例如 action_horizon=16 且 fps=10，则 delta_timestamps = [0.0, 0.1, 0.2, ..., 1.5]（秒）

ds = lerobot_dataset.LeRobotDataset(REPO,episodes=[0])


print("=" * 70)
print("数据集基本信息")
print("=" * 70)
print(f"  num_frames:    {ds.num_frames}")
print(f"  num_episodes:  {ds.num_episodes}")
print(f"  num_features:  {len(ds.features)}")
print(f"  features:")
for key, feat in ds.features.items():
    print(f"    {key}:  dtype={feat['dtype']}, shape={feat.get('shape', 'N/A')}")

# ── 打印第 0 帧 ──────────────────────────────────────────
print("\n" + "=" * 70)
print("第 0 帧 (ds[0]) — 所有 field")
print("=" * 70)

frame = ds[0]
for key, val in frame.items():
    if val is None:
        print(f"\n  {key}:  <None>")
        continue

    if hasattr(val, "shape"):
        print(f"\n  {key}:  shape={val.shape}, dtype={val.dtype}")
        if val.ndim >= 3:
            print(
                f"    min={val.min().item():.3f}, max={val.max().item():.3f}, mean={val.float().mean().item():.3f}  (图像)"
            )
        elif val.numel() <= 64:
            print(f"    values = {val.tolist()}")
        else:
            print(f"    ({val.numel()} 个元素，略过)")
    elif isinstance(val, str):
        print(
            f'\n  {key}:  str(len={len(val)}) = "{val[:150]}{"..." if len(val) > 150 else ""}"'
        )
    elif isinstance(val, (int, float)):
        print(f"\n  {key}:  {type(val).__name__} = {val}")
    else:
        print(f"\n  {key}:  {type(val).__name__}")
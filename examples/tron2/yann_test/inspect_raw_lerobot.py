#!/usr/bin/env python3
"""
直接加载 LeRobot v2.1 原始数据，查看 state/action 的真实维度和顺序。

对比分析：
  - 数据集原始格式（18维）
  - tron2_policy.py 处理后的效果（隐式假设 16 维）
  - 预期的 Tron2 关节顺序（用户期望的 16 维）
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

# ─── 关节名称定义 ─────────────────────────────────────────────────────────

# 数据集中定义的 18 维关节
DATASET_JOINT_NAMES_18 = [
    "abad_L_Joint",          # [0]  左臂髋关节1
    "hip_L_Joint",           # [1]  左臂髋关节2
    "yaw_L_Joint",           # [2]  左臂偏航
    "knee_L_Joint",          # [3]  左臂膝关节
    "wrist_yaw_L_Joint",     # [4]  左腕偏航
    "wrist_pitch_L_Joint",   # [5]  左腕俯仰
    "wrist_roll_L_Joint",    # [6]  左腕滚动
    "left_gripper",          # [7]  左夹爪
    "abad_R_Joint",          # [8]  右臂髋关节1
    "hip_R_Joint",           # [9]  右臂髋关节2
    "yaw_R_Joint",           # [10] 右臂偏航
    "knee_R_Joint",          # [11] 右臂膝关节
    "wrist_yaw_R_Joint",     # [12] 右腕偏航
    "wrist_pitch_R_Joint",   # [13] 右腕俯仰
    "wrist_roll_R_Joint",    # [14] 右腕滚动
    "right_gripper",         # [15] 右夹爪
    "head_pitch_Joint",      # [16] 头部俯仰
    "head_yaw_Joint",        # [17] 头部偏航
]

# tron2_policy.py 中假设的 16 维结构（注释中写的）
# [左臂7, 左夹爪1, 右臂7, 右夹爪1] = 16
# 但代码中 make_bool_mask(7, -1, 7, -1) 的 mask 也是 16 维
TRON2_ASSUMED_16 = [
    "abad_L_Joint",          # [0]  左臂 → delta
    "hip_L_Joint",           # [1]  左臂 → delta
    "yaw_L_Joint",           # [2]  左臂 → delta
    "knee_L_Joint",          # [3]  左臂 → delta
    "wrist_yaw_L_Joint",     # [4]  左臂 → delta
    "wrist_pitch_L_Joint",   # [5]  左臂 → delta
    "wrist_roll_L_Joint",    # [6]  左臂 → delta
    "left_gripper",          # [7]  左夹爪 → 绝对值（mask=False）
    "abad_R_Joint",          # [8]  右臂 → delta
    "hip_R_Joint",           # [9]  右臂 → delta
    "yaw_R_Joint",           # [10] 右臂 → delta
    "knee_R_Joint",          # [11] 右臂 → delta
    "wrist_yaw_R_Joint",     # [12] 右臂 → delta
    "wrist_pitch_R_Joint",   # [13] 右臂 → delta
    "wrist_roll_R_Joint",    # [14] 右臂 → delta
    "right_gripper",         # [15] 右夹爪 → 绝对值（mask=False）
    # 注意：头部两个关节（head_pitch, head_yaw）被完全忽略！
]


def load_parquet(parquet_path: str) -> pd.DataFrame:
    """加载单个 LeRobot parquet 文件。"""
    df = pd.read_parquet(parquet_path, engine="pyarrow")
    print(f"  文件: {parquet_path}")
    print(f"  帧数: {len(df)}")
    print(f"  列数: {list(df.columns)}")
    return df


def parse_images(df: pd.DataFrame, step: int):
    """解析指定帧的图像（如果需要显示）。"""
    from PIL import Image
    import io

    images = {}
    for cam in ["cam_high", "cam_left_wrist", "cam_right_wrist"]:
        col = f"observation.images.{cam}"
        if col in df.columns:
            try:
                img_data = df[col].iloc[step]
                images[cam] = np.array(Image.open(io.BytesIO(img_data["bytes"])))
            except Exception:
                images[cam] = None
    return images


def print_section(title: str):
    """打印分隔标题。"""
    print()
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)


def examine_sample(df: pd.DataFrame, step: int = 0):
    """检查单个样本的 state 和 action。"""
    state = np.asarray(df["observation.state"].iloc[step])
    action = np.asarray(df["action"].iloc[step])

    print_section(f"样本 {step} 的原始数据")
    print(f"  state.shape:  {state.shape}")
    print(f"  action.shape: {action.shape}")
    print()

    # ─── 打印带名称的 state ───
    print("  {:>4s}  {:<20s}  {:>12s}  {:>12s}  {:<12s}".format("Idx", "Joint Name", "State", "Action", "Group"))
    print("  ----  --------------------  ------------  ------------  ------------")
    for i in range(18):
        group = ""
        if i <= 6:
            group = "左臂"
        elif i == 7:
            group = "左夹爪"
        elif i <= 14:
            group = "右臂"
        elif i == 15:
            group = "右夹爪"
        else:
            group = "头部"
        print(f"  [{i:2d}]  {DATASET_JOINT_NAMES_18[i]:<20s}  {state[i]:>+12.6f}  {action[i]:>+12.6f}  {group}")

    return state, action


def examine_state_stats(df: pd.DataFrame):
    """检查 state 和 action 在整个序列中的统计信息。"""
    states = np.stack(df["observation.state"])
    actions = np.stack(df["action"])

    print_section("全部帧的统计信息")
    print(f"  states.shape:  {states.shape}  (frames, dims)")
    print(f"  actions.shape: {actions.shape} (frames, dims)")
    print()

    # 每维的统计量
    print("  {:>4s}  {:<20s}  {:>12s}  {:>12s}  {:>30s}  {:>12s}  {:>12s}  {:>30s}".format(
        "Idx", "Joint Name", "State Mean", "State Std", "State Range",
        "Action Mean", "Action Std", "Action Range"))
    print("  ----  --------------------  ------------  ------------  ------------------------------  ------------  ------------  ------------------------------")
    for i in range(18):
        sm, ss = float(states[:, i].mean()), float(states[:, i].std())
        sr_min, sr_max = float(states[:, i].min()), float(states[:, i].max())
        am, as_ = float(actions[:, i].mean()), float(actions[:, i].std())
        ar_min, ar_max = float(actions[:, i].min()), float(actions[:, i].max())
        print(f"  [{i:2d}]  {DATASET_JOINT_NAMES_18[i]:<20s}  {sm:>+12.6f}  {ss:>12.6f}  "
              f"[{sr_min:>+8.4f}, {sr_max:>+8.4f}]  "
              f"{am:>+12.6f}  {as_:>12.6f}  [{ar_min:>+8.4f}, {ar_max:>+8.4f}]")

    return states, actions


def analyze_delta_with_mask(states: np.ndarray, actions: np.ndarray):
    """分析 tron2_policy.py 中的 mask 处理逻辑对数据的影响。

    make_bool_mask(7, -1, 7, -1) 产生：
      [T,T,T,T,T,T,T, F, T,T,T,T,T,T,T, F] 共16个
      前7个左臂关节 → delta（actions -= state）
      第8个左夹爪 → 绝对值（不变）
      再7个右臂关节 → delta
      第16个右夹爪 → 绝对值
    """
    print_section("DeltaActions mask 分析 (make_bool_mask(7, -1, 7, -1))")

    mask = [True] * 7 + [False] + [True] * 7 + [False]
    print(f"  mask长度: {len(mask)}")
    print(f"  mask值:   {mask}")
    print()

    # 对于每一帧，计算 delta 后的 action 值
    print("  {:>4s}  {:<20s}  {:<10s}  {:>14s}  {:>12s}  {:>14s}".format(
        "Idx", "Joint Name", "Delta", "raw action", "state", "delta action"))
    print("  ----  --------------------  ----------  --------------  ------------  --------------")

    # 选几个中间帧
    mid = len(states) // 2
    for i in range(18):
        if i >= len(mask):
            delta_enabled = "N/A (忽略)"
            delta_val = np.nan
        else:
            delta_enabled = "是 (delta)" if mask[i] else "否 (绝对值)"
            delta_val = actions[mid, i] - (states[mid, i] if mask[i] else 0)

        if i >= len(mask):
            print(f"  [{i:2d}]  {DATASET_JOINT_NAMES_18[i]:<20s}  {'超出mask范围':<10s}  "
                  f"{actions[mid, i]:>+14.6f}  {states[mid, i]:>+12.6f}  {'(不处理)':>14s}")
        else:
            print(f"  [{i:2d}]  {DATASET_JOINT_NAMES_18[i]:<20s}  {delta_enabled:<10s}  "
                  f"{actions[mid, i]:>+14.6f}  {states[mid, i]:>+12.6f}  {delta_val:>+14.6f}")


def analyze_pipeline_mismatch(states: np.ndarray, actions: np.ndarray):
    """分析 tron2_policy.py 数据流水线与数据集之间的维度不匹配。"""
    print_section("流水线维度不匹配分析")

    # 数据集 state 维度
    print(f"  1. 数据集 state:  [{states.shape[1]}] 维 ({states.shape[1] - 16} 个头部关节)")
    print(f"  2. 数据集 action: [{actions.shape[1]}] 维")
    print()

    # Tron2Inputs 期望的
    print(f"  3. Tron2Inputs 文档声称:    [16] 维")
    print(f"  4. Tron2Outputs 截取到:     [:, :16]")
    print(f"  5. PadStatesAndActions 填充: [{actions.shape[1]}] → [32]")
    print()

    print(f"  ⚠️ 关键问题：")
    print(f"     - 数据集实际是 {states.shape[1]} 维，包含头部关节 (head_pitch, head_yaw)")
    print(f"     - tron2_policy.py 假设 16 维，头部关节通过 PadStatesAndActions 被包裹到模型输入")
    print(f"     - 但在推理时，Tron2Outputs 只取 [:, :16]，头部关节的预测值被丢弃")
    print(f"     - 头部关节在训练时也参与了 loss 计算（因为 PadStatesAndActions 把它们送到了模型）")
    print(f"     - 夹爪关节（索引 7, 15）被正确识别，且 mask=False（绝对值模式）")
    print()

    # 看看第7维和第15维的实际意义
    print(f"  📊 夹爪关节 (idx=7, 15) 的实际值范围：")
    for idx, name in [(7, "left_gripper"), (15, "right_gripper")]:
        vals = actions[:, idx]
        print(f"       {name:20s}: action 范围 [{float(vals.min()):+.4f}, {float(vals.max()):+.4f}]  "
              f"均值 {float(vals.mean()):+.4f}")
        svals = states[:, idx]
        print(f"       {'':20s}  state  范围 [{float(svals.min()):+.4f}, {float(svals.max()):+.4f}]  "
              f"均值 {float(svals.mean()):+.4f}")
        # 判断是否是夹爪（0或1附近的值）还是连续关节
        is_gripper = float(vals.max() - vals.min()) < 1.0 and float(vals.max()) <= 1.5
        print(f"       {'':20s}  {'→ 像夹爪（二值/小范围）' if is_gripper else '→ 像连续关节'}")
    print()

    # 头部关节的实际值
    print(f"  📊 头部关节 (idx=16, 17) 的实际值范围：")
    for idx, name in [(16, "head_pitch"), (17, "head_yaw")]:
        vals = actions[:, idx]
        print(f"       {name:20s}: action 范围 [{float(vals.min()):+.4f}, {float(vals.max()):+.4f}]  "
              f"均值 {float(vals.mean()):+.4f}")
        svals = states[:, idx]
        print(f"       {'':20s}  state  范围 [{float(svals.min()):+.4f}, {float(svals.max()):+.4f}]  "
              f"均值 {float(svals.mean()):+.4f}")


def check_first_and_last_few_frames(df: pd.DataFrame):
    """查看帧之间动作的连续性用于判断关节类型。"""
    actions = np.stack(df["action"])
    states = np.stack(df["observation.state"])

    print_section("帧间差异分析（delta），判断关节类型")

    # 计算相邻帧之间的 action 差异
    action_diff = np.diff(actions, axis=0)
    print(f"  action 帧间差 shape: {action_diff.shape}")
    print()

    print("  {:>4s}  {:<20s}  {:>16s}  {:<20s}".format("Idx", "Joint Name", "action diff std", "特点"))
    print("  ----  --------------------  ----------------  --------------------")
    for i in range(18):
        diff_std = float(np.std(action_diff[:, i]))
        diff_mean = float(np.mean(np.abs(action_diff[:, i])))
        # 如果几乎没变化（接近0），可能是夹爪或归零的头部关节
        if diff_std < 0.01:
            feature = "几乎恒定 → 可能未使用/归零"
        elif diff_mean < 0.02:
            feature = "微小变化"
        elif diff_mean < 0.1:
            feature = "中等变化 → 可能位置保持"
        else:
            feature = "大幅变化 → 主动运动"
        print(f"  [{i:2d}]  {DATASET_JOINT_NAMES_18[i]:<20s}  {diff_std:>+16.8f}  {feature}")


def show_joint_time_series(states: np.ndarray, actions: np.ndarray):
    """按关节分组显示时间序列的统计特征"""
    print_section("关节分组统计")

    groups = [
        ("左臂关节", slice(0, 7)),
        ("左夹爪", slice(7, 8)),
        ("右臂关节", slice(8, 15)),
        ("右夹爪", slice(15, 16)),
        ("头部关节", slice(16, 18)),
    ]

    for group_name, sl in groups:
        print(f"\n  📍 {group_name} [索引 {sl.start}:{sl.stop}]")
        # 检查状态中这些关节的均值和变化
        s_mean = np.mean(states[:, sl], axis=1) if sl.stop - sl.start > 1 else states[:, sl].flatten()
        a_mean = np.mean(actions[:, sl], axis=1) if sl.stop - sl.start > 1 else actions[:, sl].flatten()
        print(f"      state  整体均值: {float(np.mean(s_mean)):+.4f}  "
              f"标准差: {float(np.std(s_mean)):.4f}  "
              f"范围: [{float(np.min(s_mean)):+.4f}, {float(np.max(s_mean)):+.4f}]")
        print(f"      action 整体均值: {float(np.mean(a_mean)):+.4f}  "
              f"标准差: {float(np.std(a_mean)):.4f}  "
              f"范围: [{float(np.min(a_mean)):+.4f}, {float(np.max(a_mean)):+.4f}]")


def main():
    parser = argparse.ArgumentParser(description="检查 LeRobot v2.1 原始数据的 state/action 维度")
    parser.add_argument("--parquet", type=str,
                        default="/home/punk/yann_repo/tron2/example/lerobot_2026-07-20_09-31-04/data/chunk-000/episode_000000.parquet",
                        help="parquet 文件路径")
    parser.add_argument("--step", type=int, default=0, help="要详细查看的帧索引")
    args = parser.parse_args()

    # 检查文件
    if not os.path.isfile(args.parquet):
        print(f"❌ 文件不存在: {args.parquet}")
        sys.exit(1)

    print("=" * 80)
    print("  LeRobot v2.1 原始数据检查工具")
    print("=" * 80)

    # 加载数据
    df = load_parquet(args.parquet)

    # 1. 查看单帧
    examine_sample(df, args.step)

    # 2. 全序列统计
    states, actions = examine_state_stats(df)

    # 3. 分析 mask 效果
    analyze_delta_with_mask(states, actions)

    # 4. 维度不匹配分析
    analyze_pipeline_mismatch(states, actions)

    # 5. 帧间差异分析
    check_first_and_last_few_frames(df)

    # 6. 分组统计
    show_joint_time_series(states, actions)

    print()
    print("=" * 80)
    print("  结论")
    print("=" * 80)
    print()
    print(f"  • 数据集 observation.state 和 action 都是 {states.shape[1]} 维。")
    print(f"  • 顺序：左臂(7) → 左夹爪(1) → 右臂(7) → 右夹爪(1) → 头部(2)")
    print(f"  • 夹爪关节在索引 [7] 和 [15] 处，被正确识别（mask=False 用绝对值）。")
    print(f"  • 头部关节在索引 [16] 和 [17] 处，超出了 tron2_policy.py 的 16 维假设。")
    print(f"  • tron2_policy.py 中 Tron2Outputs 截取 [:, :16] 会丢弃头部关节输出。")
    print()
    print(f"  ⚠️ 如果你的数据集确实包含有效的头部关节运动，")
    print(f"     那么需要修改 action_dim 和相关代码来正确处理这 18 维数据。")
    print(f"     但如果头部关节始终为 0（未使用），则当前流水线功能上正确。")
    print()


if __name__ == "__main__":
    main()

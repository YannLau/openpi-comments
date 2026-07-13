"""
============================================================
  runpo05test.py — π₀.₅ ALOHA 策略推理测试脚本

  用伪造的观测数据（随机图像 + 假状态）测试 Policy 是否能
  正常加载并完成推理。主要用于验证：
    1. 检查点加载是否正确
    2. 变换流水线是否组装正确
    3. 推理能否正常跑通
============================================================
"""

"""

第一次运行该脚本
后面输出一个进度条，下载大约4MB左右的数据，我很好奇，我明明已经下载了完整预训练权重，为何还会下载东西，这是什么？              
  
                                                  
找到了！就是它：

```python
# tokenizer.py:18
path = download.maybe_download(
    "gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"}
)
```

**这是 PaliGemma 的 SentencePiece 分词器模型文件**，大约 **3-5 MB**，下载后缓存在 `~/.cache/openpi/big_vision/paligemma_tokenizer.model`。

跟你下载的那十几 GB 的**模型权重**完全是两回事。具体加载时间线是：

```
create_trained_policy() 内部
    │
    ├── 加载模型权重 ← 从你的本地 pi05_base/params/ 读取
    │
    ├── data_config = train_config.data.create(assets_dirs, model_config)
    │   └── ModelTransformFactory()(model_config)
    │       └── TokenizePrompt(PaligemmaTokenizer(...))
    │           └── PaligemmaTokenizer.__init__()
    │               └── download.maybe_download("gs://big_vision/paligemma_tokenizer.model")
    │                   ↓
    │        ← 第一次运行才会下载，约 4MB，缓存在 ~/.cache/openpi/ ← 你看到的进度条
    │
    ├── 组装 Policy 变换流水线
    │
    └── 返回 policy ← "策略加载成功！"
```

**为什么模型权重不包含 tokenizer 文件？**

因为 tokenizer 是模型全家桶里一个独立的小文件。`pi05_base.params` 存的是**神经网络参数**（几十亿个浮点数），而 `paligemma_tokenizer.model` 是**文本分词用的词表 + 合并规则**（SentencePiece 格式，几 MB）。这两个东西来源不同：

- 模型参数 → 从 `gs://openpi-assets/checkpoints/pi05_base/params/` 下载的
- 分词器文件 → 从 `gs://big_vision/paligemma_tokenizer.model` 下载的（Google big_vision 项目维护）

它们只是碰巧都在 GCS 上，但来自不同的 bucket。openpi 在代码里对两者分别调用 `maybe_download`，各自独立缓存。

"""


import logging

import numpy as np

from openpi.policies import policy_config as _policy_config
from openpi.policies.aloha_policy import make_aloha_example
from openpi.training import config as _config

# 设置日志级别，方便看到加载过程中的信息
logging.basicConfig(level=logging.INFO)


def main():
    # ========================================================================
    # 第 1 步：创建策略
    #
    # pi05_aloha 配置：
    #   - 模型：π₀.₅ (pi05=True)
    #   - 数据：ALOHA（默认 use_delta_joint_actions=True，
    #     即训练时关节角度转增量，推理时 AbsoluteActions 转回绝对）
    #   - asset_id="trossen"（使用 Trossen 机器人的归一化统计量）
    # ========================================================================
    print("=" * 60)
    print("加载策略（Policy）...")
    print("=" * 60)

    policy = _policy_config.create_trained_policy(
        _config.get_config("pi05_aloha"),
        "/home/punk/yann_repo/para_check_pi0.5/yann_paras/checkpoint/openpi-assets/checkpoints/pi05_base",
    )
    print("策略加载成功！\n")

    # ========================================================================
    # 第 2 步：生成伪造的观测数据
    #
    # make_aloha_example() 返回的格式：
    #   state:  [14]  — 双臂 6+1+6+1 = 14 维（关节角 + 夹爪）
    #   images: dict  — 4 个摄像头: cam_high, cam_low, cam_left_wrist, cam_right_wrist
    #                   每张图像形状 (3, 224, 224)，uint8 类型
    #   prompt: str   — 语言指令
    #
    # 这些原始数据会经过完整的输入变换流水线：
    #   AlohaInputs(图像重排+坐标系对齐) → DeltaActions(绝对→增量)
    #   → Normalize(z-score归一化) → TokenizePrompt(文本分词) → PadStatesAndActions
    # ========================================================================
    print("=" * 60)
    print("生成伪造观测数据...")
    print("=" * 60)

    fake_obs = make_aloha_example()

    print(f"   state 形状: {fake_obs['state'].shape}")
    for name, img in fake_obs["images"].items():
        print(f"   image '{name}' 形状: {img.shape}, dtype: {img.dtype}")
    print(f"   prompt: \"{fake_obs['prompt']}\"")
    print()

    # ========================================================================
    # 第 3 步：执行推理
    #
    # policy.infer(obs) 的工作流程：
    #   1. 输入变换（见上）
    #   2. 添加 batch 维度，转为 JAX Array
    #   3. 调用模型（流匹配去噪生成动作）
    #   4. 去除 batch 维度，转回 NumPy
    #   5. 输出变换：Unnormalize → AbsoluteActions(增量→绝对) → AlohaOutputs
    #
    # 返回的字典包含：
    #   - actions:  [action_horizon, 14]  — 预测的动作序列
    #   - state:    [14]                  — 原始状态（透传）
    #   - policy_timing: dict             — 推理耗时
    # ========================================================================
    print("=" * 60)
    print("执行推理...")
    print("=" * 60)

    result = policy.infer(fake_obs)

    actions = result["actions"]
    timing = result["policy_timing"]

    print(f"\n推理完成！")
    print(f"  推理耗时: {timing['infer_ms']:.1f} ms")
    print(f"  actions 形状: {actions.shape}")
    print(f"  actions 范围: [{actions.min():.4f}, {actions.max():.4f}]")
    print(f"  actions[:3] (前3步):")
    for i in range(min(3, actions.shape[0])):
        print(f"    第 {i} 步: {actions[i]}")
    print(f"  actions[-1] (最后1步): {actions[-1]}")
    print()

    # 统计 summary
    print("=" * 60)
    print("动作统计摘要")
    print("=" * 60)
    action_dim = actions.shape[-1]  # 应该为 14
    for d in range(action_dim):
        col = actions[:, d]
        if d == 6 or d == 13:
            # 夹爪维度（维度 6 和 13 是夹爪）
            print(f"   夹爪 {d}: 均值={col.mean():.4f}, 范围=[{col.min():.4f}, {col.max():.4f}]")
        else:
            # 关节维度
            print(f"   关节 {d:2d}: 均值={col.mean():.4f}, 范围=[{col.min():.4f}, {col.max():.4f}]")

    print("\n测试完成！")


if __name__ == "__main__":
    main()

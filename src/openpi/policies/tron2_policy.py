import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


def make_aloha_example() -> dict:
    """Creates a random input example for the Aloha policy."""
    return {
        "state": np.ones((14,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_low": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class Tron2Inputs(transforms.DataTransformFn):
    """Inputs for the Aloha policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [16]
    - actions: [action_horizon, 16]
    """

    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. But tron2 is unknown for joint space
    adapt_to_pi: bool = False

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")
    
    
    """ use_delta_joint_action=False  是否开启的本质是什么？为什么 @src/openpi/training/config.py 中的aloha仿真（pi0_aloha_sim）和pi05_tron_example不需要设置True？  
  我的理解是，如果是微调任务，基座模型比如pi05_base权重文件在当时训练如果用了delta_joint，后面的微调数据集是一定也要使用delta_joint的，对吗？  
  
    好问题，让我从代码和数据流的角度彻底讲清楚。

---

## 一、`use_delta_joint_action` 的本质

它是一个 **数据预处理开关**，控制是否在训练/推理流水线中插入一对互逆的变换：

| 开关      | 输入变换（训练）                               | 输出变换（推理）                                  |
| --------- | ---------------------------------------------- | ------------------------------------------------- |
| **True**  | `DeltaActions`: `actions[关节] -= state[关节]` | `AbsoluteActions`: `actions[关节] += state[关节]` |
| **False** | 无变换（直接用原始数据）                       | 无变换（模型输出直接执行）                        |

核心代码（`transforms.py:431-446`）：

```python
# DeltaActions 做的就这一行：
actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
# 即：对 mask=True 的维度，从 actions 中减去 state（当前关节角）
# 结果 = "相对于当前位置，还要移动多少"
```

所以开关的本质不是模型参数，而是 **决定模型学到的是"绝对目标位置"还是"相对于当前位置的偏移量"**。

---

## 二、为什么 `pi0_aloha_sim` 和 `pi05_tron_example` 不需要开？

先说结论：**取决于你的数据集里存的是什么格式的 action + 你的机器人底层控制器要什么格式。**

### 1. `pi0_aloha_sim` 的情况

对比两行 config：

| 配置                  | dataset         | 底层控制器       | 数据集 action 格式 | use_delta_joint_actions |
| --------------------- | --------------- | ---------------- | ------------------ | ----------------------- |
| `pi0_aloha_pen_uncap` | 实物 ALOHA 数据 | **绝对位置**控制 | 存的是绝对关节角   | ✅ True (abs→delta)     |
| `pi0_aloha_sim`       | MuJoCo 仿真     | **增量位置**控制 | 存的已经是 delta   | ❌ False (已是delta)    |

**真实 ALOHA 机器人**：底层电机控制器接收**绝对关节角**（"移动到 0.5 rad"），所以数据集里 actions 存的是绝对关节角。

**ALOHA Sim（MuJoCo）**：仿真环境底层接收**增量关节角**（"关节再转 0.02 rad"），所以数据集 `lerobot/aloha_sim_transfer_cube_human` 里的 actions 存的**已经是 delta 值了**。

如果你对 sim 数据再把 `use_delta_joint_actions=True`，那就等于 `delta - state = 二次delta`，相当于对一个已经是增量的值再减去状态——得到错误结果。

### 2. `pi05_tron_example` 的情况

Tron2 的 action 空间是 **16 维**（7+1+7+1），和 ALOHA（6+1+6+1=14维）不同。

```python
# LeRobotTronDataConfig 中的 delta mask 定义
delta_action_mask = _transforms.make_bool_mask(7, -1, 7, -1)
# 结果是：[T]*7 + [F] + [T]*7 + [F] = 16 维
```

但你设了 `use_delta_joint_action=False`，所以这段代码根本不会执行。原因可能是：

1. **你收集的 Tron2 数据集，actions 已经是 delta 格式了**（同 sim 一样）
2. **或者 Tron2 的控制器直接用绝对关节角**，模型直接输出绝对位置效果就很好
3. **或者你的数据 action 格式特殊**（比如速度控制），delta 转换在语义上没有意义

无论哪种，结论都是：**当前数据的 action 格式已经可以直接喂给模型训练，不需要额外转换。**

---

## 三、你的核心问题：微调时是否必须和基座模型一致？

> "如果基座模型比如pi05_base权重文件在当时训练如果用了delta_joint，后面的微调数据集是一定也要使用delta_joint的，对吗？"

### 代码层面的直接证据：不需要

看一下 `_CONFIGS` 中的两个配置，它们**都从同一个 `pi0_base` 权重加载，但 delta 设置不同**：

```python
# 配置 A: pi0_aloha_sim  → use_delta_joint_actions=False
TrainConfig(
    name="pi0_aloha_sim",
    data=LeRobotAlohaDataConfig(
        repo_id="lerobot/aloha_sim_transfer_cube_human",
        use_delta_joint_actions=False,          # ← 不开
    ),
    weight_loader=CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),  # ← 同一个权重
    num_train_steps=20_000,
)

# 配置 B: pi0_aloha_pen_uncap → use_delta_joint_actions=True（默认值）
TrainConfig(
    name="pi0_aloha_pen_uncap",
    data=LeRobotAlohaDataConfig(
        repo_id="physical-intelligence/aloha_pen_uncap_diverse",
        default_prompt="uncap the pen",
        # use_delta_joint_actions 取默认值 True       ← 开
    ),
    weight_loader=CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),  # ← 同一个权重
    num_train_steps=20_000,
)
```

**从同一个 `pi0_base` 出发，一个开 delta 一个不开，两个都工作。** 这直接反驳了"必须一致"这个说法。

### 为什么不需要一致？

关键在于**`weight_loader` 只加载参数的初始值，不固定模型的输出格式**：

```
         pi0_base 参数（初始值）
              │
     ┌────────┴────────┐
     ▼                 ▼
  微调+delta        微调无delta
  模型学增量动作    模型学绝对动作
     │                 │
     ▼                 ▼
  推理时用           推理时用
  AbsoluteActions    原始输出
  把delta还原为absolute
```

模型只是一个函数逼近器。训练 20k 步足够让参数调整到适应新的动作格式。pi0_base 本身在预训练阶段就处理过**多种多样**的动作表示——不同机器人平台有不同格式，所以模型参数天然具有适应不同动作格式的灵活性。

### Flow Matching 的特殊性

π₀ 用 **flow matching** 训练——模型学习从噪声到数据分布的"流场"。如果你切换了动作格式（比如从 delta 切换到 absolute），数据分布变化很大：

```
delta 动作分布:  centered around 0, 范围约 [-0.1, 0.1]  ← 集中
absolute 动作分布: 覆盖 [-π, π] 整个关节范围               ← 分散
```

这意味着流场需要大幅调整。但 **20k 步的微调完全可以做到这一点**（证据就是上面两个配置都能正常工作）。

### 真正决定是否开 delta 的三个因素

| 因素           | 问题                                   | 例子                                                 |
| -------------- | -------------------------------------- | ---------------------------------------------------- |
| **数据集格式** | 你的 actions 存的是绝对还是增量？      | ALOHA Sim 存 delta → 不开；ALOHA Real 存绝对 → 开    |
| **控制器类型** | 机器人底层吃绝对位置还是增量？         | 实物吃绝对位置 → 推理时需要 AbsoluteActions 转换回来 |
| **训练稳定性** | delta 将预测范围压缩到小值，训练更容易 | 精细操作场景（插笔帽）开 delta 更稳定                |

---

## 一句话总结

**`use_delta_joint_action` 不取决于基座模型，而取决于你的数据集和机器人的控制接口。** pi0_base/pi05_base 参数足够灵活，微调时可以对同一个基座模型选择开或不开，只要保证训练时的数据格式和推理时的输出变换一致即可。
    """
    

    def __call__(self, data: dict) -> dict:
        data = _decode_tron2(data, adapt_to_pi=self.adapt_to_pi)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        # Assume that base image always exists.
        base_image = in_images["cam_high"]

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        # Add the extra images.
        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": data["state"],
        }

        # Actions are only available during training.
        if "actions" in data:
            actions = np.asarray(data["actions"])
            actions = _encode_actions_inv(actions, adapt_to_pi=self.adapt_to_pi)
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class Tron2Outputs(transforms.DataTransformFn):
    """Outputs for the Tron policy."""

    # If true, this will convert the joint and gripper values from the standard Tron space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi: bool = False

    def __call__(self, data: dict) -> dict:
        # Only return the first 16 dims.
        actions = np.asarray(data["actions"][:, :16])
        return {"actions": _encode_actions(actions, adapt_to_pi=self.adapt_to_pi)}


def _joint_flip_mask() -> np.ndarray:
    """Used to convert between aloha and pi joint angles."""
    return np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])


def _normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def _unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def _gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with pi0 which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = _unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return np.arcsin(np.clip(value, -1.0, 1.0))

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # pi0 gripper data is normalized (0, 1) between encoder counts (2405, 3110).
    # There are 4096 total encoder counts and aloha uses a zero of 2048.
    # Converting this to radians means that the normalized inputs are between (0.5476, 1.6296)
    return _normalize(value, min_val=0.5476, max_val=1.6296)


def _gripper_from_angular(value):
    # Convert from the gripper position used by pi0 to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # We do not scale the output since the trossen model predictions are already in radians.
    # See the comment in _gripper_to_angular for a derivation of the constant
    value = value + 0.5476

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return _normalize(value, min_val=-0.6213, max_val=1.4910)


def _gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = _unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return value - 0.5476


def _decode_tron2(data: dict, *, adapt_to_pi: bool = False) -> dict:
    # state is [left_arm_joint_angles, right_arm_joint_angles, left_arm_gripper, right_arm_gripper]
    # dim sizes: [7, 7, 1, 1]
    state = np.asarray(data["state"])
    state = _decode_state(state, adapt_to_pi=adapt_to_pi) # no use

    def convert_image(img):
        img = np.asarray(img)
        # Convert to uint8 if using float images.
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        # Convert from [channel, height, width] to [height, width, channel].
        return einops.rearrange(img, "c h w -> h w c")

    images = data["images"]
    images_dict = {name: convert_image(img) for name, img in images.items()}

    data["images"] = images_dict
    data["state"] = state
    return data


def _decode_state(state: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        # Flip the joints.
        state = _joint_flip_mask() * state
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        state[[6, 13]] = _gripper_to_angular(state[[6, 13]])
    return state


def _encode_actions(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        # Flip the joints.
        actions = _joint_flip_mask() * actions
        actions[:, [6, 13]] = _gripper_from_angular(actions[:, [6, 13]])
    return actions


def _encode_actions_inv(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        actions = _joint_flip_mask() * actions
        actions[:, [6, 13]] = _gripper_from_angular_inv(actions[:, [6, 13]])
    return actions

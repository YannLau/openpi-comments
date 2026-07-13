import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_droid_example() -> dict:
    """Creates a random input example for the Droid policy."""
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(7),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


"""

❯     fake_obs = make_aloha_example() 这里的数据可以直接用吗？难道不需要进行 repack_transforms 才能用吗？如果是直接能用的，为何 droid_policy.py中的            
  def make_droid_example() -> dict:                                                                                                                            
      "Creates a random input example for the Droid policy."                                                                                         
      return {                                                                                                                                                 
          "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),                                                     
          "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),                                                          
          "observation/joint_position": np.random.rand(7),                                                                                                     
          "observation/gripper_position": np.random.rand(1),                                                                                                   
          "prompt": "do something",                                                                                                                            
      }获取的假数据结构不同，也没有使用repack_transforms进行转换？

这两个 `make_*_example()` 返回的都是各 platform 的 **canonical format（标准格式）**，不是数据集原始格式。它们之间**没有设计不一致**，只是 ALOHA 和 DROID 的硬件观测命名不同而已。

## 为什么不需要 repack_transforms

**`repack_transforms` 只在训练时有用**，推理时根本不需要。看完整的变换链就知道了：

```
训练时:
  LeRobot 数据集格式                    ← observation.images.top, observation.state, action
    │  repack_transforms               ← 把数据集的键名映射到平台标准格式        !!!!!!!!!!!!!!!!注意这里 LeRobot-> 平台标准（Aloha、Libero、droid）->模型标准
    ▼
  平台标准格式 (canonical format)
    │  data_transforms.inputs           ← AlohaInputs / DroidInputs
    │  Normalize
    │  model_transforms
    ▼
  模型

推理时:
  策略服务器/用户直接提供               ← 已经是平台标准格式了
    │  repack_transforms               ← 空的！啥也不做
    │  data_transforms.inputs           ← AlohaInputs / DroidInputs
    │  Normalize
    │  model_transforms
    ▼
  模型
```

再读一下 `policy_config.py:286-291` 的代码：

```python
transforms=[
    *repack_transforms.inputs,           # ← 推理时 repack 是用户自选的，不传就是空
    transforms.InjectDefaultPrompt(...),
    *data_config.data_transforms.inputs,  # ← AlohaInputs/DroidInputs 在这里
    ...
]
```

而 `create_trained_policy` 的 `repack_transforms` 参数默认是：

```python
repack_transforms = repack_transforms or transforms.Group()  # 空变换组，什么都不做
```

## 两个 platform 的 make_example 对比

它们都是各自 `*Inputs` 的"原生输入格式"：

|              | `make_aloha_example()` → `AlohaInputs`                                                                                   | `make_droid_example()` → `DroidInputs`                                                                                                                          |
| ------------ | ------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **状态**     | `state: [14]` 一体化的双臂向量                                                                                           | `observation/joint_position: [7]` + `observation/gripper_position: [1]` → 在 `DroidInputs.__call__` 里 `np.concatenate`                                         |
| **图像**     | `images: {cam_high, cam_low, ...}` 字典，**[C,H,W]** 格式 → `AlohaInputs` 内部用 `einops.rearrange("c h w -> h w c")` 转 | `observation/exterior_image_1_left`, `observation/wrist_image_left` 单独字段，已经是 **[H,W,C]** 格式 → `DroidInputs` 的 `_parse_image` 碰到 `shape[0]==3` 才转 |
| **命名风格** | ALOHA 硬件风格：`cam_high`                                                                                               | DROID/RLDS 风格：`observation/joint_position`                                                                                                                   |

**两者都是直接给 `*Inputs` 消费的**，不需要 repack。

## 两个例子的区别来源

它们命名风格不同，只是因为**ALOHA 和 DROID 的硬件长相不同**：

- **ALOHA 双臂**：每臂 6 关节 + 1 夹爪 = 14 维，关节和夹爪天然在一个向量里 → 用 `state: [14]`
- **DROID 单臂**：7 个关节和夹爪分开读取 → `joint_position` 和 `gripper_position` 两个字段，到 `DroidInputs` 里才 `conatenate` 成 `state: [8]`

之所以 DROID 的 example 里还保持 `observation/joint_position` 这种带斜杠的原始键名，就是因为 DROID 的原始数据集就是这格式（RLDS 字段名带 `/`），而 `DroidInputs` 直接设计成消费这个格式了——省去了 repack 这一步。ALOHA 的 `AlohaInputs` 则选择用 `images.cam_high`、`state` 这种更简洁的键名，但核心逻辑一样：**example 返回的就是 Inputs 需要的输入格式**。

所以你的脚本直接用 `make_aloha_example()` 完全正确，不用加 repack。

"""




@dataclasses.dataclass(frozen=True)
class DroidInputs(transforms.DataTransformFn):
    # Determines which model will be used.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        gripper_pos = np.asarray(data["observation/gripper_position"])
        if gripper_pos.ndim == 0:
            # Ensure gripper position is a 1D array, not a scalar, so we can concatenate with joint positions
            gripper_pos = gripper_pos[np.newaxis]
        state = np.concatenate([data["observation/joint_position"], gripper_pos])

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        base_image = _parse_image(data["observation/exterior_image_1_left"])
        wrist_image = _parse_image(data["observation/wrist_image_left"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                # We don't mask out padding images for FAST models.
                images = (base_image, np.zeros_like(base_image), wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class DroidOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 8 dims.
        return {"actions": np.asarray(data["actions"][..., :8])}

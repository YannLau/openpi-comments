"""
ALOHA 仿真环境封装模块

本模块定义了 AlohaSimEnvironment 类，它是 gym_aloha（MuJoCo 仿真）与 openpi
策略推理框架之间的"桥梁"。

职责：
  1. 初始化和管理 MuJoCo 仿真环境（gym_aloha / ALOHA 双机械臂平台）
  2. 将 gym_aloha 的观测格式转换为 openpi 模型期望的格式（图像缩放、轴序转换等）
  3. 支持覆盖仿真场景中 Cube 的初始位姿，用于测试模型的泛化能力

关键数据流：
  Gymnasium 环境 ──原始观测──→ AlohaSimEnvironment ──标准化观测──→ 策略模型
  策略模型输出动作 ──→ AlohaSimEnvironment.apply_action() ──→ Gymnasium 环境步进

注意：
  - 仿真环境使用的是 gym_aloha 库，底层是 dm_control (MuJoCo)
  - 观测类型固定为 "pixels_agent_pos"：同时返回摄像头图像和机械臂关节位置
  - 图像会缩放到 224x224（模型输入要求）并从 [H,W,C] 转成 [C,H,W]（PyTorch/JAX 惯例）
"""

import gym_aloha  # noqa: F401
# ↑ 导入 gym_aloha 会触发其注册（register）动作，将 "gym_aloha/AlohaTransferCube-v0" 等
#   环境 ID 注册到 gymnasium 的环境注册表中。这样后面的 gymnasium.make() 才能找到这些环境。
#   # noqa: F401 告诉 linter：这个导入虽然没在代码中显式使用，但它是必要的（side-effect import）。

import gymnasium  # OpenAI Gymnasium 的标准接口（强化学习环境）
import numpy as np  # 数值计算库

from gym_aloha.constants import normalize_puppet_gripper_position  # ALOHA 夹爪位置归一化函数
from gym_aloha.tasks.sim import BOX_POSE  # 可全局修改的 Cube 位姿列表（箱子的位置和朝向）
from openpi_client import image_tools  # 图像处理工具（缩放、填充、类型转换）
from openpi_client.runtime import environment as _environment  # openpi 运行时框架的环境接口
from typing_extensions import override  # Python 类型提示：显式标记方法覆盖父类


class AlohaSimEnvironment(_environment.Environment):
    """ALOHA 双机械臂 MuJoCo 仿真环境。

    这个类实现了 openpi Runtime 框架中的 Environment 接口（抽象基类），
    因此可以被 Runtime 主循环直接使用。

    它封装了 gym_aloha 库的底层细节，向上层（Runtime）提供一个统一的接口：
      - reset():   重置环境到初始状态，返回标准化观测
      - get_observation():  获取当前观测
      - apply_action():     执行动作，推进仿真一步
      - is_episode_complete(): 检查当前 episode 是否结束

    架构关系：
      Runtime (主循环)
          │
          ▼
      AlohaSimEnvironment  ←── 实现了 Environment 抽象接口
          │
          ▼
      gymnasium.Env (gym_aloha)
          │
          ▼
      dm_control (MuJoCo 物理引擎)
    """

    def __init__(
        self,
        task: str,
        obs_type: str = "pixels_agent_pos",
        seed: int = 0,
        box_pose: list[float] | None = None,
    ) -> None:
        """初始化 ALOHA 仿真环境。

        工作流程：
          1. 设置随机种子（确保环境可重复）
          2. 创建 gym_aloha 环境实例
          3. 初始化内部状态变量

        Args:
            task:       MuJoCo 任务名称，例如 "gym_aloha/AlohaTransferCube-v0"
            obs_type:   观测类型。必须为 "pixels_agent_pos"（图像+关节位置）。
                        其他值（如 "pixels" 或 "state"）在 gym_aloha 中可能未实现。
            seed:       随机种子。控制 cube 初始位置、环境随机性等。
                        固定 seed 可保证每次重置时环境状态一致（用于复现）。
            box_pose:   可选参数，覆盖 cube 的初始位置和朝向。
                        格式：[x, y, z, qw, qx, qy, qz]（位置 + 四元数）。
                        为 None 时使用环境的默认随机位置（由 seed 控制）。
        """
        # ====================================================================
        # 随机种子设置
        #
        # 这里设置了两个随机数生成器：
        #   1. np.random.seed(seed) —— 设置 NumPy 的全局随机种子。
        #      这会影响 gym_aloha 内部使用的随机数（因为 gym_aloha 可能直接用 np.random）。
        #   2. self._rng = np.random.default_rng(seed) —— 创建一个独立的局部 RNG。
        #      这是一个更新的 NumPy RNG（Generator 类），用于我们自己的随机需求。
        #      使用独立的 RNG 可以避免干扰全局随机状态。
        # ====================================================================
        np.random.seed(seed)
        self._rng = np.random.default_rng(seed)

        # 保存 cube 初始位姿覆盖值（如果为 None，则使用环境的默认随机位置）
        self._box_pose = box_pose  # [x, y, z, qw, qx, qy, qz], None=random

        # ====================================================================
        # 创建 gym_aloha 环境
        #
        # gymnasium.make() 会查找注册表中 task 对应的环境类并实例化。
        # 我们固定使用 "pixels_agent_pos" 观测类型，这样既能获取图像数据
        # 用于视觉推理，也能获取关节角度用于状态输入。
        #
        # 可选观测类型说明：
        #   - "pixels":           只返回摄像头图像
        #   - "pixels_agent_pos": 同时返回图像和关节角度（我们需要的）
        #   - "state":            只返回关节角度（纯状态，无需视觉）
        # ====================================================================
        self._gym = gymnasium.make(task, obs_type=obs_type)

        # ---- 内部状态初始化 ----
        self._last_obs = None  # 上一次观测的缓存（get_observation() 返回这个值）
        self._done = True  # 当前 episode 是否已结束（初始为 True，表示还没开始）
        self._episode_reward = 0.0  # 当前 episode 的累积奖励（用于评估表现）

    @override
    def reset(self) -> None:
        """重置环境到初始状态。

        这个流程做了以下几件事：
          1. 用独立的随机种子调用 gym.reset()，确保每次重置的初始状态可控
          2. 如果指定了 box_pose，直接修改 MuJoCo 物理引擎中 cube 的位置
          3. 将 gym 的原始观测转换为 openpi 模型需要的格式

        注意：
          - 即使设置了 box_pose，gym.reset() 仍然会先随机生成一次位置，
            然后我们再用 box_pose 覆盖它。这样做是为了保持 reset() 流程的一致性。
          - 如果 box_pose 为 None，则使用 gym_aloha 的默认随机初始化。
        """
        # ====================================================================
        # 第 1 步：调用 gymnasium 环境的标准 reset()
        #
        # 我们使用 self._rng 生成一个随机种子传给 gym.reset()。
        # 由于 self._rng 的初始状态由 __init__ 中的 seed 决定，
        # 所以每次 reset() 生成的种子序列是确定的（可复现）。
        #
        # 但是要注意：因为 self._rng.integers() 每次 reset 都会推进状态，
        # 所以第 1 次 reset 和第 2 次 reset 得到的种子不同，
        # 这意味着多次运行同一脚本时，虽然初始状态相同，但每次 reset 后的状态不同。
        # 这就是"固定 seed 但每个 episode 初始状态不同"的设计。
        # ====================================================================
        gym_obs, _ = self._gym.reset(seed=int(self._rng.integers(2**32 - 1)))

        # ====================================================================
        # 第 2 步：（可选）覆盖 cube 初始位姿
        #
        # 当指定了 box_pose 时，我们直接修改 MuJoCo 物理引擎中的 cube 状态。
        # 这允许我们测试模型在不同 cube 位置下的泛化能力。
        #
        # 默认随机范围（不覆盖时）：
        #   x 轴: [0, 0.2] 米
        #   y 轴: [0.4, 0.6] 米
        #   z 轴: 固定 0.05 米（桌面高度）
        #
        # box_pose 格式: [x, y, z, qw, qx, qy, qz]
        #   - x, y, z: 三维位置（米）
        #   - qw, qx, qy, qz: 四元数表示的朝向（单位四元数，qw^2+qx^2+qy^2+qz^2=1）
        # ====================================================================
        if self._box_pose is not None:
            # BOX_POSE 是 gym_aloha 模块级变量（一个单元素列表 [None]），
            # 用于在环境外部修改 cube 的位姿。gym_aloha 在 reset 内部会读取这个值。
            BOX_POSE[0] = np.asarray(self._box_pose, dtype=np.float64)

            # 获取 MuJoCo 物理引擎的直接引用
            # self._gym           → gymnasium.Env 包装器
            # self._gym.unwrapped → 原始 AlohaEnv（去掉 gymnasium 的包装层）
            # AlohaEnv._env      → dm_control 的 Environment 对象
            # .physics           → MuJoCo 的 Physics 对象（可直接操作物理状态）
            physics = self._gym.unwrapped._env.physics

            # physics.named.data.qpos 是 MuJoCo 所有关节位置的状态向量，
            # 我们可以通过变量名或索引来访问。
            # qpos[-7:] 取最后 7 个元素，即 cube 的 7 自由度位姿
            # （7 = 3 位置 + 4 四元数朝向）。
            # 这里将 cube 的初始位置硬设为 box_pose 的值。
            physics.named.data.qpos[-7:] = BOX_POSE[0]
            physics.forward()  # 前向传播：更新 MuJoCo 的派生量（如加速度、接触力）

            # ──── 重新构建 gymnasium 观测 ────
            # 因为手动修改了物理状态，需要重新从 MuJoCo 读取状态生成观测，
            # 否则 gym.reset() 返回的观测还是旧的 cube 位置。
            qpos = physics.data.qpos.copy()

            # ALOHA 的动作空间是 14 维：左臂 6 关节 + 左夹爪 1 + 右臂 6 关节 + 右夹爪 1
            # 但 MuJoCo 的原始关节状态略有不同（每个夹爪有 2 个指关节），
            # 所以需要做以下处理：
            #
            #   MuJoCo 原始 qpos 结构：
            #     [0:8]   左臂（6 关节 + 2 指关节）
            #     [8:16]  右臂（6 关节 + 2 指关节）
            #     [16:]   cube 位姿（7 维）
            #
            #   我们需要构造的 agent_pos（14 维）：
            #     [0:6]   左臂关节角度（直接取 qpos[0:6]）
            #     [6]     左夹爪位置（归一化到 [0,1]）
            #     [7:13]  右臂关节角度（直接取 qpos[8:14]）
            #     [13]    右夹爪位置（归一化到 [0,1]）
            #
            # normalize_puppet_gripper_position():
            #   这个函数将夹爪的原始关节角度映射到 [0,1] 范围，
            #   其中 0 = 闭合（夹紧），1 = 张开（松开）。
            agent_pos = np.concatenate([
                qpos[0:6],                                            # 左臂 6 个关节
                [normalize_puppet_gripper_position(qpos[6])],         # 左夹爪（归一化）
                qpos[8:14],                                           # 右臂 6 个关节
                [normalize_puppet_gripper_position(qpos[14])],        # 右夹爪（归一化）
            ])

            # 从顶部摄像头渲染图像（构建观测所需的图像）
            top_img = physics.render(height=480, width=640, camera_id="top")

            # 组装成 gymnasium 格式的观测（与 gym.reset() 返回格式一致）
            gym_obs = {"agent_pos": agent_pos, "pixels": {"top": top_img}}

        # ====================================================================
        # 第 3 步：将 gym 观测转换为 openpi 模型格式，并更新内部状态
        # ====================================================================
        self._last_obs = self._convert_observation(gym_obs)  # type: ignore
        self._done = False  # 标记 episode 开始
        self._episode_reward = 0.0  # 重置累积奖励

    @override
    def is_episode_complete(self) -> bool:
        """检查当前 episode 是否已经结束。

        Runtime 主循环会在每一步之后调用此方法。
        如果返回 True，Runtime 会结束当前 episode 并启动下一个。

        episode 结束的条件：
          - 任务成功（抓取 cube 并放入目标区域）→ terminated=True
          - 任务失败（超时 300 步或其他终止条件）→ truncated=True
          两种情况下 done 都会被设为 True。

        Returns:
            True 表示 episode 已完成，False 表示仍在进行中。
        """
        return self._done

    @override
    def get_observation(self) -> dict:
        """返回当前环境的观测。

        这个观测是经过 _convert_observation() 转换后的格式，
        可以直接输入给策略模型进行推理。

        Returns:
            dict: 包含以下键的观测字典：
                - "state":   机械臂关节角度 + 夹爪位置（14 维数组）
                - "images":  字典 {"cam_high": ...}，包含顶部摄像头图像
                             （形状 [3, 224, 224]，值范围 [0, 255]，uint8）

        Raises:
            RuntimeError: 如果在调用 reset() 之前就调用此方法。
        """
        if self._last_obs is None:
            raise RuntimeError("Observation is not set. Call reset() first.")

        return self._last_obs  # type: ignore

    @override
    def apply_action(self, action: dict) -> None:
        """在仿真环境中执行动作，推进一个时间步。

        这是"环境-策略"交互循环中的关键步骤：
          1. 接收策略模型预测的动作
          2. 调用 gymnasium 环境的 step() 方法执行动作
          3. 更新观测缓存和 episode 状态

        Args:
            action: 动作字典，必须包含键 "actions"。
                    "actions" 的值是一个 14 维数组（由策略模型预测）：
                      - [0:6]    左臂 6 个关节的绝对角度
                      - [6]      左夹爪位置（归一化 [0,1]，0=闭，1=开）
                      - [7:13]   右臂 6 个关节的绝对角度
                      - [13]     右夹爪位置（归一化 [0,1]）

                    注意：这里的动作是"绝对位置"（absolute joint position），
                    即直接指定目标关节角度，不是增量值（delta）。
                    这与配置中 use_delta_joint_actions=False 一致。
        """
        # gymnasium step() 的返回值：
        #   gym_obs:    执行动作后的新观测
        #   reward:     当前步的即时奖励
        #   terminated: 是否达到成功条件（任务完成）
        #   truncated:  是否达到截断条件（如超时 300 步）
        #   info:       附加信息字典（如 is_success）
        gym_obs, reward, terminated, truncated, info = self._gym.step(action["actions"])

        # 转换为 openpi 格式并缓存（供下次 get_observation() 返回）
        self._last_obs = self._convert_observation(gym_obs)  # type: ignore

        # gym_aloha 的奖励机制：
        #   reward == 4 表示成功完成（terminated=True）
        #   truncated 表示超过最大步数（300 步）
        # 任一条件触发，episode 即结束
        self._done = terminated or truncated

        # 累积奖励取最大值，因为一旦成功 reward=4，后面可能变回 0
        # （注意：gym_aloha 的任务中 reward 不是累加的，而是"当前步是否成功"的指示）
        self._episode_reward = max(self._episode_reward, reward)

    def _convert_observation(self, gym_obs: dict) -> dict:
        """将 gymnasium 原始观测转换为 openpi 模型的标准输入格式。

        转换工作包括：
          1. 图像预处理：缩放到 224x224，确保 uint8 类型，轴序 [H,W,C] → [C,H,W]
          2. 字段重命名：将 gym 的键名映射为 openpi 模型期望的键名

        为什么需要这些转换？
          - 224x224: 模型（Paligemma / Pi0）的视觉编码器（SigLIP）要求固定输入尺寸
          - [C,H,W]: PyTorch 和 JAX 都使用"通道优先"（channels-first）的惯例
          - uint8:   确保图像数据范围符合模型预期 [0,255]

        Args:
            gym_obs: gymnasium 格式的原始观测，包含：
                     - "agent_pos": 14 维关节角度 + 夹爪位置
                     - "pixels": {"top": [H, W, C] 格式的 RGB 图像}

        Returns:
            dict: openpi 模型输入格式的观测：
                  - "state":    14 维关节状态（直接透传 gym_obs["agent_pos"]）
                  - "images":   {"cam_high": [C, H, W] 格式的 uint8 图像}
                  其中 C=3（RGB）, H=224, W=224
        """
        # ──── 图像预处理 ────

        # 从顶部摄像头取出原始图像（gym_aloha 默认使用 "top" 摄像头）
        img = gym_obs["pixels"]["top"]

        # 1. 缩放并填充到 224x224（保持宽高比，不足部分用黑色填充）
        #    resize_with_pad: 先等比缩放使最长边 = 224，然后在短边两侧填充 0
        img = image_tools.resize_with_pad(img, 224, 224)

        # 2. 确保图像数据类型为 uint8（值范围 [0, 255]）
        #    如果输入是 float，convert_to_uint8 会做缩放和类型转换
        img = image_tools.convert_to_uint8(img)

        # 3. 转换轴序：从 [H, W, C] → [C, H, W]
        #    numpy 的 transpose(2, 0, 1) 含义：
        #      原形状:   [H, W, C]   索引 0=H, 1=W, 2=C
        #      新形状:   [C, H, W]   新索引 0=C, 1=H, 2=W
        #    所以 transpose(2, 0, 1) 表示：新第 0 轴 ← 原第 2 轴（C），依此类推
        img = np.transpose(img, (2, 0, 1))

        # ──── 组装模型输入 ────
        return {
            # "state": 机械臂关节角度 + 夹爪位置（14 维数组）
            # 这些值直接来自 gym_aloha，已经是归一化后的格式：
            #   6 个左臂关节角度 + 归一化左夹爪 + 6 个右臂关节角度 + 归一化右夹爪
            "state": gym_obs["agent_pos"],

            # "images": {"cam_high": ...} 格式的字典
            # 键名 "cam_high" 是 openpi 模型中定义的摄像头名称，
            # 对应 ALOHA 的"顶部"摄像头（俯视工作台）
            "images": {
                "cam_high": img,  # 形状: [3, 224, 224], dtype: uint8
            },
        }

"""
找到了完整链路！答案很清晰——**uint8 是传输格式，模型内部会自动转成 [-1, 1] 的 float32**（通过model.Observation.from_dict(xxx)）。看数据流：

## 图像全链路追踪

```
env.py（客户端）                             serve_policy.py（服务端）
╔══════════════════════════════════╗        ╔══════════════════════════════════╗
║ MuJoCo 渲染                      ║        ║  model.py: Observation.from_dict()║
║   → 输出 uint8 [0,255]           ║        ║                                  ║
║                                  ║        ║  if uint8:                       ║
║ resize_with_pad()                ║        ║    img = img.astype(float32)      ║
║   → 保持 uint8                   ║        ║         / 255.0   # [0,1]        ║
║                                  ║        ║         * 2.0 - 1 # [-1,1]       ║
║ convert_to_uint8()  ◄── 你问的    ║   ──── ║                                  ║
║   → 确保 uint8                   ║  WebSocket                               ║
║                                  ║        ║  SigLIP 视觉编码器:               ║
║ transpos to [C,H,W]              ║        ║   → 接收 [-1, 1] float32          ║
╚══════════════════════════════════╝        ╚══════════════════════════════════╝
```

**关键代码就是 `model.py:208-209`：**

```python
# 如果图像是 uint8 格式，自动转换为 [-1, 1] 的 float32
data["image"][key] = data["image"][key].astype(np.float32) / 255.0 * 2.0 - 1.0
```

## 所以为什么传 uint8 而不是 float？

**因为 uint8 比 float32 小 4 倍！** `convert_to_uint8()` 的注释也写了：

> "This is important for reducing the size of the image when sending it over the network."
>
> "这对于减少网络传输时的图像大小很重要。"

具体对比：

| 格式        | 一个像素              | 一帧 224x224 图像 | 一个 episode（~300 帧） |
| ----------- | --------------------- | ----------------- | ----------------------- |
| **uint8**   | 1 字节/通道 = 3 字节  | ~150 KB           | ~45 MB                  |
| **float32** | 4 字节/通道 = 12 字节 | ~600 KB           | ~180 MB                 |

WebSocket 传 float32 的话，带宽消耗是 uint8 的 **4 倍**。

## 为什么不直接传 [0,1] float 而要多此一举？

你说"一般都用 0-1 之间的数值表示 RGB"——这只对了一半：

- **模型内部确实用 float**（通常是 [-1, 1] 或 [0, 1]）
- **但传输和存储用 uint8**（更紧凑，且是图像最自然的格式）

这里的设计模式是：

1. **客户端（环境侧）**：保持 uint8 传输，省钱省带宽
2. **服务端（模型侧）**：`Observation.from_dict()` 自动检测 dtype，如果发现是 uint8 就 ÷255×2-1 转成 [-1, 1]
3. **SigLIP 编码器**：接收 [-1, 1] 的 float32 图像

## 如果用 float 传会怎样？

如果把 `convert_to_uint8` 去掉，传入 float32 图像：

- `resize_with_pad()` 的注释说：**"If the image is float32, it must be in the range [-1, 1]."**
- 填充值也会变成 `-1.0`（黑色）而不是 `0`（uint8 的黑色）
- 所以 float 也能传，但带宽翻 4 倍，且需要保证已经是 [-1, 1] 范围

**一句话：uint8 是网络传输的"压缩格式"，服务端模型收到后会自动解压（÷255×2-1）成模型需要的 float32 [-1, 1]。**

"""

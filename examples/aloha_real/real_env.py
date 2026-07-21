"""
ALOHA 真机底层驱动环境 —— 直接与 Interbotix VX300S 机械臂通信的硬件控制层

本模块实现了 RealEnv —— 真实 ALOHA 双机械臂机器人的底层驱动封装。
它是整个 openpi-ALOHA 系统中"离硬件最近"的一层代码。

本文件的定位：
  ┌──────────────────────────────────────────────────────────────┐
  │  Runtime（主循环）                                            │
  │    └─ AlohaRealEnvironment（env.py  —— 适配器层）             │
  │         └─ RealEnv（real_env.py  —— ★ 本文件，驱动层)         │
  │              ├─ InterbotixManipulatorXS SDK → 物理机械臂       │
  │              ├─ Recorder（ROS 订阅器） → 读取关节状态           │
  │              └─ ImageRecorder（ROS 订阅器） → 读取摄像头图像     │
  └──────────────────────────────────────────────────────────────┘

ALOHA 机器人的硬件架构：
  ALOHA 使用"主从式"（Master-Puppet）双臂设计：
    - Master（主臂 / 领导臂）：操作员手持控制，力矩关闭（可被手动拖动）
      通过 ROS 发布 joint_states 供 Puppet 读取或记录数据。
    - Puppet（从臂 / 跟随臂）：接收位置指令，力矩打开（执行动作）
      需要通电并打开力矩才能运动。

  本文件 RealEnv 只控制 Puppet（从臂）—— 它发送位置指令让机器人执行动作。
  Master（主臂）仅在数据采集时使用（由 get_action() 函数读取主臂位姿记录数据）。

代码来源说明：
  此文件大部分代码源自 ACT 项目（https://github.com/tonyzhaozh/act），
  因此跳过了 ruff 代码检查（ruff: noqa）以保持与原版的比例一致性。

术语说明：
  - qpos: joint position（关节位置/角度），单位：弧度（rad）
  - qvel: joint velocity（关节速度），单位：弧度/秒（rad/s）
  - effort: 关节力矩/力，单位：N·m 或 N
  - gripper position: 夹爪开合位置（单位：米或弧度）
  - gripper joint: 夹爪关节角度（单位：弧度）
"""

import collections  # OrderedDict —— 保持观测字典的键顺序
import time  # 睡眠函数 —— 控制每步的执行周期（20ms）
from typing import Optional, List

import dm_env  # DeepMind 环境接口库 —— TimeStep 数据结构
from interbotix_xs_modules.arm import (
    InterbotixManipulatorXS,
)  # Interbotix 机械臂 SDK（控制 VX300S 系列）
from interbotix_xs_msgs.msg import (
    JointSingleCommand,
)  # ROS 消息类型 —— 单个关节指令（用于夹爪）
import numpy as np  # 数值计算库

from examples.aloha_real import constants  # ALOHA 常量定义（夹爪极限位置、归一化函数等）
from examples.aloha_real import robot_utils  # ALOHA 工具函数（Recorder、ImageRecorder、电机设置等）

# ── 默认复位位姿 ──
# 这是 ALOHA 标准运行时的起始关节角度（弧度）。
# [腰部, 肩部, 肘部, 前臂旋转, 手腕角度, 手腕旋转]
#
# 这个位姿选择的原因：
#   - 腰部居中（0）：机械臂对称地指向正前方
#   - 肩部下倾（-0.96）：手臂略微前倾，不会上扬到危险高度
#   - 肘部下弯（1.16）：肘部弯曲，摄像头可以朝下看到工作台
#   - 手腕归零（0, -0.3, 0）：夹爪平行于地面，可准备抓取
#
# 注意这个位姿只包含 6 个臂关节，不包含夹爪的 1 个关节。
# 夹爪的复位在 _reset_gripper() 中单独处理。
DEFAULT_RESET_POSITION = [0, -0.96, 1.16, 0, -0.3, 0]


"""  为何 @src/openpi/training/config.py 中只有aloha平台才有policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]}这个配置？别的平台都没有？另外它的值设置的和   
  @examples/aloha_real/real_env.py 中的 DEFAULT_RESET_POSITION = [0, -0.96, 1.16, 0, -0.3, 0] 还不一样，请你解释下  

现在全看明白了。让我系统回答你。

---

## 一、为什么只有 ALOHA 有 `policy_metadata`？

因为 **ALOHA 是唯一一个在仓库中有完整物理机器人运行时的平台**。

完整的调用链：

```
config.py (定义 reset_pose)
    │  policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]}
    ▼
policy_config.py (存到 Policy 对象中)
    │  metadata=train_config.policy_metadata
    ▼
websocket_policy_server.py (发给 WebSocket 客户端)
    │  作为第一条消息发送给连接的客户端
    ▼
examples/aloha_real/main.py (客户端收到)
    │  metadata = ws_client_policy.get_server_metadata()
    │  reset_position = metadata.get("reset_pose")   ← 第 210 行
    ▼
AlohaRealEnvironment(reset_position=...)
    ▼
RealEnv(self._reset_position)
    │  直接作为电机目标角度调用 move_arms()
    ▼
物理机械臂移动到指定角度
```

**DROID 和 LIBERO 为什么没有？** 不是因为不需要复位，而是因为这两个平台的物理机器人运行时不在这个仓库中。它们的 config 主要是两种用途：

- **推理配置**（`pi0_droid`, `pi05_droid`）：用于 `serve_policy.py` 启动策略服务器，具体连接哪台机器人由外部客户端决定，reset 由客户端自行管理
- **微调配置**（`pi0_libero`, `pi0_fast_libero`）：只用于训练，不涉及物理机器人复位

**ALOHA 是唯一一个在 `examples/aloha_real/` 下有完整"策略服务器 + 环境适配器 + 硬件驱动"三层代码的平台。**

---

## 二、两个值的含义和差异

两个值都在 **原始 ALOHA 关节空间**（不用经过 adapt_to_pi 变换，直接是电机角度）：

| 位置                                   | 值                             | 来源            | 用途                                                              |
| -------------------------------------- | ------------------------------ | --------------- | ----------------------------------------------------------------- |
| `config.py` ALOHA 配置                 | `[0, -1.5, 1.5, 0, 0, 0]`      | openpi 开发团队 | 默认的策略 `policy_metadata`，从 config 下发到客户端              |
| `real_env.py`                          | `[0, -0.96, 1.16, 0, -0.3, 0]` | 原始 ACT 项目   | `RealEnv` 的默认 fallback，当没有从外部传入 reset_position 时使用 |
| `websocket_policy_server.py` 第 150 行 | `[0, -0.96, 1.16, 0, -0.3, 0]` | 同上            | 服务器 metadata 的硬编码默认值                                    |
| `examples/aloha_sim/main.py` 第 326 行 | `[0, -1.5, 1.5, 0, 0, 0]`      | openpi 团队     | 仿真环境的 reset_pose                                             |

**它们关系是：**

```python
# real_env.py — 底层 fallback
DEFAULT_RESET_POSITION = [0, -0.96, 1.16, 0, -0.3, 0]

class RealEnv:
    def __init__(self, *, reset_position=None):
        # 如果外部没有传入，用 DEFAULT_RESET_POSITION
        self._reset_position = reset_position[:6] if reset_position else DEFAULT_RESET_POSITION
```

```
优先级链：
  config.py 的 policy_metadata → websocket server metadata 
      → main.py 读取 → AlohaRealEnvironment(reset_position=)
          → RealEnv._reset_position
              ↓ 如果 metadata 中没有 reset_pose （如 key 不存在）
              → 使用 DEFAULT_RESET_POSITION ([0, -0.96, 1.16, 0, -0.3, 0])
```

**所以程序运行时实际生效的值取决于你怎么启动策略服务器：**

1. **`serve_policy.py --policy.config=pi0_aloha`**：config 中定义了 `policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]}`，metadata 会包含这个值。客户端收到后用 `[0, -1.5, 1.5, 0, 0, 0]`。
2. **直接硬编码启动 WebSocket 服务器**（不传入 metadata）：服务器用默认值 `[0, -0.96, 1.16, 0, -0.3, 0]`。客户端收到这个值。

### 为什么两个值不同？它们是什么物理意义？

看关节索引对应关系（来自 `real_env.py` 注释第 58 行）：

```python
# [腰部, 肩部, 肘部, 前臂旋转, 手腕角度, 手腕旋转]
#    0     1      2       3        4        5
```

| 关节        | config.py / 仿真 | real_env.py / ACT 默认 | 差异含义                                 |
| ----------- | ---------------- | ---------------------- | ---------------------------------------- |
| j0 腰部     | 0                | 0                      | 一致，都居中                             |
| j1 肩部     | **-1.5**         | **-0.96**              | 仿真：肩膀更低（手臂更下探）             |
| j2 肘部     | **1.5**          | **1.16**               | 仿真：肘部更弯（手臂更回收）             |
| j3 前臂旋转 | 0                | 0                      | 一致                                     |
| j4 手腕角度 | 0                | **-0.3**               | 物理机器人：手腕略微下倾（摄像头朝桌面） |
| j5 手腕旋转 | 0                | 0                      | 一致                                     |

**`[0, -0.96, 1.16, 0, -0.3, 0]`** 是物理机器人的已知安全位姿——手臂稍微前倾，手腕略微下看桌面。这来自原始 ACT 仓库的默认值。

**`[0, -1.5, 1.5, 0, 0, 0]`** 是 openpi 团队为仿真环境选择的位姿——肩膀更下沉、肘部更弯曲、手腕水平。仿真中不需要考虑线缆拉扯或碰撞，所以取值范围更大。

### 你注意到 websocket 服务器中的不一致了吗？

看 `websocket_policy_server.py` 第 150 行：

```python
class WebsocketPolicyServer:
    def __init__(self, policy, ..., metadata=None):
        self._metadata = metadata or {}  # ← 如果没有传入 metadata，空字典
```

但**第 150 行的注释**里写的是：
```python
# "reset_pose": [0, -0.96, 1.16, 0, -0.3, 0],
```

这只是注释中的**示例值**，不是实际代码中设置的默认值。如果 `metadata` 参数传入 `None`，`self._metadata = {}` 是**空字典**，客户端 `metadata.get("reset_pose")` 返回 `None`，`RealEnv` 会回退到 `DEFAULT_RESET_POSITION`。

所以实际流程是：
```
serve_policy.py 从 config 获取 policy_metadata → 传给 WebsocketPolicyServer
    → metadata 中的 reset_pose = [0, -1.5, 1.5, 0, 0, 0] → 覆盖默认
    → 客户端 main.py 获取到相同的值
    → RealEnv 用它复位
```

**总结：它们是不同团队在不同阶段选择的两个物理位姿，实际运行时由 `policy_metadata` 从 config 向下传递。如果不传递，物理机器人的 `DEFAULT_RESET_POSITION` 作为安全的 fallback。**
"""

"""
 训练模型的数据集都是从一个固定的起始点开始操作的吗，如果是，那么推理时一定要保证机器人起始状态和训练时数据集起始状态一致吗？ 
 
 
 现在完全搞清楚了。让我给你一个系统性的回答。

---

## 核心发现：仿真中机器人起始位姿是固定的，但 cube 位置是随机的

看 `gym_aloha/tasks/sim.py` 第 108-117 行的 reset 逻辑：

```python
class TransferCubeTask(BimanualViperXTask):
    def initialize_episode(self, physics):
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE   # ← 机器人双臂：每次都固定
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7:] = BOX_POSE[0]      # ← cube 位置：每次都随机
```

`START_ARM_POSE` 的值（`gym_aloha/constants.py:50`）：
```python
START_ARM_POSE = [
    0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239,   # 左臂
    0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239,   # 右臂（完全对称）
]
```

这就是 `real_env.py` 中 `DEFAULT_RESET_POSITION = [0, -0.96, 1.16, 0, -0.3, 0]` 加上夹爪值。

而 cube 位置 `BOX_POSE` 在每个 episode 中通过 `sample_box_pose(seed)` 随机采样：
```
x: [0, 0.2] 米     ← 桌面的前后方向
y: [0.4, 0.6] 米   ← 桌面的左右方向
```

**所以训练数据集的初始状态是：机器人双臂永远相同，只有目标物体位置在变化。**

---

## 现在回答你的两个问题

### Q1: 训练数据的起始位姿是固定的吗？

**仿真中：是固定的。** `START_ARM_POSE` 硬编码，训练数据就是在这个位姿下开始采集的。

**真实 ALOHA：近似固定。** 人类操作员在每个 episode 开始前通过 reset 按钮把机器人回到同一个大致位姿，但存在微小的操作误差。

**其他平台（DROID, LIBERO）：不一定固定。** 这些数据集通常包含大量不同场景下的数据，起始位姿天然有变化。

### Q2: 推理时的起始位姿必须和训练时完全一致吗？

**不必须，而且你对这个问题的直觉可以打消了——我们直接有你的实验证据。**

看你的 `config.py` 中自己加的配置：

```python
# config.py:1420 — 你成功的实验
TrainConfig(name="my_pi0_aloha_sim",
    policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
)
```

你的 `reset_pose = [0, -1.5, 1.5, 0, 0, 0]`，而 gym_aloha 的 `START_ARM_POSE` 是 `[0, -0.96, 1.16, 0, -0.3, 0]`。

**这两个值不同——肩部从 -0.96 变成 -1.5（更下沉），肘部从 1.16 变成 1.5（更弯曲）——但你仍然成功拿起了 cube。** 这直接证明：起始位姿不一样也可以工作。

---

## 为什么不一样也可以工作？三个原因

### 原因 1：模型不是从"记忆训练轨迹"工作的

模型每步推理时，都**输入当前的 state + images**。它知道你当前在哪，不需要猜测。

```
你担心的"固定起始"模式：
  ┌─────────┐    ┌─────────┐    ┌─────────┐
  │ 位姿 A  │───→│  动作1  │───→│  位姿 B │───→...
  └─────────┘    └─────────┘    └─────────┘
  模型只会从 A 出发
```

```
实际的闭环控制模式：
  ┌────────────────┐
  │  当前位置 + 视觉 │───→ 模型推理 → 下一步动作
  └────────────────┘                │
           ▲                        ▼
           └──────── 执行 + 物理反馈 ──┘
  模型每次都"看"到当前位置，自适应地选择动作
```

### 原因 2：视觉信号提供了"目标导向"

模型看到的不是 "state=[0, -0.96, 1.16, ...]" 这么简单。它还看到了 `cam_high`、`cam_left_wrist` 等摄像头图像。图像中包含 cube 的当前位置。

**当你把 cube 放在不同位置时，模型已经泛化过了（训练时 cube 位置就是随机的）。** 所以机器人起始位姿不同，本质上是同一个泛化问题——视觉告诉模型目标在哪，状态告诉模型手在哪，模型推断出"从当前位置到目标需要什么动作"。

### 原因 3：闭环操作（action chunking + receding horizon）

ActionChunkBroker 的模式：模型每 N 步重新推理一次，每次都**基于当前位置和当前视觉**重新规划。这天然提供了对初始偏差的校正能力：

```
第 1 步推理: 当前位置 = [0, -1.5, 1.5, ...] (你的 pose)
            → 预测 50 步动作 (chunk)
            → 执行 actions[0]

第 10 步推理: 当前位置变了 (手已经移动了)
             → 基于新位置再次预测
             → 重新规划剩余轨迹
```

每次重新推理都是基于真正的当前位置，所以初始偏差在几步之内就会被校正。

---

## 那什么情况下起始位姿不一样会出问题？

主要有三种情况：

| 场景                      | 为什么失败                                                                       | 代码中的例子                                                                          |
| ------------------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| **机器人遮住了物体**      | 起始位姿不同 → 手臂挡住了工作区的 cube → 模型看不到目标 → 无法决策               | 如果你的 `reset_pose` 让手直接挡在摄像头前面                                          |
| **state 完全超出分布**    | 模型从未见过某个关节值范围 → 预测无意义                                          | 比如你把肩膀抬到 +1.5（向天上指），训练数据里从来没有这个角度                         |
| **非增量模型 + 大幅偏离** | 用绝对动作的模型预测的是"目标绝对位置"，如果当前偏差太大，预测的目标位置可能不对 | 这就是为什么你的 `use_delta_joint_actions=False` 时仍然能工作——5 个关节的偏差不算太大 |

你的 `reset_pose = [0, -1.5, 1.5, 0, 0, 0]` 成功的原因是：虽然具体数值不同，但**整体物理姿态是相似的**（手臂指向桌面正前方，稍微前倾），摄像头看到的工作区画面是相似的。模型看到"一样的场景 + 稍微不同的手位置"→ 仍然知道怎么抓。

**你失败的 `yann_pi0_aloha_sim`（"根本不遵从指令"）不是因为起始位姿，而是因为 `default_prompt="Push the cube out desk"`——模型的 prompt 变了，它去执行"推出"动作而不是"抓取"动作了。这和起始位姿无关。**

---

## 一句话总结

> **不需要完全一致。模型每步都接收 state + images 作为输入，它是一个闭环控制器而不是开环轨迹播放器。只要起始位姿大致在训练数据的分布范围内（手臂不要指向完全意外的方向）、摄像头画面没被遮挡，模型就能自适应地调整。** 你的实验已经证明了这个结论——`[0, -1.5, 1.5, ...]` 和训练集的 `[0, -0.96, 1.16, ...]` 明显不同但依然成功抓取。
 
"""




class RealEnv:
    """ALOHA 真实机器人环境 —— 最底层的硬件驱动封装。

    这个类封装了两个 Interbotix VX300S 机械臂（左右各一）以及
    4 个摄像头的控制逻辑。它直接通过如下方式与硬件交互：
      - InterbotixManipulatorXS SDK：发送臂和夹爪的位置指令
      - ROS 话题订阅（通过 Recorder）：读取关节状态（位置、速度、力矩）
      - ROS 话题订阅（通过 ImageRecorder）：读取摄像头图像

    动作空间（14 维 —— 为双臂的 6 关节 + 1 夹爪）：
      indices    内容             单位                 说明
      ────────────────────────────────────────────────────────────────
      0-5       左臂 6 关节角度   弧度（rad）          绝对关节位置
      6          左夹爪开合值      归一化 [0, 1]        0=闭合, 1=张开
      7-12      右臂 6 关节角度   弧度（rad）          绝对关节位置
      13         右夹爪开合值      归一化 [0, 1]        0=闭合, 1=张开

    观测空间：
      键          形状             说明
      ────────────────────────────────────────────────────────────────
      qpos       (14,)           关节位置 + 归一化夹爪位置
      qvel       (14,)           关节速度 + 归一化夹爪速度
      effort     (14,)           关节力矩
      images     dict[4]         4 个摄像头图像 (480x640x3, uint8)

    注意：
      1. 夹爪在观测中使用"归一化位置"（0~1），在控制中使用"关节角度"（弧度）
      2. RealEnv 使用 Recorder 读取状态（ROS 订阅），使用 Interbotix SDK 发送指令
      3. RealEnv 不直接管理 ROS 节点（由 AlohaRealEnvironment 通过 init_node 控制）

    参考：
      - Interbotix VX300S 规格书（6-DoF 机械臂 + 1-DoF 夹爪）
      - dm_env.TimeStep 规范（DeepMind Reinforcement Learning 环境接口）
    """

    def __init__(
        self,
        init_node,
        *,
        reset_position: Optional[List[float]] = None,
        setup_robots: bool = True,
    ):
        """初始化真实机器人环境。

        构造函数做了以下事情：
          1. 创建两个 InterbotixManipulatorXS 对象（左右机械臂的 SDK 控制接口）
          2. 创建两个 Recorder 对象（左右臂的 ROS 关节状态订阅器）
          3. 创建 ImageRecorder 对象（摄像头的 ROS 图像订阅器）
          4. 初始化夹爪控制消息
          5. 可选地执行电机初始配置（setup_robots）

        两种控制接口的区别：
          InterbotixManipulatorXS（puppet_bot_*）vs Recorder（recorder_*）：

            InterbotixManipulatorXS（写操作）  Recorder（读操作）
            ┌──────────────────────────────┐  ┌────────────────────┐
            │ arm.set_joint_positions()    │  │ .qpos（从ROS订阅） │
            │   → 发送位置指令给电机        │  │   ← 从电机编码器读│
            │ gripper.core.pub_single      │  │ .qvel / .effort   │
            │   → 发送夹爪指令到 ROS 话题   │  └────────────────────┘
            └──────────────────────────────┘

          InterbotixManipulatorXS 主要用于"写"（发送指令），
          Recorder 主要用于"读"（通过 ROS 订阅接收传感器反馈）。
          它们访问的硬件相同，只是接口不同。

        Args:
            init_node: 是否在此构造函数中创建 ROS 节点（rospy.init_node）。
                       - True：由 RealEnv 创建 ROS 节点（主节点）
                       - False：假设外部已有一个 ROS 节点
                       在同一进程中，rospy.init_node() 只能被调用一次。
                       因此右臂、Recorder、ImageRecorder 都传入 False，
                       复用左臂 init_node=True 时创建的主节点。

            reset_position: 机器人复位位姿（6 个臂关节角度，弧度）。
                            为 None 时使用 DEFAULT_RESET_POSITION。
                            取前 6 个值作为臂关节目标，夹爪部分在
                            _reset_gripper() 中独立处理。

            setup_robots: 是否在初始化时执行电机配置（重启夹爪、设置
                          运行模式、开启力矩）。
                          - True：自动执行 setup_robots()（正常情况）
                          - False：跳过，在外部手动调用 setup_robots()
                                   （需要精细控制初始化顺序时使用）
        """
        # ── 复位位姿 ──
        # 只取前 6 个值（臂关节），丢弃可能的夹爪部分
        self._reset_position = reset_position[:6] if reset_position else DEFAULT_RESET_POSITION

        # ================================================================
        # 创建 Interbotix 机械臂控制对象
        # ================================================================
        # InterbotixManipulatorXS 是 Interbotix 官方 SDK 提供的机械臂控制类。
        # 它封装了：
        #   - arm.set_joint_positions()：设置臂关节位置
        #   - gripper.core.pub_single：用于控制夹爪的 ROS 发布器
        #   - dxl：Dynamixel 电机底层接口（用于扭矩/模式设置、电机复位等）
        #
        # robot_model="vx300s"：Interbotix 的 VX300S 6-DoF 机械臂型号
        # robot_name="puppet_left" / "puppet_right"：ROS 话题名称前缀
        #   左臂话题如：/puppet_left/joint_states
        #   右臂话题如：/puppet_right/joint_states
        #
        # init_node 处理：
        #   左臂创建时 init_node=True → 创建 ROS 节点
        #   右臂创建时 init_node=False → 复用左臂创建的 ROS 节点
        #   （否则重复调用 rospy.init_node() 会报错）
        self.puppet_bot_left = InterbotixManipulatorXS(
            robot_model="vx300s",
            group_name="arm",
            gripper_name="gripper",
            robot_name="puppet_left",
            init_node=init_node,
        )
        self.puppet_bot_right = InterbotixManipulatorXS(
            robot_model="vx300s",
            group_name="arm",
            gripper_name="gripper",
            robot_name="puppet_right",
            init_node=False,  # 复用左臂的 ROS 节点
        )

        # ── 电机初始配置（可选） ──
        # setup_robots 的作用：
        #   1. 重启夹爪电机（清除可能的故障状态）
        #   2. 设置臂关节为"位置控制模式"
        #   3. 设置夹爪为"基于电流的位置控制模式"
        #   4. 打开电机力矩（使机器人保持位置）
        # 这些配置只在程序启动时执行一次。
        if setup_robots:
            self.setup_robots()

        # ================================================================
        # 创建 ROS 传感器订阅器
        # ================================================================
        # Recorder 通过 ROS 订阅 /puppet_{side}/joint_states 话题，
        # 实时接收机械臂的关节位置（qpos）、速度（qvel）、力矩（effort）反馈。
        # 它通过 ROS 回调函数更新内部属性，因此 get_qpos() 等读取方法
        # 返回的是"最近一次 ROS 消息的快照"。
        self.recorder_left = robot_utils.Recorder("left", init_node=False)
        self.recorder_right = robot_utils.Recorder("right", init_node=False)

        # ImageRecorder 通过 ROS 订阅 4 个摄像头的图像话题，
        # 每个摄像头以 ~30Hz 的频率发布 RGB 图像。
        # 它内部使用 CvBridge 将 ROS 图像消息转换为 OpenCV 格式（numpy 数组）。
        self.image_recorder = robot_utils.ImageRecorder(init_node=False)

        # ── 夹爪控制消息（可复用的 ROS 消息对象） ──
        # 在 set_gripper_pose() 中，我们创建一条 JointSingleCommand 消息，
        # 设置其 cmd 字段为目标位置，然后通过 ROS 话题发布。
        # 这里提前创建好消息对象，避免每步都重新分配内存。
        self.gripper_command = JointSingleCommand(name="gripper")

    def setup_robots(self):
        """配置左右两个机械臂的电机参数（启动时执行一次）。

        这个方法执行以下操作：
          1. 重启夹爪电机（清除故障状态）
          2. 设置臂关节运行模式为"位置控制"（position control）
             ── 接收目标位置，用内部 PID 控制器驱动电机到达
          3. 设置夹爪运行模式为"基于电流的位置控制"（current_based_position）
             ── 位置控制为主，但限制最大电流（防止夹碎物体）
          4. 打开电机力矩（使机器人有能力抵抗重力，保持在指令位置）

        如果不调用此方法，电机可能处于：
          - 力矩关闭（torque off）状态 → 手臂因重力下垂
          - 错误状态（fault） → 电机完全不动
          - 错误的控制模式 → 响应异常

        这个方法在构造函数中通过 setup_robots 参数控制是否自动调用。
        """
        robot_utils.setup_puppet_bot(self.puppet_bot_left)
        robot_utils.setup_puppet_bot(self.puppet_bot_right)

    # ==================================================================
    # 传感器读取方法（通过 ROS Recorder 获取）
    # ==================================================================

    def get_qpos(self):
        """获取当前关节位置（"本体感觉"）。

        从 Recorder（ROS 订阅器）中读取最新的关节状态数据，
        提取出双臂的 14 维关节位置向量。

        原始数据格式（来自 ROS JointState 消息）：
          每个臂的 Recorder.qpos 包含 8 个值（索引 0-7）：
            [臂关节1, ..., 臂关节6, ? , 夹爪关节角度]
              0           5         6         7

          注：索引 6 似乎是一个未使用的臂关节取数，实际臂只用了前 6 个索引；
              索引 7 是夹爪的真实关节角度（弧度）。

        输出处理：
          臂关节（索引 0-5）：直接取原始弧度值
          夹爪（索引 7）：通过归一化函数，将关节角度映射到 [0, 1] 范围
            0 = 闭合, 1 = 张开

        为什么夹爪要归一化？
          不同机器人的夹爪行程不同，归一化后可以在不同硬件间共享相同的
          策略模型。训练时，模型学会输出 [0,1] 的值，推理时再反归一化
          为具体电机的实际关节角度。

        Returns:
            np.ndarray shape=(14,): [左臂6关节弧度, 左夹爪归一化位置,
                                      右臂6关节弧度, 右夹爪归一化位置]
        """
        # ── 获取原始数据 ──
        left_qpos_raw = self.recorder_left.qpos  # shape=(8,)
        right_qpos_raw = self.recorder_right.qpos  # shape=(8,)

        # ── 提取臂关节 ──
        # 取前 6 个值（腰部、肩部、肘部、前臂旋转、手腕角度、手腕旋转）
        left_arm_qpos = left_qpos_raw[:6]
        right_arm_qpos = right_qpos_raw[:6]

        # ── 提取夹爪位置并归一化 ──
        # 原始数据中夹爪在索引 7，单位是弧度。
        # PUPPET_GRIPPER_POSITION_NORMALIZE_FN 将弧度映射到 [0,1]：
        #   完全闭合（0.01844 rad）→ 0.0
        #   完全张开（0.05800 rad）→ 1.0
        # 注意这里用的是"位置（position）"归一化而不是"关节（joint）"归一化，
        # 两者的区别在于映射范围不同（见 constants.py）。
        left_gripper_qpos = [
            constants.PUPPET_GRIPPER_POSITION_NORMALIZE_FN(left_qpos_raw[7])
        ]
        right_gripper_qpos = [
            constants.PUPPET_GRIPPER_POSITION_NORMALIZE_FN(right_qpos_raw[7])
        ]

        # ── 拼接为 14 维向量 ──
        # 顺序：[左臂6] + [左夹爪1] + [右臂6] + [右夹爪1]
        return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])

    def get_qvel(self):
        """获取当前关节速度。

        与 get_qpos() 结构相同，但数据来源于 Recorder.qvel。
        速度的单位是弧度/秒（rad/s）。

        夹爪速度归一化的含义：
          PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN 将原始速度除以
          夹爪的行程范围（OPEN - CLOSE）。
          因此归一化后的速度表示"每秒钟夹爪开合范围的多少倍"。
          正值表示张开，负值表示闭合。

        Returns:
            np.ndarray shape=(14,): [左臂6关节速度, 左夹爪归一化速度,
                                      右臂6关节速度, 右夹爪归一化速度]
        """
        left_qvel_raw = self.recorder_left.qvel
        right_qvel_raw = self.recorder_right.qvel
        left_arm_qvel = left_qvel_raw[:6]
        right_arm_qvel = right_qvel_raw[:6]
        left_gripper_qvel = [constants.PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(left_qvel_raw[7])]
        right_gripper_qvel = [constants.PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(right_qvel_raw[7])]
        return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])

    def get_effort(self):
        """获取当前关节力矩（电机输出力）。

        与 get_qpos() 不同，effort 没有夹爪的归一化处理 —— 直接取原始值。
        臂关节取前 7 个值（包括索引 6 的未知关节值）。

        力矩的单位取决于电机型号，通常为 N·m（牛顿·米）。
        这个数据可用于：
          - 检测碰撞（力矩突变）
          - 检测抓取状态（夹爪夹到物体力矩会增大）
          - 监控机器人健康状态（力矩异常 = 可能有问题）

        Returns:
            np.ndarray shape=(14,): 力矩值。前 7=左臂，后 7=右臂。
        """
        left_effort_raw = self.recorder_left.effort
        right_effort_raw = self.recorder_right.effort
        left_robot_effort = left_effort_raw[:7]
        right_robot_effort = right_effort_raw[:7]
        return np.concatenate([left_robot_effort, right_robot_effort])

    def get_images(self):
        """获取 4 个摄像头的最新图像帧。

        委托给 ImageRecorder.get_images()，它会从最近一次 ROS 回调
        中读取缓存的图像数据。

        返回的字典包含 8 个键（4 个摄像头 × 2 种图像类型）：
          - "cam_high":           顶部摄像头 RGB 图像 (480, 640, 3) uint8
          - "cam_high_depth":     顶部摄像头深度图 (480, 640) uint16
          - "cam_low":            底部摄像头 RGB 图像 (480, 640, 3) uint8
          - "cam_left_wrist":     左腕摄像头 RGB 图像 (480, 640, 3) uint8
          - "cam_right_wrist":    右腕摄像头 RGB 图像 (480, 640, 3) uint8
          - 以及对应的 _depth 版本

        注意：深度图会被上一层（AlohaRealEnvironment.get_observation()）
        过滤掉，因为策略模型只使用 RGB 图像。

        Returns:
            dict: 图像字典，键为摄像头名称，值为 np.ndarray (HWC uint8/uint16)。
        """
        return self.image_recorder.get_images()

    # ==================================================================
    # 动作执行方法
    # ==================================================================

    def set_gripper_pose(
        self,
        left_gripper_desired_pos_normalized,
        right_gripper_desired_pos_normalized,
    ):
        """设置两个夹爪的目标位置。

        这个方法将"归一化的夹爪位置"（0=闭合, 1=张开）转换为
        电机能够理解的"关节角度"（弧度），然后通过 ROS 话题发布指令。

        反归一化流程：
          归一化值 (0.0~1.0)
            │
            ▼
          PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN
            │  公式：x * (OPEN_JOINT - CLOSE_JOINT) + CLOSE_JOINT
            │  = x * (1.4910 - (-0.6213)) + (-0.6213)
            │  = x * 2.1123 - 0.6213
            ▼
          关节角度（弧度）：0.0 → -0.6213（闭合），1.0 → 1.4910（张开）

        JointSingleCommand 消息说明：
          - name="gripper"：指定这条命令作用于夹爪（而非臂关节）
          - cmd：目标位置/角度值
          - 通过 ROS 话题 /puppet_{side}/commands/joint_single 发布

        注意：
          - 左右夹爪共享同一个 JointSingleCommand 对象（self.gripper_command），
            每次发送前修改其 cmd 字段。
          - 这个方法立即返回（ROS 发布是非阻塞的）。

        Args:
            left_gripper_desired_pos_normalized:  左夹爪目标 [0, 1]
            right_gripper_desired_pos_normalized: 右夹爪目标 [0, 1]
        """
        # ── 左夹爪 ──
        # 将归一化位置反归一化为关节角度
        left_gripper_desired_joint = constants.PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(
            left_gripper_desired_pos_normalized
        )
        # 设置命令值
        self.gripper_command.cmd = left_gripper_desired_joint
        # 通过 ROS 话题发布（非阻塞发送）
        self.puppet_bot_left.gripper.core.pub_single.publish(self.gripper_command)

        # ── 右夹爪 ──
        right_gripper_desired_joint = constants.PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(
            right_gripper_desired_pos_normalized
        )
        self.gripper_command.cmd = right_gripper_desired_joint
        self.puppet_bot_right.gripper.core.pub_single.publish(self.gripper_command)

    # ==================================================================
    # 复位方法
    # ==================================================================

    def _reset_joints(self):
        """将两个机械臂移动到复位位姿。

        使用 robot_utils.move_arms() 执行插值运动：
          1. 读取当前两个臂的关节位置
          2. 在 1 秒内从当前位置插值到 _reset_position
          3. 每一步（20ms）发送中间位置指令

        插值运动的好处：
          - 平滑：避免机器人"跳"到目标位置
          - 安全：缓慢移动到复位位姿，减少碰撞风险
          - 可预测：1 秒内稳定到达，时序可控

        move_time=1 的含义：用 1 秒完成整个复位动作。
        在 20ms/步的控制频率下，共 50 个插值步。
        """
        robot_utils.move_arms(
            [self.puppet_bot_left, self.puppet_bot_right],
            [self._reset_position, self._reset_position],
            move_time=1,
        )

    def _reset_gripper(self):
        """复位夹爪 —— 先完全闭合再完全张开。

        这个方法实现了"夹爪校准"过程：

        为什么先闭合再张开（而不是先张开再闭合）？
          原始 ACT 代码先张开再闭合，但 Pi 内部发现：
          1. Pi 的 ALOHA 训练数据都是在夹爪张开状态下开始采集的
          2. 如果复位后夹爪保持完全闭合状态，会增加电机报错频率
          3. 先闭合可以清除夹爪的 PWM 控制状态，再张开到起始位置

        具体过程：
          1. 完全闭合（move_time=1 秒）：
             ── 清除之前的 PWM 状态，让电机回到已知的机械位置
             ── 使用 PUPPET_GRIPPER_JOINT_CLOSE = -0.6213 rad
          2. 完全张开（move_time=0.5 秒）：
             ── 到达 episode 的起始夹爪位置（张开）
             ── 使用 PUPPET_GRIPPER_JOINT_OPEN = 1.4910 rad
             ── 0.5 秒比闭合快，因为张开需要克服的阻力更小

        关于夹爪控制模式：
          正常运行时夹爪处于"基于电流的位置控制"模式。
          在复位时，move_grippers 会更改控制模式，复位完成后
          用夹爪的位置控制模式恢复。
        """
        # 第一步：先完全闭合（清除 PWM 状态，复位到已知机械位置）
        robot_utils.move_grippers(
            [self.puppet_bot_left, self.puppet_bot_right],
            [constants.PUPPET_GRIPPER_JOINT_CLOSE] * 2,
            move_time=1,
        )
        # 第二步：再完全张开（达到 episode 起始夹爪位置）
        robot_utils.move_grippers(
            [self.puppet_bot_left, self.puppet_bot_right],
            [constants.PUPPET_GRIPPER_JOINT_OPEN] * 2,
            move_time=0.5,
        )

    # ==================================================================
    # dm_env 兼容接口
    # ==================================================================

    def get_observation(self):
        """收集完整的环境观测数据，打包为 OrderedDict。

        这个方法组合了以下传感器数据：
          1. 关节位置（qpos）：14 维，来自 ROS JointState
          2. 关节速度（qvel）：14 维，来自 ROS JointState
          3. 关节力矩（effort）：14 维，来自 ROS JointState
          4. 摄像头图像（images）：4 个摄像头，来自 ROS 图像话题

        使用 OrderedDict 而不是普通 dict：
          保持键的插入顺序，这在某些场景（如数据序列化）中很重要。
          Python 3.7+ 中普通 dict 也保持插入顺序，但 OrderedDict
          的语义更明确。

        Returns:
            collections.OrderedDict: 包含 qpos、qvel、effort、images 的观测。
        """
        obs = collections.OrderedDict()
        obs["qpos"] = self.get_qpos()
        obs["qvel"] = self.get_qvel()
        obs["effort"] = self.get_effort()
        obs["images"] = self.get_images()
        return obs

    def get_reward(self):
        """获取当前奖励值（始终返回 0）。

        在真实机器人环境中，没有自动化的奖励函数。
        奖励是在数据采集后由人工标注的：
          - 任务成功 = 1
          - 任务失败 = 0
          - 过程中 = 0

        如果在运行时需要奖励信号（如强化学习），
        可以在这里实现基于视觉或力反馈的自动奖励函数。

        Returns:
            int: 始终为 0。
        """
        return 0

    def reset(self, *, fake=False):
        """重置环境到初始状态（物理归位）。

        这个方法执行以下步骤：
          1. 重启夹爪电机（清除故障状态，如过载保护）
          2. 移动机械臂到复位位姿（1 秒内插值到达）
          3. 夹爪先闭合再张开（校准夹爪位置）
          4. 收集复位后的完整观测
          5. 包装为 dm_env.TimeStep 返回

        dm_env.TimeStep 格式：
          - step_type=FIRST：标记这是一个 episode 的起点
          - reward=0：初始奖励始终为 0
          - discount=None：折扣因子无效（非强化学习场景）
          - observation：包含 qpos、qvel、effort、images 的 OrderedDict

        fake 模式：
          当 fake=True 时，跳过所有物理归位操作，仅返回当前状态的观测。
          这在以下场景有用：
            - 调试和测试（只读取数据，不移动机器人）
            - 需要在某个特定姿态下启动（不覆盖当前位姿）
            - 已经手动将机器人移动到目标位姿

        Args:
            fake: 是否执行"假重置"（True=跳过物理归位，仅返回当前观测）。

        Returns:
            dm_env.TimeStep: 包含 step_type=FIRST 和当前观测的时间步。
        """
        if not fake:
            # 1. 重启夹爪电机（清除可能发生的电机故障）
            #    robot_reboot_motors("single", "gripper", True) 中的 True 表示
            #    在重启后启用（enable）电机。电机在过载或位置误差过大时会
            #    进入故障保护状态，需要重启才能清除。
            self.puppet_bot_left.dxl.robot_reboot_motors("single", "gripper", True)
            self.puppet_bot_right.dxl.robot_reboot_motors("single", "gripper", True)

            # 2. 移动臂关节到复位位姿
            self._reset_joints()

            # 3. 校准夹爪（先闭后开）
            self._reset_gripper()

        # 返回包含初始观测的 TimeStep
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation(),
        )

    def step(self, action):
        """在真实机器人上执行一步动作，返回下一时刻的观测。

        这是整个环境中最重要的方法 —— 它执行"动作→观测"的转换。

        执行流程：
          1. 将 14 维动作向量分为左右各 7 维
          2. 发送左臂 6 个关节位置指令（非阻塞）
          3. 发送右臂 6 个关节位置指令（非阻塞）
          4. 发送左夹爪位置指令（非阻塞）
          5. 发送右夹爪位置指令（非阻塞）
          6. 等待 20ms（控制周期，对应 50Hz）
          7. 收集执行后的观测
          8. 返回包含新观测的 TimeStep

        非阻塞发送（blocking=False）：
          Interbotix SDK 的 set_joint_positions() 可以配置为阻塞或非阻塞。
          阻塞模式会等待电机到达目标位置才返回（通常耗时 50-200ms）。
          非阻塞模式立即返回，电机在后台继续运动。

          对于每步 20ms 的高频控制来说，阻塞模式会导致：
            - 控制周期不稳定（阻塞时间不确定）
            - 主循环频率下降（等待时间占主导）
          因此这里使用非阻塞模式，让电机"追赶"目标位置。

        Args:
            action: 14 维动作向量。
                    格式：[左臂6关节, 左夹爪归一化, 右臂6关节, 右夹爪归一化]

        Returns:
            dm_env.TimeStep: 包含 step_type=MID（非起始步）和
                             动作执行后的观测的时间步。
        """
        # ── 分割左右臂动作 ──
        # 14 维向量前后各半：前 7 = 左臂，后 7 = 右臂
        state_len = int(len(action) / 2)  # = 7
        left_action = action[:state_len]  #  左臂：[6关节, 1夹爪]
        right_action = action[state_len:]  # 右臂：[6关节, 1夹爪]

        # ── 设置臂关节位置（非阻塞） ──
        # left_action[:6] 是左臂 6 个关节的目标角度（弧度）
        self.puppet_bot_left.arm.set_joint_positions(left_action[:6], blocking=False)
        self.puppet_bot_right.arm.set_joint_positions(right_action[:6], blocking=False)

        # ── 设置夹爪位置（归一化值 → 关节角度） ──
        # left_action[-1] 是左夹爪位置的归一化值 [0, 1]
        # right_action[-1] 是右夹爪位置的归一化值 [0, 1]
        self.set_gripper_pose(left_action[-1], right_action[-1])

        # ── 等待控制周期完成 ──
        # constants.DT = 0.001（咦，这只有 1ms？）
        #
        # 等一下，constants.DT 定义是 0.001（1ms），但 50Hz 控制需要 20ms。
        # 实际上，这个 DT 是 ALOHA 递归控制的最小时间单位，
        # 在 robot_utils.move_arms() 和 move_grippers() 中作为插值步长使用。
        # 这里的 time.sleep(constants.DT) 只等待 1ms，真正的 20ms 控制周期
        # 由 Runtime 的 max_hz=50 来保证（在 Runtime._step() 中有 sleep 逻辑）。
        # 所以这个 sleep 是"保证至少等待 1ms"，让电机起始运动，不负责完整的 20ms。
        time.sleep(constants.DT)

        # ── 返回执行后的观测 ──
        # step_type=MID：标记这是一个 episode 中间的步骤（非起始也非终结）
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation(),
        )


# ==================================================================
# 工具函数
# ==================================================================


def get_action(master_bot_left, master_bot_right):
    """从主臂（Master）读取操作员动作，用于数据采集（DAgger 等）。

    这个函数在"数据采集模式"下使用 —— 操作员手动操作主臂（Master）时，
    读取主臂的关节位置作为"示范动作"，用于收集训练数据。

    Master vs Puppet 的数据采集流程：
      1. 操作员握住 Master 主臂，做出抓取动作
      2. Master 的电机力矩已关闭（torque off），可被自由拖动
      3. Master 的关节编码器实时读取关节位置
      4. 这个函数读取出这些位置作为"示范动作"
      5. 如果 Puppet 与 Master 力矩同步，Puppet 会复现 Master 的动作

    返回值格式与 RealEnv.step() 的 action 参数完全一致，
    因此采集到的数据可以直接用于训练。

    Master 夹爪与 Puppet 夹爪的差异：
      Interbotix VX300S 的夹爪有两种编码方式：
        - 关节角度（joint）：夹爪电机的实际旋转角度，范围约 -0.68 ~ +0.31 rad
        - 位置（position）：夹爪指尖的直线位移，范围约 0.012 ~ 0.024 m
      主臂使用"关节角度归一化"，从臂使用"位置归一化"。
      这里读取的是主臂的关节角度，需要归一化到 [0, 1]。

    Args:
        master_bot_left:  左主臂的 InterbotixManipulatorXS 对象。
        master_bot_right: 右主臂的 InterbotixManipulatorXS 对象。

    Returns:
        np.ndarray shape=(14,): 与 RealEnv.step() action 格式相同的动作向量。
                                [左臂6关节弧度, 左夹爪归一化值,
                                 右臂6关节弧度, 右夹爪归一化值]
    """
    action = np.zeros(14)

    # ── 臂关节 ──
    # 直接从主臂 Dynamixel 电机的 joint_states 中读取位置
    # master_bot_left.dxl.joint_states.position 包含 7 个值（6 臂 + 1 夹爪）
    # 取前 6 个作为臂关节目标
    action[:6] = master_bot_left.dxl.joint_states.position[:6]
    action[7 : 7 + 6] = master_bot_right.dxl.joint_states.position[:6]

    # ── 夹爪 ──
    # 主臂夹爪的关节角度（索引 6）归一化到 [0, 1]
    # MASTER_GRIPPER_JOINT_NORMALIZE_FN 将关节角度映射到 [0, 1]
    # 注意这里索引 6 和 13 与观测的不同：
    #   观测中夹爪在索引 6（左）和 13（右）
    #   这里也是索引 6（左）和 13（右）—— 一致
    action[6] = constants.MASTER_GRIPPER_JOINT_NORMALIZE_FN(
        master_bot_left.dxl.joint_states.position[6]
    )
    action[7 + 6] = constants.MASTER_GRIPPER_JOINT_NORMALIZE_FN(
        master_bot_right.dxl.joint_states.position[6]
    )

    return action


def make_real_env(
    init_node,
    *,
    reset_position: Optional[List[float]] = None,
    setup_robots: bool = True,
) -> RealEnv:
    """RealEnv 的工厂函数。

    这是一个简单的工厂函数，用于创建 RealEnv 实例。
    使用工厂函数而不是直接构造的好处：
      - 统一创建入口（方便注入依赖或添加前置逻辑）
      - 函数签名比类构造函数的 @dataclass 风格更清晰
      - 可在创建前后添加日志、监控等额外操作

    Args:
        init_node:       是否创建 ROS 节点。
        reset_position:  机器人复位位姿（6 维臂关节角度）。
        setup_robots:    是否执行电机初始配置。

    Returns:
        RealEnv: 配置好的真实机器人环境实例。
    """
    return RealEnv(
        init_node,
        reset_position=reset_position,
        setup_robots=setup_robots,
    )

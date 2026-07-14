"""
ALOHA 仿真环境策略推理入口脚本

本脚本用于在 MuJoCo 仿真环境（gym_aloha / ALOHA）中运行训练好的机器人策略（policy）。
它连接到一个远程策略服务器（WebSocket），获取动作预测，并在仿真环境中执行，
同时录制仿真过程的视频，便于后续观察和调试。

工作流程:
  1. 连接远程策略服务器 → 2. 初始化 ALOHA 仿真环境 → 3. 循环执行: 观察→推理→执行→记录
"""

import dataclasses  # 用于创建简洁的数据容器类（@dataclass）
import logging      # 日志系统，用于输出运行状态信息
import pathlib      # 跨平台路径处理，用于管理输出目录

import env as _env  # ALOHA 仿真环境封装（MuJoCo 任务）
from openpi_client import action_chunk_broker  # 动作块分发器，将推理出的动作块逐帧分发
from openpi_client import websocket_client_policy as _websocket_client_policy  # WebSocket 策略客户端（连接远程推理服务）
from openpi_client.runtime import runtime as _runtime  # "运行时"框架，协调环境、策略和订阅者之间的主循环
from openpi_client.runtime.agents import policy_agent as _policy_agent  # "策略代理"，封装策略的推理逻辑
import saver as _saver  # 结果保存器（视频录制）
import tyro             # 命令行参数解析库（基于类型注解自动生成 CLI）


@dataclasses.dataclass
class Args:
    """命令行参数定义，所有字段均带有默认值，tyro 会自动从字段类型和注解生成 CLI 帮助信息。"""

    # -- 输出设置 ---------------------------------------------------------------
    out_dir: pathlib.Path = pathlib.Path("data/aloha_sim/videos")
    """录制视频的输出目录（默认: data/aloha_sim/videos）"""

    # -- 环境设置 ---------------------------------------------------------------
    task: str = "gym_aloha/AlohaTransferCube-v0"
    """MuJoCo 仿真任务名称（gym_aloha 中的标准任务，如 TransferCube）"""

    seed: int = 44
    """仿真环境的随机种子，设置后保证每次运行的初始状态可复现"""

    # -- 策略设置 ---------------------------------------------------------------
    action_horizon: int = 10
    """动作预测窗口大小 —— 每次推理会预测 action_horizon 步未来的动作，然后逐帧执行（smooth 的滑动窗口方式）"""

    # -- 远程策略服务器连接 ------------------------------------------------------
    host: str = "0.0.0.0"
    """WebSocket 策略服务器的主机地址（0.0.0.0 表示连接本地）"""

    port: int = 8000
    """WebSocket 策略服务器的端口"""

    # -- 可视化设置 -------------------------------------------------------------
    display: bool = False
    """是否弹出仿真渲染窗口（设为 True 会打开 MuJoCo 可视化窗口，方便实时观察；在无图形界面的服务器上请保持 False）"""

    # -- 仿真场景参数 -----------------------------------------------------------
    box_pose: tuple[float, ...] | None = None
    """覆盖 Cube 的初始位置和姿态。
       默认情况下 cube 在 x=[0,0.2], y=[0.4,0.6] 范围内随机生成。
       格式: --box-pose x y z qw qx qy qz  (例如 --box-pose 0.4 0.3 0.05 1 0 0 0)
       其中 qw qx qy qz 是四元数表示朝向。
       设为 None 则使用环境的默认随机位置。
    """
    
"""
❯ 如果这里的seed固定为一个值，那么模型输出的动作，保存的视频将会完全一致吗？我运行同样的命令两次MUJOCO_GL=egl uv run examples/aloha_sim/main.py --args.seed    
  1234567 为何一次成功拿起cube，一次失败了，怎么回事？  
  
现在我已经掌握了所有关键信息。让我给你一个全面的分析。

---

**简短答案：种子只保证了环境初始化是确定性的，但策略推理（远程 GPU 服务器）是非确定性的。** 让我展开说。

## 种子覆盖了什么（确定性的部分）

追踪一下种子从传入到使用的完整链路：

### 1. `aloha_sim/env.py` — 环境创建时

```python
np.random.seed(seed)          # 全局 numpy RNG
self._rng = np.random.default_rng(seed)  # 局部 Generator
```

### 2. `reset()` → gym_aloha → cube 位置采样

`env.py:33` → `self._gym.reset(seed=int(self._rng.integers(2**32 - 1)))`

进入 `gym_aloha/env.py:150`:
```python
self._env.task.random.seed(seed)            # dm_control 随机状态
self._env.task._random = np.random.RandomState(seed)  # 另一个 RNG
BOX_POSE[0] = sample_box_pose(seed)          # cube 位置 ← 确定性采样!
```

`sample_box_pose()` 使用 `np.random.RandomState(seed)` — **传入相同 seed，永远返回相同的 cube 位置**。

### 3. 所以环境初始化是 100% 确定性的

相同 seed → 相同 cube 位置 → 相同初始关节角度 → 相同初始渲染图像 → **相同初始观测** ✅

## 那为什么两次结果不同？（非确定性的来源）

### 🎯 根本原因：远程策略服务器

这个脚本是 **WebSocket 客户端**，真正的策略推理在远端服务器上运行：

```
main.py ──WebSocket──→ 策略服务器 (GPU PyTorch/JAX)
```

策略服务器推理时引入非确定性的原因：

| 来源                   | 说明                                                                                                     |
| ---------------------- | -------------------------------------------------------------------------------------------------------- |
| **GPU 浮点非确定性**   | PyTorch 默认 `torch.backends.cudnn.deterministic=False`，同一输入两次 forward 可能得到**微小不同**的输出 |
| **TF32 / FP16 精度**   | 混合精度训练/推理在 GPU 上不是 bitwise 确定的                                                            |
| **Dropout / 训练模式** | 如果模型意外处于训练模式，每次推理随机丢弃不同的神经元                                                   |
| **服务端负载**         | 两次运行间隔中，服务端 GPU memory 状态、warmup 情况不同                                                  |

### 🎲 微小差异 → 混沌放大

策略输出的动作是**机器人关节角度**。第一次推理哪怕只有 `1e-6` 的差异：

```
第一步动作差 1e-6 弧度 
  → MuJoCo 物理仿真产生不同状态
    → 第二次观测不同
      → 策略下一次推理差异更大
        → ... 指数级发散（蝴蝶效应）
```

300 步后两条轨迹**完全不一样**，一个成功了，一个失败了，这很正常。

### 🔍 其他次要因素

1. **`BoxPose` 覆盖时机** — `aloha_sim/env.py` 在 `gym.reset()` 之后才检查 `box_pose`。但你没设 `--box-pose`，所以这部分不影响。

2. **`nondeterministic=True`** — 这个标记只是给 Gymnasium `check_env` 测试用的，不影响运行时。但它的注释说明了一切：
   ```python
   # Even after seeding, the rendered observations are slightly different,
   # so we set `nondeterministic=True` to pass `check_env` tests
   ```
   说明即便 seed 固定，**渲染结果也可能有微小差异**（GPU rendering 非确定性），这也会影响策略观测。

3. **`time.sleep()` 计时** — `runtime.py` 用 wall-clock 限制 50Hz，系统负载不同会让 sleep 精度不同。但 MuJoCo 仿真步长由 `control_timestep` 控制，不受 wall-clock 影响，所以这个影响较小。

## 如何验证？

### 1️⃣ 排除环境不确定性（确认 seed 生效）
加一行日志看两次的 cube 位置是否相同：

```python
# 在 aloha_sim/env.py reset() 中
print(f"Cube pose: {BOX_POSE[0]}, Seed used: {self._rng.integers(2**32 - 1)}")
```

### 2️⃣ 验证策略服务器是否确定
如果策略服务器是 PyTorch，在服务端推理代码中加入：

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
```

然后重新运行测试。

### 3️⃣ 直接对比两次推理输出
在 `action_chunk_broker.py` 中把第一次推理结果打印出来，看两次是否完全一致：

```python
if self._last_results is None:
    self._last_results = self._policy.infer(obs)
    logging.info(f"First inference actions: {self._last_results}")
```

## 总结

| 组件                                   | 确定性？                         |
| -------------------------------------- | -------------------------------- |
| MuJoCo 环境初始化 (cube位置, 关节角度) | ✅ 种子固定则完全确定            |
| 观测渲染 (render → image)              | ⚠️ GPU 渲染可能略有差异          |
| **远程策略推理 (GPU)**                 | **❌ 默认非确定 — 这是主要原因** |
| MuJoCo 物理仿真执行                    | ✅ 输入相同则输出相同            |

**核心结论：** `seed` 只保证了"环境初始条件一致"，但策略模型推理本身是**非确定性的**（尤其是 GPU 推理）。微小的推理差异在 300 步仿真中被指数放大，导致一次成功一次失败。要想完全复现，必须**同时固定策略服务器的随机种子和 GPU 确定性模式**。

------


你猜对了！**这正是根源所在。** 让我给你看铁证：

## 直接证据：噪声是每次新生成的

### 1️⃣ `policy.py` — 每次推理都"分裂"出新 RNG key

```python
# policy.py:261 — 每次 infer() 必执行
self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
```

这意味着 **每次推理的随机数状态都在变化**，即使 `self._rng` 初始 seed 都是 0。

### 2️⃣ `pi0.py` — 采样时用这个 RNG 生成噪声

```python
# pi0.py:813-814
if noise is None:
    noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
```

**没有传 `noise` 参数 → 每次都从 `jax.random.normal()` 采样全新的噪声。**

### 3️⃣ 流程链

```
第1轮 infer()  →  jax.random.split(key_0) → key_1 + sample_key_1 → 噪声₁（从 sample_key_1 生成）
第2轮 infer()  →  jax.random.split(key_1) → key_2 + sample_key_2 → 噪声₂（从 sample_key_2 生成）
第3轮 infer()  →  jax.random.split(key_2) → key_3 + sample_key_3 → 噪声₃（从 sample_key_3 生成）
...
```

**每一轮推理的噪声都不同** → 每一步动作都不同 → 轨迹指数级发散。

## 所以"两次运行结果不一样"的原因链条

| 层级           | 原因                                                                      |
| -------------- | ------------------------------------------------------------------------- |
| **底层**       | π₀ 是 flow matching 生成模型，需要从随机噪声起步"雕刻"动作                |
| **每次推理**   | `jax.random.split()` 产生不同的 RNG key → 不同的初始噪声                  |
| **第一步动作** | 噪声不同 → 第一步动作有微小差异                                           |
| **后续步骤**   | 由于第一步动作不同，MuJoCo 物理仿真状态不同 → 策略观测不同 → 动作差异放大 |
| **最终结果**   | 完全相同 seed + 完全相同初始环境 → **不同轨迹 → 一次成功一次失败**        |

## 如何让结果完全可复现？

如果你真的想固定两次推理完全一致，需要**同时固定初始噪声**。`noise` 参数就是为此设计的：

```python
# 生成一个固定噪声，每次推理都用同一个
fixed_noise = np.random.randn(1, action_horizon, action_dim)  # [batch, horizon, dim]
runtime.run()  # 所有 infer 调用都从同一个噪声起步... 
```

不过这只是 debug 用的——**实际部署中你通常不希望固定噪声**，因为随机噪声让机器人更具适应性（多模态动作分布），如果噪声固定，遇到预料之外的情况可能会更脆弱。

**一句话：seed 只固定了"环境"，而流匹配的"噪声"每一轮推理都是新的，这才是两次结果不同的根本原因。** 你的怀疑完全正确！



"""



def main(args: Args) -> None:
    """主入口函数：组装并运行策略推理-环境交互循环。"""

    # 如果传入了 box_pose，将元组转为列表（环境接口期望 list 类型）；否则保持 None
    box_pose = list(args.box_pose) if args.box_pose is not None else None

    # ==========================================================================
    # 构建 Runtime（运行时）
    # Runtime 是本项目的核心编排器，它将三个组件连接在一起:
    #   1. Environment —— 机器人仿真/真实环境（提供观察、接收动作）
    #   2. Agent      —— 决策智能体（在这里是策略网络，负责将观察映射为动作）
    #   3. Subscribers —— 可插拔的观察者（如视频录制器，在每个时间步被回调）
    #
    # 主循环逻辑: environment → 观察 → agent 推理 → 动作 → environment.step → 通知 subscribers → 重复
    # ==========================================================================
    runtime = _runtime.Runtime(
        # ---- 环境 ----
        environment=_env.AlohaSimEnvironment(
            task=args.task,          # 选择具体的 ALOHA MuJoCo 任务
            seed=args.seed,          # 随机种子，确保环境初始化可复现
            box_pose=box_pose,       # 可选的 cube 初始位姿覆盖
        ),
        # ---- 智能体 ----
        agent=_policy_agent.PolicyAgent(
            # PolicyAgent 将策略模型包装为一个 agent。
            # 策略内部再次经过一层包装:
            #   ActionChunkBroker ── 接收策略预测的一个"动作块"（连续多步动作），
            #                        然后在后续的每一步中逐步"分发"其中的单步动作，
            #                        同时用新的预测滚动更新剩余动作（类似 MPC 的 receding horizon）。
            #   WebsocketClientPolicy ── 通过 WebSocket 连接远程策略服务器进行推理。
            policy=action_chunk_broker.ActionChunkBroker(
                policy=_websocket_client_policy.WebsocketClientPolicy(
                    host=args.host,
                    port=args.port,
                ),
                action_horizon=args.action_horizon,  # 每次推理预测多少帧的动作
            )
        ),
        # ---- 订阅者（每步回调） ----
        subscribers=[
            # VideoSaver 在每个时间步保存一帧图像，最终合成为视频。
            # 输出目录由 args.out_dir 指定。
            _saver.VideoSaver(args.out_dir),
        ],
        # 最大控制频率（Hz），限制环境 step 的上限速率
        max_hz=50,
    )

    # 启动主循环 —— 阻塞调用，直到 episode 结束或手动中断
    runtime.run()


if __name__ == "__main__":
    # 配置日志输出级别和强制重置（force=True 确保覆盖已有配置）
    logging.basicConfig(level=logging.INFO, force=True)

    # tyro.cli() 会自动解析命令行参数（基于 Args 的类型注解生成 --help 等），
    # 然后调用 main() 函数。
    tyro.cli(main)

"""
❯ 我又遇到一个问题， @examples/aloha_sim/main.py 中的参数    action_horizon: int = 10   ，但是我运行的推理服务器中的模型是 @src/openpi/training/config.py 中的 
     TrainConfig(  ## YannLau   推理  成功了，拿起了cube                                                                                                       
          name="my_pi0_aloha_sim",                                                                                                                             
          model=pi0_config.Pi0Config(),                                                                                                                        
          data=LeRobotAlohaDataConfig(                                                                                                                         
              assets=AssetsConfig(asset_id="lerobot/aloha_sim_transfer_cube_human"),                                                                           
              default_prompt="Transfer cube",                                                                                                                  
              use_delta_joint_actions=False,  # 仿真中不使用 delta action                                                                                      
          ),                                                                                                                                                   
          policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},                                                                                             
      ),配置，这里面没有指定pi0模型的action_horizon，那么默认就是50，为何仍然可以使用？还是模型照样推理50步，但是客户端只要10步？  
      
      
完美，现在全部链条都清楚了。让我来给你一个完整的解释。

## 🎯 答案：两端的 `action_horizon` 完全解耦，各管各的

看数据流你就明白了：

### 一次完整的推理链路

```
客户端 (main.py)                  服务端 (serve_policy.py)
─────────────────                 ─────────────────────
                           ┌──►  model.sample_actions()
ActionChunkBroker          │       └── 输出形状 [1, 50, 14]
  ├── 缓存服务器结果 ◄─────┘         ↑ 模型用的是 Pi0Config(action_horizon=50)
  ├── 第1步: 取 actions[0]
  ├── 第2步: 取 actions[1]         ModelConfig 的 action_horizon=50
  ├── ...
  ├── 第10步: 取 actions[9]
  └── 缓存用完 → 重新推理 ◄───── 服务端输出新的 [1, 50, 14]
```

**关键发现：**

### 1️⃣ 模型永远输出 50 步，客户端只取前 10 步

`ActionChunkBroker` 的代码证实了这一点：

```python
# action_chunk_broker.py
def infer(self, obs):
    if self._last_results is None:
        self._last_results = self._policy.infer(obs)  # 服务端返回 [50, 14]
        self._cur_step = 0

    def slicer(x):
        if isinstance(x, np.ndarray):
            return x[self._cur_step, ...]  # 只取当前步！不管 chunk 有多大
        return x

    results = tree.map_structure(slicer, self._last_results)
    self._cur_step += 1

    if self._cur_step >= self._action_horizon:  # 这里用客户端自己的 10
        self._last_results = None  # 清缓存，下步重新推理
```

**它不关心服务器返回了多少步，只取自己需要的步数。**

### 2️⃣ 所以两端可以不一样，但有代价

|              | 客户端 `action_horizon` | 服务端模型 `action_horizon` |
| ------------ | ----------------------- | --------------------------- |
| 你的配置     | **10**                  | **50**（Pi0Config 默认值）  |
| 作用         | 控制多久重新推理一次    | 模型输出的动作序列长度      |
| 能否自由设置 | ✅ 可以任意             | 由训练时决定，推理时固定    |

但因为服务端永远输出 50 步而客户端只用 10 步，意味着 **每次推理有 40/50 = 80% 的计算量被浪费了**：

```
推理一次输出 50 步：
  ████████████████████████████████████████████████
  └──── 客户端只用前 10 步 ────┘└── 丢弃 40 步 ──┘
```

### 3️⃣ 如果客户端设成 50（匹配模型），会怎样？

```
推理一次输出 50 步，取 50 步
  → 50 步后才重新推理
  → 推理频率降低 5 倍
  → 机器人"反应更慢"，但计算量减为 1/5
```

如果设成比 50 更大的（例如 60）：
```
推理一次输出 50 步
  → 取 50 步，缓存清空
  → 再推理一次，输出 50 步，取前 10 步凑满 60
  → 效果相当于：每 60 步中，50 步来自第一次推理，10 步来自第二次
  → 怪异！不推荐
```

### 4️⃣ 实际影响

你的观察 "推理成功了，拿起了 cube" 说明 **action_horizon=10 配 action_horizon=50 也能工作**，因为：

- 更频繁地重新推理（每 10 步 vs 每 50 步）→ 对观测变化更敏感
- 实际上更像一个 **Receding Horizon（滚动时域）** 策略
- 代价就是服务器多算了 4/5 的没用动作

### 推荐做法

理想情况是把两端设成一致（比如都设 10），避免浪费：

**服务端启动时明确指定 action_horizon=10：**
```python
# 在你的 config.py 中
TrainConfig(
    name="my_pi0_aloha_sim",
    model=pi0_config.Pi0Config(action_horizon=10),  # ← 明确指定！
    ...
)
```

**客户端也设成同样的值：**
```bash
uv run examples/aloha_sim/main.py --args.action_horizon 10 --args.host ...
```

**总结：并不是"模型照样推理 50 步客户端只要 10 步"——而是完全正确！两端 `action_horizon` 是独立参数，互不约束。**


"""

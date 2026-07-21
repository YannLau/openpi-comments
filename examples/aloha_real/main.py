"""
ALOHA 真机策略推理入口

本脚本是 openpi 框架中"真实 ALOHA 机器人策略推理"的启动脚本。
它演示了如何将 openpi 的运行时框架与真实机器人硬件连接起来：

                      ┌──────────────────────────────────────────┐
                      │              本脚本 (main.py)              │
                      │                                          │
                      │  1. 创建 WebSocket 客户端连接到策略服务器   │
                      │  2. 包装为"动作块分发器"实现缓存推理        │
                      │  3. 包装为"智能体"适配运行时接口            │
                      │  4. 创建"真实机器人环境"                    │
                      │  5. 将所有组件注入"运行时"并启动            │
                      └──────────────────────────────────────────┘

工作流程（调用链）：
  Runtime.run()
    ├─ AlohaRealEnvironment.reset()       ← 机器人回到初始位姿
    ├─ loop (每 20ms 执行一次):
    │   ├─ environment.get_observation()  ← 读取摄像头 + 关节编码器
    │   ├─ agent.get_action(obs)          ← 策略推理（见下方调用链）
    │   ├─ environment.apply_action()     ← 发送指令给机器人
    │   └─ 检查结束条件
    └─ environment.reset()                ← 最终复位（安全归位）

策略推理调用链（agent.get_action 内部）：
  PolicyAgent.get_action(obs)
    └─ ActionChunkBroker.infer(obs)       ← 检查缓存
        ├─ 缓存有剩余 → 从缓存取一步动作（不触发网络请求）
        └─ 缓存用完   → WebsocketClientPolicy.infer(obs)
                         └─ [WebSocket] → 远程策略服务器
                           ← 返回 H 步动作块 → 缓存起来逐帧分发

运行方式：
  uv run python examples/aloha_real/main.py \
      --host 127.0.0.1 \
      --port 8000 \
      --action_horizon 25 \
      --num_episodes 1 \
      --max_episode_steps 1000

前置条件：
  1. 远程策略服务器已启动（serve_policy.py）
  2. ALOHA 真机硬件已连接并上电
  3. interbotix 驱动库已安装（ROS 环境） !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ROS!!!!!!!!!
"""

import dataclasses  # 数据类装饰器——用来定义简洁的"配置结构体"
import logging  # 日志系统——输出运行状态信息

# ── openpi-client 运行时框架组件 ──
from openpi_client import action_chunk_broker  # 动作块分发器（缓存+逐步分发）
from openpi_client import (
    websocket_client_policy as _websocket_client_policy,
)  # WebSocket 远程策略客户端
from openpi_client.runtime import (
    runtime as _runtime,
)  # 运行时主循环（观测→推理→执行→记录）
from openpi_client.runtime.agents import (
    policy_agent as _policy_agent,
)  # 策略智能体（将策略包装为 Agent 接口）
import tyro  # 命令行参数解析库（基于类型注解自动生成 CLI）

# ── 真机环境 ──
from examples.aloha_real import env as _env  # ALOHA 真实机器人环境


@dataclasses.dataclass
class Args:
    """命令行参数 —— 通过 tyro 自动解析，支持 `--key value` 方式传入。

    所有参数都有默认值，直接运行脚本将使用这些默认值。
    可以在命令行覆盖：`python main.py --port 8001 --num_episodes 3`
    """

    # ── 远程策略服务器连接参数 ──
    host: str = "0.0.0.0"
    """策略服务器的主机地址。

    默认值 "0.0.0.0" 表示连接本地策略服务器。
    如果策略服务器运行在另一台机器上，改成其 IP 地址：
      - 局域网: "192.168.1.100"
      - 同一台机器: "127.0.0.1" 或 "0.0.0.0"
    """

    port: int = 8000
    """策略服务器的 WebSocket 端口号。

    必须与 serve_policy.py 启动时使用的端口一致。
    默认端口 8000 是 openpi 策略服务器的惯例端口。
    """

    # ── 策略推理参数 ──
    action_horizon: int = 25
    """动作块大小（Action Horizon）。

    策略模型每次推理会预测未来 H 步的动作序列（动作块），
    ActionChunkBroker 会缓存这个块，然后每步取出一个动作执行。

    示例（action_horizon=25，控制频率 50Hz）：
      - 每 25 步推理一次 → 每隔 500ms 推理一次（25 步 × 20ms/步）
      - 推理间隔期间不访问远程服务器，延迟更低

    理解：
      - 值越大：推理频率越低，计算负担越小，但对观测变化的响应越慢
      - 值越小：响应更及时，但推理频率更高
      - 典型值范围：10~50（对应 200ms~1000ms 的预测窗口）
    """

    # ── 运行控制参数 ──
    num_episodes: int = 1
    """要运行的 episode（回合）数量。

    每个 episode 代表一次完整的"从初始状态到任务完成"的过程：
      - Episode 开始：机器人回到起始位姿
      - Episode 中：策略控制机器人执行任务
      - Episode 结束：停止控制，进入下一 episode 或退出

    设置为 >1 可以持续运行多个回合（例如连续抓取 10 个物体）。
    """

    max_episode_steps: int = 1000
    """每个 episode 的最大步数上限（安全保护）。

    真实机器人不能无限运行。这个参数防止以下情况：
      - 策略陷入死循环（一直输出非终结动作）
      - 控制信号丢失（服务器断开后机器人保持上一个位置）
      - 任务失败但没有被检测到（物体掉落，策略还在试图抓取）

    计算：max_episode_steps × 每步时间 ≈ 最大运行时间
      示例：1000 步 × 20ms = 20 秒
      时间到后，无论任务是否完成，episode 都会结束。

    安全建议：
      - 先设小一点（如 300）测试
      - 根据任务平均耗时，留出 2~3 倍余量
    """


def main(args: Args) -> None:
    """主函数 —— 连接所有组件，启动运行时。

    这个函数是"装配工厂"——它创建并连接以下组件：

    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
    │  WebSocket   │    │ ActionChunk  │    │  PolicyAgent │
    │  Client      │◄───│  Broker      │◄───│  (Adapter)   │
    │  (远程推理)   │    │  (缓存分发)   │    │              │
    └──────────────┘    └──────────────┘    └──────┬───────┘
                                                   │
    ┌──────────────┐    ┌──────────────┐           │
    │  ALOHA 真机   │    │  Runtime     │◄──────────┘
    │  环境         │◄───│  (主循环)    │
    └──────────────┘    └──────────────┘

    理解这个"层层包装"的架构：
      每一层只解决一个问题，组合起来完成复杂功能：
      - 最内层（WebSocketClientPolicy）：只知道如何通过网络收发数据
      - 中间层（ActionChunkBroker）：只知道如何缓存和分发
      - 上层（PolicyAgent）：只知道如何适配 Runtime 接口
      - 编排者（Runtime）：只知道按顺序调用各个组件
      - 环境（AlohaRealEnvironment）：只知道如何与硬件交互

    这就是"关注点分离"（Separation of Concerns）的设计原则。
    """

    # ================================================================
    # Step 1: 创建 WebSocket 策略客户端
    # ================================================================
    # WebsocketClientPolicy 是 BasePolicy 的一个实现。
    # 它不执行任何模型推理，而是通过 WebSocket 将观测数据发送到
    # 远程推理服务器，然后接收服务器返回的动作结果。
    #
    # 为什么用远程推理而不是本地推理？
    #   1. GPU 通常在服务器上，真机工控机可能没有 GPU
    #   2. 策略服务器可以服务于多个机器人
    #   3. 模型更新只需要重启服务器，不需要接触机器人
    #
    # 等价的本地方案（有 GPU 的工控机）：
    #   from openpi.policies import policy_config
    #   local_policy = policy_config.create_trained_policy(...)
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )

    # 获取服务器的元数据（包含策略运行所需的信息）
    # 例如 reset_pose（机器人的初始关节位姿）等
    logging.info(f"Server metadata: {ws_client_policy.get_server_metadata()}")

    # ================================================================
    # Step 2: 构建运行时并启动
    # ================================================================
    metadata = ws_client_policy.get_server_metadata()

    # Runtime 是"总控制器"——它协调环境、智能体和订阅者之间的交互。
    # 这是典型的"控制反转"（IoC）设计：Runtime 不知道具体的机器人或策略，
    # 它只是按照固定的循环调用传入的组件。
    runtime = _runtime.Runtime(
        # ── 环境（感知 + 执行） ──
        # AlohaRealEnvironment 是对 ALOHA 真机的抽象封装：
        #   - get_observation(): 读取 4 个摄像头图像 + 14 个关节角度/速度
        #   - apply_action():    发送 14 维动作向量到左右臂电机
        #   - reset():           机器人回到初始位姿
        #   - is_episode_complete(): 真机总是返回 False（由 max_steps 控制结束）
        #
        # reset_position 从服务器元数据获取，确保与训练时使用的初始位姿一致。
        environment=_env.AlohaRealEnvironment(
            reset_position=metadata.get("reset_pose")
        ),
        # ── 智能体（决策） ──
        # PolicyAgent 将 BasePolicy 适配为 Runtime 所需的 Agent 接口。
        # 它内部包装了 ActionChunkBroker → WebsocketClientPolicy 的调用链：
        #
        # Runtime._step()
        #   → PolicyAgent.get_action(obs)
        #     → ActionChunkBroker.infer(obs)
        #       → 缓存命中？取缓存 → 返回
        #       → 缓存未命中？WebsocketClientPolicy.infer(obs)
        #         → [WebSocket 请求] → 远程服务器推理 → 返回 H 步动作块
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=ws_client_policy,
                action_horizon=args.action_horizon,
            )
        ),
        # ── 订阅者（记录/监控） ──
        # 订阅者可以观察每一步的观测和动作，用于：
        #   - 录制视频（记录训练数据）
        #   - 实时显示状态（仪表盘）
        #   - 异常检测（监控关节力矩、温度等）
        # 当前没有订阅者（空列表）。
        # 如果启用了视频录制，可以添加：
        #   subscribers=[VideoSaver(output_dir="./recordings")]
        subscribers=[],
        # ── 控制频率 ──
        # 50Hz = 每步 20ms，这是 ALOHA 标准的控制频率。
        # 包括：传感器读取 + 网络传输 + 远程推理 + 动作执行
        # 如果某一步超时（>20ms），不会追赶，直接等待下一周期。
        max_hz=50,
        # ── 运行控制 ──
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )

    # 启动主循环！这是阻塞调用，会一直运行直到所有 episode 完成。
    # 运行结束后，机器人会自动复位到初始位姿（安全归位）。
    #
    # 主循环内部流程（每步）：
    #   1. get_observation()  ← 读取所有传感器
    #   2. agent.get_action() ← 策略推理（可能触发远程调用）
    #   3. apply_action()     ← 给机器人发送动作指令
    #   4. on_step()          ← 通知订阅者（本脚本中为空）
    #   5. 检查结束条件       ← done?（超时或环境报告完成）
    #
    # 如果想在后台运行（不阻塞当前线程）：
    #   runtime.run_in_new_thread()
    runtime.run()


if __name__ == "__main__":
    # ── 配置日志输出 ──
    # logging.basicConfig 必须在任何 logging 调用之前设置
    # INFO 级别会显示：一般运行信息（不包含 DEBUG 调试信息）
    # force=True 确保覆盖可能已经配置过的日志设置
    logging.basicConfig(level=logging.INFO, force=True)

    # ── 启动入口 ──
    # tyro.cli() 会自动解析命令行参数并调用 main() 函数
    # 参数解析基于 Args 数据类的类型注解：
    #   python main.py --host 127.0.0.1 --port 8000
    #   等价于 Args(host="127.0.0.1", port=8000)
    #
    # tyro 的优势（相比于 argparse）：
    #   - 自动从类型注解生成 --help 文档
    #   - 支持嵌套数据类（复杂参数结构）
    #   - 支持 List、Dict、枚举、联合类型等复杂类型
    tyro.cli(main)


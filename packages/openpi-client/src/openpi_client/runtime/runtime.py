"""
运行时（Runtime）主循环模块 ── 机器人策略推理的"心脏"

本模块定义了 Runtime 类，它是整个 openpi 运行时框架的"总调度器"。
Runtime 负责协调三个核心组件之间的交互：

    ┌──────────────────────────────────────────────────────────┐
    │                        Runtime                            │
    │  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐  │
    │  │  Environment  │  │    Agent     │  │  Subscribers   │  │
    │  │  (环境/机器人) │◄─┤  (策略智能体) │  │  (订阅者/观察者) │  │
    │  └──────┬───────┘  └──────────────┘  └────────────────┘  │
    │         │                                                  │
    │         └─────── 主循环: 观测 → 推理 → 执行 ─────────    │
    └──────────────────────────────────────────────────────────┘

主循环逻辑（每一步）：
  1. get_observation()    ← 从环境读取传感器数据（图像、关节角度等）
  2. agent.get_action()   ← 调用策略模型推理，生成动作
  3. apply_action()       ← 在环境/机器人上执行动作
  4. on_step()            ← 通知所有订阅者（如视频录制器）
  5. is_episode_complete() ← 检查 episode 是否结束

设计理念：
  - 关注点分离：Environment 只管"感知-执行"，Agent 只管"决策"，
    Subscriber 只管"记录"，Runtime 只管"编排"
  - 可扩展性：通过 Subscriber 模式，可以任意添加日志、可视化、监控等功能
  - 线程安全：提供 run_in_new_thread() 用于并发运行
"""

import logging  # 日志系统，用于输出运行时状态
import threading  # 线程库，支持在后台线程中运行
import time  # 时间库，用于帧率控制和性能测量

from openpi_client.runtime import agent as _agent  # 智能体接口（决策者）
from openpi_client.runtime import environment as _environment  # 环境接口（感知-执行者）
from openpi_client.runtime import subscriber as _subscriber  # 订阅者接口（观察者）


class Runtime:
    """运行时主循环 —— 机器人策略推理的核心调度器。

    这个类是"策略-环境"交互循环的编排者。它不知道具体的机器人是什么、
    策略模型是什么、视频怎么保存 —— 它只负责"按正确的顺序调用正确的接口"。

    工作方式：
      ┌─────────────────────────────────────────────────────┐
      │  run()                                               │
      │    ┌─ 循环 episode ─────────────────────┐            │
      │    │  for _ in range(num_episodes):      │            │
      │    │    _run_episode()                   │            │
      │    │      ├─ reset()   ← 环境、智能体、订阅者就位    │
      │    │      └─ while _in_episode:          │            │
      │    │           _step()                   │            │
      │    │             ├─ get_observation()    │            │
      │    │             ├─ agent.get_action()   │            │
      │    │             ├─ apply_action()       │            │
      │    │             ├─ subscriber.on_step() │            │
      │    │             └─ 检查结束条件          │            │
      │    └─────────────────────────────────────┘            │
      │    reset()  ← 最后复位，确保机器人回到安全位置        │
      └─────────────────────────────────────────────────────┘

    典型使用场景（来自 main.py）：
      runtime = Runtime(
          environment=AlohaSimEnvironment(...),  # MuJoCo 仿真环境
          agent=PolicyAgent(...),                 # 策略推理智能体
          subscribers=[VideoSaver(...)],          # 视频录制器
          max_hz=50,                              # 限制 50 Hz
      )
      runtime.run()  # 阻塞，直到所有 episode 完成
    """

    def __init__(
        self,
        environment: _environment.Environment,
        agent: _agent.Agent,
        subscribers: list[_subscriber.Subscriber],
        max_hz: float = 0,
        num_episodes: int = 1,
        max_episode_steps: int = 0,
    ) -> None:
        """初始化运行时。

        这是将所有组件"插接"到一起的地方。Runtime 本身不创建任何组件，
        它只是接收已有的组件实例并按正确顺序调用它们。

        Args:
            environment:     环境实例（仿真或真实机器人）。
                              必须实现 Environment 接口（reset, get_observation,
                              apply_action, is_episode_complete）。
            agent:           智能体实例（策略模型）。
                              必须实现 Agent 接口（get_action, reset）。
            subscribers:     订阅者列表（观察者/记录器）。
                              每个订阅者必须实现 Subscriber 接口
                              （on_episode_start, on_step, on_episode_end）。
            max_hz:          最大运行频率（Hz）。
                              - 50: 每秒最多 50 步（20ms 一步）
                              - 0:  不限速，尽可能快地运行（适用于调试或离线处理）
                              频率限制通过 time.sleep() 实现，确保真实机器人
                              或仿真的控制频率不会过快。
            num_episodes:    要运行的 episode 数量。默认为 1。
                              每个 episode 从 environment.reset() 开始，
                              到 is_episode_complete() 返回 True 结束。
            max_episode_steps: 每个 episode 的最大步数上限（防止无限循环）。
                               - 0: 不限制，完全由环境决定何时结束
                               - >0: 超过此步数强制结束 episode
                               安全措施：防止机器人永远运行下去。
        """
        # ---- 三大核心组件 ----
        self._environment = environment  # "世界"——感知传感器状态，执行动作
        self._agent = agent  # "大脑"——接收观测，输出动作
        self._subscribers = subscribers  # "观众"——记录/监控每一步

        # ---- 控制参数 ----
        self._max_hz = max_hz  # 频率上限（Hz），0 = 不限速
        self._num_episodes = num_episodes  # 运行多少个 episode
        self._max_episode_steps = max_episode_steps  # 单 episode 最大步数

        # ---- 运行状态 ----
        self._in_episode = False  # 是否正在 episode 中
        self._episode_steps = 0  # 当前 episode 已执行的步数

    def run(self) -> None:
        """启动主循环，运行所有 episode。

        这个方法会阻塞直到所有 episode 完成。这是最常用的入口，
        适用于"一次运行，从头到尾"的场景。

        运行流程：
          1. 按顺序执行每个 episode
          2. 所有 episode 完成后，做最后一次环境复位
             （重要！对于真实机器人，这个复位能将机器人移回安全起始位姿，
              避免在切换程序或关闭电源时发生意外碰撞）

        注意：
          - 这是同步阻塞调用。如果想在后台运行，使用 run_in_new_thread()。
          - 如果 environment.reset() 在真实机器人上执行了"回 home"动作，
            这个最后的 reset() 会在所有 episode 完成后让机器人归位。
        """
        # 顺序运行每个 episode
        for _ in range(self._num_episodes):
            self._run_episode()

        # 最终复位 —— 把所有组件恢复到初始状态。
        # 对于真实机器人特别重要：将机器人移回 home 位置，
        # 防止在程序结束后机器人悬在半空中或处于危险位姿。
        self._environment.reset()

    def run_in_new_thread(self) -> threading.Thread:
        """在新线程中启动主循环。

        适用于以下场景：
          - 你需要主线程做其他事情（如处理用户输入、WebSocket 通信）
          - 你的订阅者中有人需要异步操作（如写入大文件）
          - 你希望主循环不阻塞终端交互

        用法：
          thread = runtime.run_in_new_thread()
          # ... 主线程继续做其他事情 ...
          thread.join()  # 等待主循环完成

        Returns:
            threading.Thread: 运行主循环的后台线程。可以 join() 等待它完成。
        """
        thread = threading.Thread(target=self.run)
        thread.start()  # 启动线程，主循环在后台运行
        return thread

    def mark_episode_complete(self) -> None:
        """标记当前 episode 结束。

        这个方法可以被外部调用者用来提前终止正在运行的 episode。
        比如：
          - 用户按下了"停止"按钮
          - 另一个线程检测到了异常（如机器人过载、传感器断连）
          - 远程控制台发送了"停止"指令

        它只是设置一个标志位，主循环中的 while 条件会在
        下一次循环迭代时检查到 _in_episode=False 从而退出。
        不会立即中断正在执行的 _step()。
        """
        self._in_episode = False

    def _run_episode(self) -> None:
        """运行单个 episode —— 这是主循环的核心。

        一个 episode 代表一次完整的"从初始状态到任务完成"的过程。
        对于机器人抓取任务，就是一个 episode = 从起始位姿到抓到物体。

        episode 的生命周期：
          1. 初始化阶段：重置环境、智能体、通知订阅者
          2. 运行阶段：循环执行 _step()，直到完成
          3. 收尾阶段：通知订阅者 episode 结束

        帧率控制：
          Runtime 通过简单的 time.sleep() 来控制运行频率。
          这是"软实时"控制：如果某一步的计算时间超过了帧间隔，
          不会等待，直接进入下一步（不会试图"追赶"）。
          因此实际帧率 ≤ max_hz。
        """
        logging.info("Starting episode...")

        # ============ 初始化阶段 ============
        # 1. 重置环境：机器人回到起始位姿，任务目标重新初始化
        self._environment.reset()

        # 2. 重置智能体：清空内部状态（如 ActionChunkBroker 的动作缓存）
        self._agent.reset()

        # 3. 通知所有订阅者：新 episode 开始了
        #    比如 VideoSaver 会清空帧缓存
        for subscriber in self._subscribers:
            subscriber.on_episode_start()

        # ============ 运行阶段 ============
        self._in_episode = True
        self._episode_steps = 0

        # 计算每步的目标间隔时间（秒）
        #   50 Hz → 1/50 = 0.02 秒 = 20ms 每步
        #    0 Hz → 不限速，步长为 0
        step_time = 1 / self._max_hz if self._max_hz > 0 else 0

        # 记录上一步的完成时间，用于计算需要 sleep 多久
        last_step_time = time.time()

        # ---- 主循环 ----
        while self._in_episode:
            self._step()  # 执行一步（观测→推理→执行→记录）
            self._episode_steps += 1

            # ---- 帧率控制 ----
            # 计算上一步花费了多少时间
            now = time.time()
            dt = now - last_step_time

            if dt < step_time:
                # 如果这一步比目标间隔快，就 sleep 剩余时间
                # 例如：目标 20ms，这一步花了 12ms → sleep 8ms
                time.sleep(step_time - dt)
                last_step_time = time.time()
            else:
                # 如果这一步比目标间隔慢（计算超时），不 sleep，直接继续
                # 注意：这里不会补偿，上次的"延迟"不会影响下次的计时起点
                last_step_time = now

        # ============ 收尾阶段 ============
        logging.info("Episode completed.")
        for subscriber in self._subscribers:
            subscriber.on_episode_end()  # 通知订阅者保存数据、关闭文件等

    def _step(self) -> None:
        """单步交互 —— 策略推理循环的最小单元。

        这是整个系统最关键的"心跳"方法，执行标准的机器人控制循环：

        ┌─────────────────────────────────────────────────────────┐
        │  感知 (Sense)                                           │
        │    └─ environment.get_observation()                     │
        │       └─ 读取摄像头图像、关节编码器、力传感器等           │
        │                                                         │
        │  决策 (Plan)                                            │
        │    └─ agent.get_action(observation)                     │
        │       └─ 策略模型前向传播 → 输出关节目标位置/速度        │
        │                                                         │
        │  执行 (Act)                                             │
        │    └─ environment.apply_action(action)                  │
        │       └─ 发送指令到机器人 / 更新 MuJoCo 物理状态        │
        │                                                         │
        │  记录 (Record)                                          │
        │    └─ subscribers[i].on_step(obs, action)               │
        │       └─ 保存视频帧、记录日志、更新仪表盘等              │
        │                                                         │
        │  检查 (Check)                                           │
        │    └─ environment.is_episode_complete()?                │
        │       └─ 任务成功？机器人摔倒？超时？                    │
        └─────────────────────────────────────────────────────────┘

        设计决策：
          - 为什么 get_observation() 和 apply_action() 分开？
            因为环境和模型之间可能有网络延迟（WebSocket 策略服务器），
            分开可以更清晰地追踪每个阶段的耗时。
          - 为什么订阅者在 apply_action() 之后才被调用？
            因为订阅者通常需要"这一步执行完成后的完整状态"来做记录。
        """
        # ====== 1. 感知 ======
        # 从环境获取当前观测（图像、关节状态等）
        observation = self._environment.get_observation()

        # ====== 2. 决策 ======
        # 智能体根据观测推理出要执行的动作
        action = self._agent.get_action(observation)

        # ====== 3. 执行 ======
        # 将动作发送到环境/机器人，推进一个时间步
        self._environment.apply_action(action)

        # ====== 4. 记录 ======
        # 通知所有订阅者（视频录制器、日志等）
        for subscriber in self._subscribers:
            subscriber.on_step(observation, action)

        # ====== 5. 检查结束条件 ======
        # 检查环境是否报告 episode 已完成（任务成功/失败）
        # 或者是否达到了最大步数限制
        if self._environment.is_episode_complete() or (
            self._max_episode_steps > 0 and self._episode_steps >= self._max_episode_steps
        ):
            # 满足任一条件 → 标记当前 episode 结束
            # 主循环中的 while self._in_episode 将在下一次迭代时退出
            self.mark_episode_complete()

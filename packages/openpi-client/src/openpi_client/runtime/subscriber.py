"""
订阅者（Subscriber）接口抽象定义模块

本模块定义了 Subscriber 抽象基类，它是 Runtime 框架中"观察者"组件的标准接口。

什么是"订阅者"（Subscriber）？
  订阅者是一种"旁观者"——它不参与决策，也不影响环境的运行，
  但它能在每个关键时间点"看到"正在发生的事情并做出响应。

  最常见的用途：
    - 视频录制器：每一步保存一帧图像，episode 结束时合成视频
    - 数据记录器：将每一步的观测和动作写入磁盘，用于后续分析
    - 性能监控器：统计每一步的耗时、检测异常
    - 仪表盘推送：将实时数据发送到前端可视化界面

设计模式：观察者模式（Observer Pattern）
  Runtime（主题）               Subscriber（观察者）
  ┌────────────────┐            ┌──────────────────────┐
  │  主循环         │            │                      │
  │  on_episode_start() ──────→ │  清空帧缓存           │
  │                 │            │                      │
  │  on_step()     ──────→     │  保存当前帧到列表      │
  │                 │            │                      │
  │  on_episode_end()  ──────→ │  合成 MP4 视频文件    │
  └────────────────┘            └──────────────────────┘

  这种模式的优点：
    1. 解耦：Runtime 不需要知道订阅者在做什么，只管通知
    2. 可组合：可以同时添加多个订阅者，互不干扰
    3. 可插拔：可以在运行时动态添加或移除订阅者

三个回调方法的执行时机（在 Runtime 的主循环中）：
  ┌─────────────────────────────────────────────────────────┐
  │  _run_episode()                                          │
  │    ├─ environment.reset()                                │
  │    ├─ agent.reset()                                      │
  │    ├─ on_episode_start()  ←── 所有订阅者收到"新 episode"  │
  │    │                                                      │
  │    ├─ while _in_episode:                                 │
  │    │    ├─ _step()                                       │
  │    │    │   ├─ get_observation()                         │
  │    │    │   ├─ get_action()                              │
  │    │    │   ├─ apply_action()                            │
  │    │    │   └─ on_step()  ←── 所有订阅者收到"新的一步"    │
  │    │    └─ 检查结束条件                                    │
  │    │                                                      │
  │    └─ on_episode_end()  ←── 所有订阅者收到"episode 结束"  │
  └─────────────────────────────────────────────────────────┘

参考实现：
  - VideoSaver（examples/aloha_sim/saver.py）— 将仿真过程录制成视频
"""

import abc  # Python 抽象基类库，用于定义接口契约


class Subscriber(abc.ABC):
    """订阅者抽象基类 —— 监听 Runtime 生命周期事件的"观察者"。

    订阅者可以在三个时间点介入：
      - episode 开始时（on_episode_start）：做准备工作
      - 每一步后（on_step）：记录数据
      - episode 结束时（on_episode_end）：做收尾工作

    生命周期：
      for each episode:
          on_episode_start()     ← 初始化（如清空帧缓存）
          for each step:
              on_step(obs, act)  ← 记录每一步（如保存一帧图像）
          on_episode_end()       ← 收尾（如合成视频文件）

    实现注意事项：
      - 所有回调都是同步的——Runtime 会等待每个订阅者完成后才继续。
        因此不要在回调中做耗时操作，否则会拖慢整个主循环。
        如果确实需要耗时操作（如写入大文件），应该使用异步/线程。
      - 回调接收的是观测和动作的引用（不是副本），
        如果要在回调之后继续使用这些数据，记得做深拷贝。
      - on_step 的 observation 和 action 是"刚刚执行完这一步后"的状态，
        不是"执行这一步之前"的状态。
    """

    @abc.abstractmethod
    def on_episode_start(self) -> None:
        """Episode 开始时的回调。

        调用时机：Runtime 的 _run_episode() 中，在 environment.reset()
        和 agent.reset() 之后，进入主循环之前被调用。

        此时：
          - 环境已经重置到初始状态（机器人回到起始位姿，任务重新初始化）
          - 智能体已经重置（动作缓存被清空）
          - 订阅者应该在这里做"新 episode"的准备工作

        典型实现：
          - VideoSaver：清空上一轮的帧缓存列表（self._images = []）
          - 数据记录器：创建新的日志文件
          - 性能监控器：重置计时器计数器
        """

    @abc.abstractmethod
    def on_step(self, observation: dict, action: dict) -> None:
        """每一步完成后的回调。

        调用时机：Runtime 的 _step() 中，在 environment.apply_action()
        执行完毕之后被调用。

        Args:
            observation: 执行动作后的环境观测。包含：
                         - images: 摄像头图像（已转换为模型输入格式）
                         - state:  关节角度、夹爪位置等状态信息
            action:      智能体本轮推理出的动作。包含：
                         - actions: 关节目标位置/速度指令

        典型实现：
          - VideoSaver：从 observation["images"] 中提取图像，缓存到列表
          - 数据记录器：将 (obs, action) 对写入磁盘
          - 仪表盘推送：将关键指标发送到 WebSocket

        注意：
          - observation 是"执行动作后的状态"——即动作已经生效了。
            如果需要"执行动作前"的状态，需要在 get_action() 之前自己记录。
          - 不要在回调中修改 observation 或 action（只读访问）。
          - 如果订阅者之间有依赖关系，需要自行协调（Runtime 不保证顺序）。
        """

    @abc.abstractmethod
    def on_episode_end(self) -> None:
        """Episode 结束时的回调。

        调用时机：Runtime 的 _run_episode() 中，在主循环退出之后、
        下一个 episode 开始之前被调用。

        此时：
          - 所有 step 已经执行完毕
          - 下一个 episode 还没有开始
          - 订阅者应该在这里做"收尾工作"——保存数据、关闭文件、释放资源

        典型实现：
          - VideoSaver：将所有缓存的帧合成为 MP4 视频文件写到磁盘
          - 数据记录器：关闭当前日志文件，写入元数据
          - 性能监控器：打印本 episode 的平均步耗时、成功率等统计信息

        注意：
          - 如果有多个 episode 连续运行，on_episode_end() 之后
            紧接着就是下一个 episode 的 on_episode_start()。
            所以一定要分清"episode 级别"的资源和"全局"资源。
            例如 VideoSaver 在 on_episode_start() 中清空帧缓存，
            在 on_episode_end() 中将帧写入文件——这样每个 episode
            产生一个独立的视频文件。
        """

"""
π₀ 模型部署脚本 —— 连接 Tron2 机器人并执行推理控制循环。

【整体工作流水线】
    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
    │  Tron2 机器人  │ ──→│  获取观测数据  │ ──→│  WebSocket   │ ──→│  执行动作    │
    │  (10.192.1.2) │ ←──│  get_obs()   │ ←──│  策略推理    │ ←──│  step()     │
    └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                             │                     │
                             ▼                     ▼
                      记录到 record_state    保存动作到 record_action
                      并保存在 CSV 文件中

Usage:
    python examples/tron2/pi_client.py

注意：
    1. 目前控制顺序和反馈顺序是：左臂 → 左手 → 右臂 → 右手
       （即动作向量的排列顺序为：[左臂7, 左夹爪1, 右臂7, 右夹爪1]）
    2. 默认数据是正常的，不需要乘以 bias（偏移校正）
"""

import numpy as np
import time
import einops
from pathlib import Path
from PIL import Image

from openpi_client import websocket_client_policy, image_tools
from real_env import Tron2Env, EnvConfig
from robot_utils import Tron2Config


if __name__ == "__main__":
    # ================================================================
    # 第 1 步：机器人配置
    # ================================================================

    # 初始关节位置（"挂衣服"任务的预设姿态）
    # 结构：[左臂7关节, 左夹爪1, 右臂7关节, 右夹爪1] = 14 维
    # 这是机器人在控制循环开始前需要移动到的初始姿态，
    # 目的是让机器人从已知的安全位置开始执行任务。
    init_joints_clothes = [
        0.026899,   0.2612,    -0.02709991, -1.5477003,    # 左臂关节 1-4
        0.265,      0.0180999, -0.0614999,                 # 左臂关节 5-7
        # （索引 7 是左夹爪，这里未显示 —— 在完整 14 维中下标 7 是左夹爪）
        0.008999,  -0.269,     0.02069998, -1.5567001,    # 右臂关节 1-4
        -0.254,    -0.02309972, 0.06469989                # 右臂关节 5-7
        # （索引 13 是右夹爪，这里未显示）
    ]

    # 备选初始关节位置（注释掉了，可能是其他任务的初始姿态）
    # init_joints_clothes = [
    #     -0.63819385,  0.83982128, -1.03469932, -1.24587011,
    #      0.82801813, -0.23849821, -0.71935195,
    #      0.008999,   -0.269,       0.02069998, -1.5567001,
    #     -0.254,      -0.02309972,  0.06469989
    # ]

    # 初始头部姿态（2 维：[俯仰, 偏航] 角度）
    init_head = [1.0467, -0.0139998]

    # Tron2Config: 机器人硬件配置
    # - robot_ip: Tron2 机器人控制器的 IP 地址
    # - init_joints: 控制循环开始前机器人需要移动到的初始关节位置
    # - init_head: 控制循环开始前头部（摄像头）的初始姿态
    robot_config = Tron2Config(
        robot_ip="10.192.1.2",
        init_joints=init_joints_clothes,
        init_head=init_head
    )

    # ================================================================
    # 第 2 步：环境配置
    # ================================================================

    # 摄像头序列号到逻辑名称的映射（注释掉了，暂未启用）
    # serial_to_name = {
    #     'serial_to_name': {
    #         "245022302696": 'head_camera_image',     # 头部摄像头
    #         "409122274385": 'left_wrist_image',      # 左腕部摄像头
    #         "230322276915": 'right_wrist_image'      # 右腕部摄像头
    #     }
    # }

    # EnvConfig: 机器人运行环境配置
    # - robot_config: 上面定义的机器人硬件配置
    # - interp_points: 插值点数 —— 控制策略输出的动作在发送给机器人之前
    #   被插值细分为多少个中间点。值越大，动作执行越平滑。
    #   【理解】策略通常以较低的频率输出动作（如 10Hz），
    #   但机器人需要更高频率的控制信号（如 50Hz）。
    #   interp_points 就是用来在两次策略输出之间进行线性插值的点数。
    # - time_sync_tolerance: 时间同步容差（秒）。用于对齐不同传感器
    #   （如摄像头和关节编码器）的时间戳，确保观测数据的时间一致性。
    env_config = EnvConfig(
        robot_config=robot_config,
        interp_points=8,
        time_sync_tolerance=0.01,
        # raw_config = {'camera': serial_to_name}
    )

    # ================================================================
    # 第 3 步：主控制循环
    # ================================================================

    # Tron2Env 实现了 Python 上下文管理器协议（with 语句），
    # 在进入时连接机器人并移动到初始位置，退出时断开连接。
    with Tron2Env(env_config) as env:
        # 重置环境 —— 将机器人移动到 init_joints 指定的初始位置
        env.reset()

        # 创建 WebSocket 策略客户端
        # WebsocketClientPolicy 负责与运行策略推理的服务器通信。
        # 策略服务器（serve_policy.py）加载训练好的 π₀ 模型，
        # 通过 WebSocket 提供推理服务。
        # - host='0.0.0.0': 连接到本地所有网络接口
        # - port=8000: 策略服务器的 WebSocket 端口
        ws_client_policy = websocket_client_policy.WebsocketClientPolicy(
            host='0.0.0.0',
            port=8000,
        )

        # 控制循环计数器
        t = 0

        # 上一次执行的动作（截取前 14 维，不包括可能的 padding 维）
        # 用于计算动作之间的差异，检测异常跳跃
        last_action = env.last_action[:14]

        # ---- 数据记录缓冲区 ----
        # 记录每个时间步的状态观测值，用于后续分析
        record_state = []
        # 记录每个时间步的策略输出动作，用于后续分析
        record_action = []

        # ---- 主循环：最多执行 100 步 ----
        while t < 100:
            print("\n\n", "#" * 10, "begin infer", "#" * 10)

            # ========================================================
            # 步骤 A：获取机器人观测数据
            # ========================================================
            obs = env.get_obs()

            # 记录当前状态到缓冲区
            record_state.append(obs["state"].copy())

            # --- 可选：保存原始 RGB 图像到本地文件，用于调试 ---
            rgb_images = obs["images"]
            Path("examples/tron2/recorded_rgb").mkdir(parents=True, exist_ok=True)
            [
                Image.fromarray(
                    image_tools.convert_to_uint8(rgb_images[c])  # 确保图像是 uint8 格式
                    if rgb_images[c].dtype != np.uint8
                    else rgb_images[c]
                ).save(f"examples/tron2/recorded_rgb/t_{t:04d}_{c}.png")
                for c in rgb_images
            ]

            print(f"states:{obs['state']}")

            # ========================================================
            # 步骤 B：图像预处理 —— 将原始图像转换为模型输入格式
            # ========================================================
            for cam_name in rgb_images:
                # 1. 调整图像大小并填充到 224x224（π₀ 模型的标准输入尺寸）
                #    - resize_with_pad: 等比例缩放并用黑色填充边缘，
                #      保持原始图像的宽高比，避免图像变形
                img = image_tools.resize_with_pad(
                    obs["images"][cam_name], 224, 224
                )

                # 2. 转换为 uint8（确保像素值范围是 [0, 255]）
                img = image_tools.convert_to_uint8(img)

                # 3. 通道重排：[H, W, C] → [C, H, W]
                #    π₀ 模型期望的图像格式是 [C, H, W]（通道在前），
                #    而机器人摄像头通常输出 [H, W, C]（通道在后）
                obs["images"][cam_name] = einops.rearrange(img, "h w c -> c h w")

            # ========================================================
            # 步骤 C：通过 WebSocket 调用策略推理
            # ========================================================
            ts = time.time()  # 记录推理开始时间

            # infer() 发送观测数据到策略服务器，返回预测的动作计划
            # 输入格式：包含 "state"、"images"、"prompt" 等字段的字典
            # 输出格式：包含 "actions" 的字典
            #   - "actions": 形状为 [action_horizon, action_dim] 的数组
            #     * action_horizon: 策略一次预测的多个未来动作（动作块/chunk）
            #     * action_dim: 动作空间维度
            ans = ws_client_policy.infer(obs)

            te = time.time()  # 记录推理结束时间
            print("infer time:", te - ts)

            # ========================================================
            # 步骤 D：处理预测结果并执行动作
            # ========================================================

            # 从推理结果中提取动作计划并堆叠为数组
            # action_plan: 一个列表，包含 action_horizon 个时间步的动作
            # actions: 形状 [action_horizon, action_dim] 的 numpy 数组
            action_plan = ans['actions']
            actions = np.stack(action_plan, axis=0)

            # 打印第一个动作和最后一个动作，用于调试
            # actions[0][:8]  = 第一个时间步的 左臂7 + 左夹爪1
            # actions[0][8:]  = 第一个时间步的 右臂7 + 右夹爪1
            print("左臂开始:", actions[0][:8])
            print("右臂开始:", actions[0][8:])
            print("左臂结束:", actions[-1][:8])
            print("右臂结束:", actions[-1][8:])

            # 记录推理时间（用于日志）
            infer_time = time.time()
            # 将当前动作计划保存到记录缓冲区
            record_action.append(actions)

            # ---- 逐时间步执行动作块中的每个动作 ----
            for action in actions:
                # 组装完整的 14 维机械臂动作（排除第 7 个元素：左夹爪）
                # 原 action 结构：[左臂7, 左夹爪1, 右臂7, 右夹爪1] = 16 维
                # arm_action 结构：[左臂7, 右臂7] = 14 维（仅关节角，不含夹爪）
                # 注意：action[7] 是左夹爪，这里被跳过了
                arm_action = np.concatenate((action[:7], action[8:15]))

                # ---- 安全检查：检测关节动作的异常跳跃 ----
                # 计算当前动作与上一次动作每个关节的差异
                error = np.abs(arm_action - last_action)
                id = np.argmax(error)          # 找出最大差异的关节索引
                max_diff = error[id]           # 最大差异值

                # 【安全机制】如果某个关节的移动量超过 0.5（弧度），
                # 打印警告信息。这可能是策略输出异常或数值错误的信号。
                # 注意：这里只打印警告，不阻止动作执行。
                # 在实际部署中，可以考虑跳过该动作或停止执行。
                if max_diff >= 0.5:
                    print(f"joint {id} 's error is {max_diff}")

                # 【数据过滤示例（注释掉了）】
                # 如果夹爪动作值小于 0.7，强制设为 0（夹爪闭合）
                # 这种硬编码阈值通常用于任务特定的约束
                # if action[7] < 0.7:
                #     action[7] = 0  # gripper close
                #     print("gripper open")

                # 执行动作：发送到机器人控制器
                env.step(action)

                # 更新上一次动作记录
                last_action = arm_action

                # 可选：控制执行频率，避免过快执行导致机器人动作抖动
                # time.sleep(0.03)

            # 增加循环计数器（当前被注释掉了，所以循环是无限的！）
            # t += 1

        # ================================================================
        # 第 4 步：保存记录数据到 CSV 文件（用于事后分析）
        # ================================================================

        # 将记录的动作列表堆叠为一个大数组
        # record_action 是 list of [action_horizon, 16] 数组，
        # vstack 后形状为 [步数 × action_horizon, 16]
        com_array = np.vstack(record_action)

        # 将记录的状态堆叠为一个大数组
        # record_state 是 list of [14] 数组，
        # vstack 后形状为 [步数, 14]
        com2_array = np.vstack(record_state)

        # 保存到 CSV 文件（格式：逗号分隔，三位小数精度）
        np.savetxt(
            'examples/tron2/clothes_action_data2.csv',
            com_array,
            delimiter=',',
            fmt='%.3f'
        )
        np.savetxt(
            'examples/tron2/clothes_state_data2.csv',
            com2_array,
            delimiter=',',
            fmt='%.3f'
        )

        print("数据已保存到 examples/tron2/clothes_action_data2.csv 和 clothes_state_data2.csv")

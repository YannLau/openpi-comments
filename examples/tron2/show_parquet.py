#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
import sys


class ParquetVisualizer:
    def __init__(self, file_path="/home/limx/openpi/datasets/episode_000000.parquet"):
        """
        初始化Parquet可视化器
        
        Args:
            file_path: Parquet文件路径
        """
        self.file_path = file_path
        self.df = None
        self._load_data()
    
    def _load_data(self):
        """加载Parquet文件数据"""
        if self.df is None:
            try:
                self.df = pd.read_parquet(self.file_path, engine='pyarrow')
                assert 'index','observation.state' in self.df.columns
                self.actions = np.stack(self.df['action'])
                self.states = np.stack(self.df['observation.state'])
                self.left_images = self.df['observation.images.cam_left_wrist']
                self.right_images = self.df['observation.images.cam_right_wrist']
                self.head_images = self.df['observation.images.cam_high']

                self.episode_idx = self.df['episode_index']
                self.frame_idx = self.df['frame_index']
                self.idx = self.df['index'] # 在所有的episode中的idx
                self.timestamp = self.df['timestamp']
                assert self.actions.shape == self.states.shape
            except Exception as e:
                print(f"❌ 读取 Parquet 文件失败: {e}", file=sys.stderr)
                raise


    def get_obs(self, step):
        """
        获取指定step的观测数据
        
        Args:
            step: 时间步索引
            
        Returns:
            obs: 包含images、state、action的字典
        """
        if step >= len(self.states):
            raise IndexError(f"Step {step} 超出范围。最大步数为 {len(self.states)-1}")
        
        # 解析图像数据
        image_dict = {}
        
        # 处理 cam_high
        try:
            image_dict['cam_high'] = np.array(Image.open(io.BytesIO(self.head_images[step]['bytes'])))
            image_dict['cam_left_wrist'] = np.array(Image.open(io.BytesIO(self.left_images[step]['bytes'])))
            image_dict['cam_right_wrist'] = np.array(Image.open(io.BytesIO(self.right_images[step]['bytes'])))
        except Exception as e:
            print(f"❌ 无法解码图像: {e}", file=sys.stderr)

        # 构造观测字典
        obs = {
            'images': image_dict,
            'state': self.states[step],
        }
        
        return obs

    def show_action_state(self, actions = None, states_or_pre_action=None):
        """
        显示所有step的两条曲线
        action vs state
        or
        action vs pred_action
        """
        if actions is None or states_or_pre_action is None:
            dataA = self.actions
            dataB = self.states
        else:
            dataA = actions
            dataB = states_or_pre_action

        # 创建 Figure 1: action[0:7]/state[0:7] + gripper: action[14]
        fig1, axes1 = plt.subplots(int(dataA.shape[1]/2), 2, figsize=(18, 3*dataA.shape[1]/2))
        x = self.df.index
        x_values_timestamp = self.timestamp.to_list()
        for i in range(int(dataA.shape[1]/2)):
            # 绘制 action/state 第i维
            left_y_values_state = dataB[:,i]
            left_y_values_action = dataA[:,i]
            right_y_values_state = dataB[:,i+8]
            right_y_values_action = dataA[:,i+8]

            # if i == 7:
            #     left_y_values_action = dataA[:,14]
            #     left_y_values_state = dataB[:,14]
            #     right_y_values_state = dataB[:,15]
            #     right_y_values_action = dataA[:,15]

            # axes1[i][0].plot(x_values_timestamp, left_y_values_state, label=f'State', color='tab:blue', linewidth=1.5)
            # axes1[i][0].plot(x_values_timestamp, left_y_values_action, label=f'Action', color='tab:orange', linewidth=1.5)
            # axes1[i][1].plot(x_values_timestamp, right_y_values_state, label=f'State', color='tab:blue', linewidth=1.5)
            # axes1[i][1].plot(x_values_timestamp, right_y_values_action, label=f'Action', color='tab:orange', linewidth=1.5)

            axes1[i][0].scatter(x_values_timestamp, left_y_values_state, label='State', color='tab:blue', s=1, alpha=1)
            axes1[i][0].scatter(x_values_timestamp, left_y_values_action, label='Action', color='tab:orange', s=1, alpha=0.5)
            axes1[i][1].scatter(x_values_timestamp, right_y_values_state, label='State', color='tab:blue', s=1, alpha=1)
            axes1[i][1].scatter(x_values_timestamp, right_y_values_action, label='Action', color='tab:orange', s=1, alpha=0.5)

            # axes1[i].set_xlabel('Time Step (Index)')
            axes1[i][0].set_ylabel(f'Joint{i}')
            axes1[i][0].grid(True, linestyle='--', alpha=0.6)
            axes1[i][1].grid(True, linestyle='--', alpha=0.6)
            # axes1[i][1].set_ylabel(f'Joint{i+7}')
        axes1[0][0].set_title(f'Left Arm')
        axes1[0][1].set_title(f'Right Arm')
        axes1[0][0].legend()
        axes1[0][1].legend()

        plt.tight_layout()
        plt.show()

    def show_step_images(self, step, axeses=None):
        """
        显示指定step的图像
        
        Args:
            step: 时间步索引
        """
        obs = self.get_obs(step)
        if axeses is None:
            # 创建三个图像
            fig3, axeses = plt.subplots(1, 3, figsize=(18, 6))
        cam_names = ['cam_left_wrist','cam_high' , 'cam_right_wrist']
        titles = ['Left Wrist', 'High Camera', 'Right Wrist']
        for i, (cam_name, title) in enumerate(zip(cam_names, titles)):

            # 将numpy数组转换为PIL图像用于显示
            img_array = obs['images'][cam_name]
            # img_pil = Image.fromarray(img_array)
            axeses[i].imshow(img_array)
            axeses[i].set_title(f'{title} - Step {step}')
            # 在图下方显示 action[step]
            if cam_name == 'cam_left_wrist':
                axeses[i].set_xlabel(f'Left Gripper action: {self.actions[step][7]:.3f}' )
            if cam_name == 'cam_right_wrist':
                axeses[i].set_xlabel(f'Right Gripper action: {self.actions[step][15]:.3f}' )
            # axeses[i].axis('off')
        if axeses is None:
            plt.tight_layout()
            plt.show()
    
    def show_episode_images(self):
        """
            显示所有步骤的图像，支持键盘控制播放（自动/手动模式）
        """
        frame_index = self.frame_idx.tolist()
        total_frames = len(frame_index)
        if total_frames == 0:
            print("❌ 没有可用的帧数据")
            return

        # 创建图形和轴
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        plt.tight_layout()
        
        # 初始化状态
        current_frame_index = 0
        auto_play = False
        timer = None  # 用于自动播放的定时器
        
        # 显示第一帧
        def display_frame(index):
            nonlocal current_frame_index
            current_frame_index = index
            
            # 清空所有轴
            for ax in axes:
                ax.clear()
                # ax.axis('off')
            
            # 获取并显示图像
            try:
                self.show_step_images(step=index,axeses=axes)
                plt.suptitle(f'Episode {self.episode_idx[index]} | Frame {index}', fontsize=12)
                plt.draw()
            except Exception as e:
                print(f"❌ 显示帧 {index} 失败: {e}")
        
        # 自动播放回调函数
        def auto_play_callback():
            nonlocal current_frame_index
            if current_frame_index > total_frames:
                timer.stop()
            current_frame_index = (current_frame_index + 1) #% total_frames
            display_frame(current_frame_index)
        
        # 键盘事件处理
        def on_key(event):
            nonlocal current_frame_index, auto_play, timer
            if event.key == ' ':
                auto_play = not auto_play
                if auto_play:
                    print("✅ 切换到自动播放模式 (按空格键切回手动模式)")
                    # 启动自动播放定时器
                    if timer is None:
                        timer = fig.canvas.new_timer(interval=50)  # 50ms间隔
                        timer.add_callback(auto_play_callback)
                        timer.start()
                else:
                    print("✅ 切换到手动模式 (按左右键切换帧)")
                    # 停止自动播放
                    if timer:
                        timer.stop()
                        timer = None
            elif not auto_play:
                if event.key == 'left':
                    if current_frame_index > 0:
                        current_frame_index -= 1
                        display_frame(current_frame_index)
                elif event.key == 'right':
                    if current_frame_index < total_frames:
                        current_frame_index += 1
                        display_frame(current_frame_index)
            elif event.key == 'r':
                print("复位")
                current_frame_index = 0
                display_frame(current_frame_index)
            elif event.key == 'q':
                print("退出程序")
                if timer:
                    timer.stop()
                plt.close(fig)
        
        # 连接键盘事件
        fig.canvas.mpl_connect('key_press_event', on_key)

        # 窗口关闭事件处理：停止定时器防止回调错误
        def on_close(event):
            nonlocal timer
            if timer:
                timer.stop()
                timer = None
        fig.canvas.mpl_connect('close_event', on_close)

        # 显示初始帧
        display_frame(0)
        
        # 显示图形
        plt.show()

    

# 示例使用
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parquet数据可视化工具")
    parser.add_argument("--file", type=str, 
                        default="/home/limx/Downloads/lerobot_limx-tron2%2Flerobot_dataset_1770032136920/lerobot_dataset/data/chunk-000/episode_000013.parquet",
                        help="Parquet文件路径")
    parser.add_argument("--show-action-state", action="store_true",default=True,
                        help="显示action vs state的曲线图")
    parser.add_argument("--show-episode-images", action="store_true",
                        help="显示所有帧的图像（支持键盘控制）")
    parser.add_argument("--show-step-images", type=int, default=None,
                        help="显示指定step的图像")
    parser.add_argument("--start-step", type=int, default=14,
                        help="打印state的起始步数")
    parser.add_argument("--end-step", type=int, default=700,
                        help="打印state的结束步数")
    parser.add_argument("--print-states", action="store_true",
                        help="打印指定范围内的state")
    
    args = parser.parse_args()
    
    # 创建可视化器实例
    visualizer = ParquetVisualizer(args.file)
    
    # 执行选定的操作
    if args.show_action_state:
        visualizer.show_action_state()
    
    if args.show_episode_images:
        visualizer.show_episode_images()
    
    if args.show_step_images is not None:
        visualizer.show_step_images(args.show_step_images)
    
    if args.print_states:
        for step in range(args.start_step, args.end_step):
            obs = visualizer.get_obs(step)
            print(f"Step {step} - State : {obs['state']}")
    
    # 如果没有指定任何操作，显示帮助信息
    if not any([args.show_action_state, args.show_episode_images, 
                args.show_step_images is not None, args.print_states]):
        parser.print_help()




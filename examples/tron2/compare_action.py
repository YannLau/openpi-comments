from openpi_client import websocket_client_policy
from openpi_client import image_tools
import einops
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from show_parquet import ParquetVisualizer
import time

class ActionCompare():
    def __init__(self,file_path) -> None:
        self.sim_env = ParquetVisualizer(file_path=file_path)
        self.gt_actions = self.sim_env.actions
        self.total_step = len(self.gt_actions)
        self.ws_client = websocket_client_policy.WebsocketClientPolicy(
            host="0.0.0.0",
            port=12222,
        )
        self.chunk_size = 50

    def compare_one_chunk(self,start_step):
        obs = self.sim_env.get_obs(start_step)
        pre_action = self.get_one_chunk_pre_action(obs=obs)

    def get_one_chunk_pre_action(self,obs):
        ts = time.time()
        ans = self.ws_client.infer(obs)
        te = time.time()
        print("推理耗时：",te-ts)
        pre_action = ans["actions"].tolist()
        assert len(pre_action) == self.chunk_size
        return pre_action
    

    def show_action_vs_pre_action(self,gt_actions,pre_actions):
        self.sim_env.show_action_state(actions=gt_actions, states_or_pre_action=pre_actions)


    def compare_action_prediction(self, num_chunks=3):
        """
        比较基于起始步的观测预测的chunk_size个动作与真实动作序列（一次性预测，所有数据画在同一个图中，包含推理点标记）
        
        Args:
            chunk_size: 每个chunk的长度（默认10）
            num_chunks: 要比较的chunk数量（默认3）
        """
        total_pred_steps = num_chunks * self.chunk_size
        # 检查是否足够长
        if total_pred_steps > self.total_step:
            print(f"❌ 无法比较，总步数 {self.total_step} 不足 {total_pred_steps} 步")
            return
        
        print(f"📊 正在比较 {num_chunks} 个连续chunk（每个大小 {self.chunk_size}），共 {total_pred_steps} 步...")

        
        # 获取预测动作序列（一次性预测）
        pred_actions = []
        
        # 为每个chunk获取预测
        for chunk_idx in range(num_chunks):
            start_step = chunk_idx * self.chunk_size
            # 获取起始步的观测
            obs = self.sim_env.get_obs(start_step)
            
            # 预处理图像 (与模型输入一致)
            for cam_name in obs["images"]:
                img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(obs["images"][cam_name], 224, 224)
                )
                obs["images"][cam_name] = einops.rearrange(img, "h w c -> c h w")
            
             # 模型一次性推理（生成self.chunk_size个动作）
            one_chunk_pred_action = self.get_one_chunk_pre_action(obs)
            
            # 将当前chunk的预测动作添加到列表
            pred_actions.extend(one_chunk_pred_action)
        
        pred_actions = np.array(pred_actions)  # (total_pred_steps, 14)
        
        # 计算推理点（每个chunk的起始步）
        inference_points = [chunk_idx * self.chunk_size for chunk_idx in range(num_chunks)]
        
        # 绘制前8维对比（所有数据在一个图中，包含推理点标记）
        fig1, axes1 = plt.subplots(8, 2, figsize=(18, 21), sharex=True)
        
        for i in range(8):
            left_y_values_pre= pred_actions[:total_pred_steps, i]
            left_y_values_action = self.gt_actions[:total_pred_steps, i]
            right_y_values_pre= pred_actions[:total_pred_steps, i+7]
            right_y_values_action = self.gt_actions[:total_pred_steps, i+7]

            if i == 7:
                left_y_values_action = self.gt_actions[:total_pred_steps, 14]
                left_y_values_pre= pred_actions[:total_pred_steps, 14]
                right_y_values_pre= pred_actions[:total_pred_steps, 15]
                right_y_values_action = self.gt_actions[:total_pred_steps, 15]
                
            # 绘制真实动作
            axes1[i][0].plot(range(total_pred_steps), left_y_values_action, 'b-', label='Ground Truth', linewidth=1.5)
            # 绘制预测动作
            axes1[i][0].plot(range(total_pred_steps), left_y_values_pre, 'r--', label='Predicted', linewidth=1.5)
            axes1[i][1].plot(range(total_pred_steps), right_y_values_action, 'b-', label='Ground Truth', linewidth=1.5)
            axes1[i][1].plot(range(total_pred_steps), right_y_values_pre, 'r--', label='Predicted', linewidth=1.5)
            

            # 添加推理点标记（垂直线）
            for point in inference_points:
                axes1[i][0].axvline(x=point, color='g', linestyle='--', alpha=0.7, label='Inference' if i == 0 else "")
                axes1[i][1].axvline(x=point, color='g', linestyle='--', alpha=0.7, label='Inference' if i == 0 else "")
            
            axes1[i][0].grid(True, linestyle='--', alpha=0.6)
            axes1[i][0].set_ylabel(f'Joint{i}')

        axes1[0][0].legend(loc='best')
        axes1[-1][1].set_xlabel('Time Step')
        axes1[-1][0].set_xlabel('Time Step')
        plt.suptitle(f'GT_Action vs Prediction (Steps 0 to {total_pred_steps-1})', fontsize=16)
        plt.tight_layout()
        plt.show()

    
if __name__ == "__main__":
    file_path="/home/limx/openpi/datasets/episode_000000.parquet"
    # 创建可视化器实例
    comp = ActionCompare(file_path)
    comp.compare_action_prediction()


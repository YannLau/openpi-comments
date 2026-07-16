"""
pi的部署脚本，连接机器人并推理
usage:
python examples/tron2/pi_client.py

注意：
1.目前是左臂+左手+右臂+右手的控制顺序和反馈顺序
2.默认是数据正常的，不需要乘以bias
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
    init_joints_clothes = [0.026899, 0.2612, -0.02709991, -1.5477003, 0.265, 0.0180999, -0.0614999,
                           0.008999, -0.269, 0.02069998, -1.5567001, -0.254, -0.02309972, 0.06469989]
    # init_joints_clothes = [-0.63819385,  0.83982128 ,-1.03469932, -1.24587011,  0.82801813, -0.23849821, -0.71935195,  
    #                      0.008999, -0.269,  0.02069998, -1.5567001, -0.254, -0.02309972, 0.06469989]
    init_head = [1.0467, -0.0139998]
    # 使用新的配置接口
    robot_config = Tron2Config(
        robot_ip="10.192.1.2",
        init_joints=init_joints_clothes,
        init_head=init_head
    )
    # env_config = EnvConfig(robot_config=robot_config)
    serial_to_name = {'serial_to_name':{
            "245022302696": 'head_camera_image',
            "409122274385": 'left_wrist_image',
            "230322276915": 'right_wrist_image'
        }}
    env_config = EnvConfig(
        robot_config=robot_config,
        interp_points=8,
        time_sync_tolerance=0.01,
        # raw_config = {'camera':serial_to_name}
    )
    with Tron2Env(env_config) as env:
        env.reset()
        
        ws_client_policy = websocket_client_policy.WebsocketClientPolicy(
            host='0.0.0.0',
            port=8000,
        )
        
        t = 0
        last_action = env.last_action[:14]
        
        # for logging
        record_state = []
        record_action = []
        
        while t < 100:
            print("\n\n", "#"*10, "begin infer", "#"*10)
            obs = env.get_obs()
            record_state.append(obs["state"].copy())
            
            rgb_images = obs["images"]
            Path("examples/tron2/recorded_rgb").mkdir(parents=True, exist_ok=True)
            [Image.fromarray(image_tools.convert_to_uint8(rgb_images[c]) if rgb_images[c].dtype != np.uint8 else rgb_images[c]).save(f"examples/tron2/recorded_rgb/t_{t:04d}_{c}.png") for c in rgb_images]
            print(f"states:{obs['state']}")
            
            for cam_name in rgb_images:
                img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(obs["images"][cam_name], 224, 224)  
                )
                obs["images"][cam_name] = einops.rearrange(img, "h w c -> c h w")
            
            ts = time.time()
            ans = ws_client_policy.infer(obs)
            te = time.time()
            print("infer time:", te-ts)
            
            action_plan = ans['actions']
            actions = np.stack(action_plan, axis=0)
            
            print("左臂开始:", actions[0][:8])
            print("右臂开始:", actions[0][8:])
            print("左臂结束:", actions[-1][:8])
            print("右臂结束:", actions[-1][8:])
            
            infer_time = time.time()
            record_action.append(actions)
            
            for action in actions:
                arm_action = np.concatenate((action[:7], action[8:15]))
                error = np.abs(arm_action - last_action)
                id = np.argmax(error)
                max_diff = error[id]
                # if action[7] < 0.7:
                #     action[7] = 0  # gripper close
                #     print("gripper open")
                if max_diff >= 0.5:
                    print(f"joint {id} 's error is {max_diff}")
                
                env.step(action)
                last_action = arm_action
                
                # time.sleep(0.03)  # 控制频率，避免过快执行
            
            # t += 1
        
        com_array = np.vstack(record_action)
        com2_array = np.vstack(record_state)
        np.savetxt('examples/tron2/clothes_action_data2.csv', com_array, delimiter=',', fmt='%.3f')
        np.savetxt('examples/tron2/clothes_state_data2.csv', com2_array, delimiter=',', fmt='%.3f')
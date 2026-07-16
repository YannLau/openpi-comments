
import numpy as np
import time

from real_env import Tron2Env
from real_env import GR00TPolicy

# 使用示例
if __name__ == "__main__":
    groot_init_joints = [-0.63819385,  0.83982128 ,-1.03469932, -1.24587011,  0.82801813, -0.23849821, -0.71935195,  
                         0.008999, -0.269,  0.02069998, -1.5567001, -0.254, -0.02309972, 0.06469989]
    env = Tron2Env(robot_ip="10.192.1.2",init_joints=groot_init_joints)
    env.reset()

    policy = GR00TPolicy(host="localhost",port=5555)
    t = 0
    bias = np.array([-1,-1,-1,-1,1,-1,-1, 1,  1,-1,-1,1,1,1,-1,  1])# 该顺序

    last_action = env.first_action
    record_state = []
    record_action = []
    while t < 100:
        print("\n\n","#"*10,"begin infer ","#"*10)
        obs = env.get_obs()
        record_state.append(obs["state"].copy())
        print(f"states:{obs['state']}")
        # 抓香蕉的数据需要再乘以bias
        obs['state'] = obs['state']*bias
        # obs['state'] = np.concatenate([obs['state'][:7],obs['state'][7:8],obs["state"][8:15],obs['state'][15:]])*bias
        # rgb_images = obs["images"]
        # for cam_name in rgb_images:
        #         img = image_tools.convert_to_uint8(
        #             image_tools.resize_with_pad(obs["images"][cam_name], 224, 224)  
        #         )
        #         obs["images"][cam_name] = einops.rearrange(img, "h w c -> c h w") # hum: take a lot of time at first time
        ts = time.time()
        actions = policy.get_action(obs)
        te = time.time()
        print("infer time: ", te-ts)
        # left_arm, left_hand, right_arm, right_hand顺序
        actions = actions * bias
        # actions = np.concatenate( (actions[:][:7],actions[:][14:15],actions[:][7:14],actions[:][15:]))
        # print("left arm 0",np.concatenate( (actions[0][:7],actions[0][14:15])))
        # print("right arm 0",np.concatenate( (actions[0][7:14],actions[0][15:])))
        print("left arm 0",actions[0][:8])
        print("right arm 0",actions[0][8:])

        print("left arm -1",actions[-1][:8])
        print("right arm -1",actions[-1][8:])
                                                                                                                                                                                                                                

        infer_time = time.time()
        # print("action execute timestamp:",infer_time)
        record_action.append(actions)
        for action in actions:
            arm_action = np.concatenate((action[:7],action[8:15]))
            error = np.abs(arm_action - last_action)
            id  = np.argmax(error)
            max_diff = error[id]
            # print("error:",error)
            env.step(action)
            last_action = arm_action
            if max_diff < 0.5:
                pass
            else:
                print(f"joint {id} 's error is {max_diff}")
                print(last_action)
                print(arm_action)
        t+=1
    com_array=np.vstack(record_action)
    com2_array=np.vstack(record_state)
    np.savetxt('/home/limx/cobot_magic_ubd/openpi/examples/tron2/banana_action_data2.csv',com_array,delimiter=',',fmt='%.3f')
    np.savetxt('/home/limx/cobot_magic_ubd/openpi/examples/tron2/banana_state_data2.csv',com2_array,delimiter=',',fmt='%.3f')

import numpy as np

from real_env import Tron2Env, EnvConfig
from robot_utils import Tron2, Tron2Config
from show_parquet import ParquetVisualizer
from mcap_ros2.reader import read_ros2_messages

def get_mcap_data(file_path):
    left_ee_pose_list = []
    right_ee_pose_list = []
    states_list = []
    gripper_list =[]
    for msg in read_ros2_messages(file_path):
        if 'joint_state' in msg.channel.topic:
            states_list.append(msg.ros_msg.position)
        if msg.channel.topic == "/left_arm/ee_pose":
            left_arm_x = msg.ros_msg.pose.position.x
            left_arm_y = msg.ros_msg.pose.position.y
            left_arm_z = msg.ros_msg.pose.position.z
            left_arm_orin_x = msg.ros_msg.pose.orientation.x
            left_arm_orin_y = msg.ros_msg.pose.orientation.y
            left_arm_orin_z = msg.ros_msg.pose.orientation.z
            left_arm_orin_w = msg.ros_msg.pose.orientation.w
            left_arm_pose = [left_arm_x, left_arm_y, left_arm_z,left_arm_orin_x, left_arm_orin_y, left_arm_orin_z,left_arm_orin_w]
            left_ee_pose_list.append(left_arm_pose)

        if msg.channel.topic == "/right_arm/ee_pose":
            right_arm_x = msg.ros_msg.pose.position.x
            right_arm_y = msg.ros_msg.pose.position.y
            right_arm_z = msg.ros_msg.pose.position.z
            right_arm_orin_x = msg.ros_msg.pose.orientation.x
            right_arm_orin_y = msg.ros_msg.pose.orientation.y
            right_arm_orin_z = msg.ros_msg.pose.orientation.z
            right_arm_orin_w = msg.ros_msg.pose.orientation.w

            right_arm_position = [right_arm_x, right_arm_y, right_arm_z,right_arm_orin_x, right_arm_orin_y, right_arm_orin_z,right_arm_orin_w]
            right_ee_pose_list.append(right_arm_position)
        if msg.channel.topic =='/gripper_state':
            gripper_state = msg.ros_msg.position
            gripper_list.append(gripper_state)
    return states_list, gripper_list, left_ee_pose_list, right_ee_pose_list

if __name__ == "__main__":
    data_type = "parquet"  # "mcap"  or "parquet"
    init_joints = [0.026899, 0.2612, -0.02709991, -1.5477003,  0.265, 0.0180999 , -0.0614999,
                   0.008999, -0.269,  0.02069998, -1.5567001, -0.254, -0.02309972, 0.06469989]
    
    if data_type == "mcap":
        file_path = "examples/tron2/test_dataset/mcaps_1770349378837/LimxT-shirt30_2026-01-26_15-07-34_0.mcap"
        states_list, gripper_list, left_ee_pose_list, right_ee_pose_list = get_mcap_data(file_path)
        
        # 使用新的配置接口
        robot_config = Tron2Config(
            robot_ip="10.192.1.2",
            init_joints=init_joints
        )
        
        with Tron2(robot_config) as robot:
            curr_ee_pose = robot.get_ee_poses()
            first_left_ee_pose = left_ee_pose_list[0]
            first_right_ee_pose = right_ee_pose_list[0]
            first_action = first_left_ee_pose + first_right_ee_pose
            robot.movep(first_action)
            
            last_left_ee_pose = first_left_ee_pose
            last_right_ee_pose = first_right_ee_pose
            
            for step in range(len(left_ee_pose_list)):
                left_ee_pose = left_ee_pose_list[step]
                right_ee_pose = right_ee_pose_list[step]
                left_gripper = gripper_list[step][0]
                rigit_gripper = gripper_list[step][1]
                print(f"Step {step} - left_ee_pose : {left_ee_pose}, right_ee_pose : {right_ee_pose}")
                
                err_vec = np.abs(np.array(left_ee_pose + right_ee_pose) - np.array(last_left_ee_pose + last_right_ee_pose))
                id = np.argmax(err_vec)
                max_diff = err_vec[id]
                
                if max_diff < 0.5:
                    robot.set_gripper(left_opening=left_gripper,right_opening=rigit_gripper)
                    robot.servop(left_ee_pose, right_ee_pose)
                    last_left_ee_pose = left_ee_pose
                    last_right_ee_pose = right_ee_pose
                else:
                    print(f"joint {id} 's error is {max_diff}")
    else:
        # 使用新的配置接口
        robot_config = Tron2Config(
            robot_ip="10.192.1.2",
            init_joints=init_joints
        )
        env_config = EnvConfig(robot_config=robot_config)
        
        with Tron2Env(env_config) as env:
            env.reset()
            file_path = "examples/tron2/test_dataset/lerobot_limx-tron2_2Flerobot_dataset_1770432725286/lerobot_dataset/data/chunk-000/episode_000000.parquet"
            
            # 创建可视化器实例
            visualizer = ParquetVisualizer(file_path)
            last_arm_action = env.last_action[:14]
            for step in range(13, 700):
                # 获取第step步的观测数据
                obs = visualizer.get_obs(step)
                states = obs["state"]
                curr_arm_action = np.concatenate([states[:7], states[8:15]])
                # print(f"Step {step} - actions : {states}")

                error = np.abs(curr_arm_action - last_arm_action)
                max_diff = np.max(error[:14])
                id = np.argmax(error)
                
                if max_diff < 0.5:
                    env.step(states)
                    last_arm_action = curr_arm_action
                else:
                    print(f"joint {id} 's error is {max_diff}")
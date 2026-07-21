import numpy as np
import matplotlib.pyplot as plt

actions = np.loadtxt('examples/tron2/clothes_action_data2.csv',delimiter=',')
states = np.loadtxt('examples/tron2/clothes_state_data2.csv',delimiter=',')
chunk_size = int(len(actions)/len(states))
print(actions.shape)
print(states.shape)
print(chunk_size)
max_infer_times  = 5# states.shape[0]
state_limit = int(max_infer_times)
action_limit = state_limit * chunk_size
states_x_value = range(0,state_limit*chunk_size,int(chunk_size))
x_value = range(0,action_limit)
fig1, axes1 = plt.subplots(int(actions.shape[1]/2), 2, figsize=(18, 3*actions.shape[1]/2))

for i in range(int(actions.shape[1]/2)):
    # 绘制 action/state 第i维
    
    left_y_values_state = states[:state_limit,i]
    left_y_values_action = actions[:action_limit,i]
    right_y_values_state = states[:state_limit,i+8]
    right_y_values_action = actions[:action_limit,i+8]

    # if i == 7:
    #     left_y_values_action = actions[:action_limit,14]
    #     left_y_values_state = states[:state_limit,14]
    #     right_y_values_state = states[:state_limit,15]
    #     right_y_values_action = actions[:action_limit,15]

    # axes1[i][0].plot(states_x_value, left_y_values_state, label=f'State', color='tab:blue', linewidth=1.5)
    # axes1[i][0].plot(x_value, left_y_values_action, label=f'Action', color='tab:orange', linewidth=1.5)
    # axes1[i][1].plot(states_x_value, right_y_values_state, label=f'State', color='tab:blue', linewidth=1.5)
    # axes1[i][1].plot(x_value, right_y_values_action, label=f'Action', color='tab:orange', linewidth=1.5)

    axes1[i][0].scatter(states_x_value, left_y_values_state, label='State', color='tab:blue', s=4.5, alpha=1)
    axes1[i][0].scatter(x_value, left_y_values_action, label='Action', color='tab:orange', s=4.5, alpha=1.0)
    axes1[i][1].scatter(states_x_value, right_y_values_state, label='State', color='tab:blue', s=4.5, alpha=1)
    axes1[i][1].scatter(x_value, right_y_values_action, label='Action', color='tab:orange', s=4.5, alpha=1.0)

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

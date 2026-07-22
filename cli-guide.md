# Mujoco-aloha 仿真命令

uv run scripts/serve_policy.py policy:checkpoint \
      --policy.config=my_pi0_aloha_sim \
      --policy.dir=/home/punk/yann_repo/para_check_pi0_aloha_sim/yann_paras/checkpoint/openpi-assets/checkpoints/pi0_aloha_sim

uv run scripts/serve_policy.py policy:checkpoint \
      --policy.config=yann_pi0_aloha_sim \
      --policy.dir=/home/punk/yann_repo/para_check_pi0_aloha_sim/yann_paras/checkpoint/openpi-assets/checkpoints/pi0_aloha_sim    

MUJOCO_GL=egl uv run examples/aloha_sim/main.py

  # 测试默认随机位置（等效于不指定）
  MUJOCO_GL=egl uv run examples/aloha_sim/main.py

--args.box-pose 0.2 0.5 0.05 1 0 0 0

  # 测试 Cube 放在右侧远处 (x=0.4, y=0.3)
 MUJOCO_GL=egl uv run examples/aloha_sim/main.py --args.box-pose 0.40 0.30 0.05 1 0 0 0

  # 测试 Cube 放在左侧 (x=0.05, y=0.55)  
MUJOCO_GL=egl uv run examples/aloha_sim/main.py --args.box-pose 0.05 0.55 0.05 1 0 0 0

  # 测试 Cube 放在正前方远处 (x=0.25, y=0.50)
MUJOCO_GL=egl uv run examples/aloha_sim/main.py --args.box-pose 0.25 0.50 0.05 1 0 0 0


----------------------------------

# single_data_test_lerobotv2.1_video_tron2

## 开启wandb
export WANDB_API_KEY="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"
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9  uv run scripts/train.py pi05_tron_single_data_lora --exp-name=tron2_single_data --overwrite --wandb_enabled --batch_size=1

## 关闭wandb
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9  uv run scripts/train.py pi05_tron_single_data_lora --exp-name=tron2_single_data --overwrite --no-wandb_enabled --batch_size=1

-----------------------------------


# 并行智算云中的运行命令

export HF_LEROBOT_HOME="/root/shared-nvme/data/"

uv run scripts/compute_norm_stats.py --config-name=<config_name>

export WANDB_MODE=disabled

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_tron_all_data_lora --exp-name=tron2lora --overwrite

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_tron_all_data_lora --exp-name=tron2lora --overwrite --save_interval=10000
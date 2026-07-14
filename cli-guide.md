uv run scripts/serve_policy.py policy:checkpoint \
      --policy.config=my_pi0_aloha_sim \
      --policy.dir=/home/punk/yann_repo/para_check_pi0_aloha_sim/yann_paras/checkpoint/openpi-assets/checkpoints/pi0_aloha_sim

uv run scripts/serve_policy.py policy:checkpoint \
      --policy.config=yann_pi0_aloha_sim \
      --policy.dir=/home/punk/yann_repo/para_check_pi0_aloha_sim/yann_paras/checkpoint/openpi-assets/checkpoints/pi0_aloha_sim    


MUJOCO_GL=egl uv run examples/aloha_sim/main.py

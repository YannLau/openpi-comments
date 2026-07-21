import os

from openpi.shared import download

os.environ['HF_ENDPOINT']= 'https://hf-mirror.com'
os.environ["OPENPI_DATA_HOME"] = f"/home/punk/yann_repo/para_check_pi0_aloha_sim/yann_paras/checkpoint"

path = "gs://openpi-assets/checkpoints/pi0_aloha_sim"

checkpoint_dir = download.maybe_download(path)

print("运行完成，保存在：",checkpoint_dir)
'''
用于数据转换，平台导出数据集的 parquet 文件重新写入，修正 schema 问题
'''

import os
import pyarrow.parquet as pq
from datasets import Dataset, Features, Sequence, Value, Image
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME

dir = HF_LEROBOT_HOME / "toast_bread"   # Path 对象，用 / 拼接
root1 = dir / "data" / "chunk-000"



# correct schema
features = Features({
    "observation.images.cam_high": Image(),
    "observation.images.cam_left_wrist": Image(),
    "observation.images.cam_right_wrist": Image(),
    "observation.images.cam_left_wrist_2": Image(),
    "observation.state": Sequence(Value("float32"), length=16),
    "action": Sequence(Value("float32"), length=16),
    "timestamp": Value("float32"),
    "frame_index": Value("int64"),
    "episode_index": Value("int64"),
    "index": Value("int64"),
    "task_index": Value("int64"),
})

def fix_file(path: str):
    # read old
    table = pq.read_table(path)

    # strip metadata
    schema = table.schema.remove_metadata()
    table = table.cast(schema)

    # make HF Dataset and recast
    ds = Dataset(table).cast(features)

    # overwrite in place
    tmp_path = path + ".tmp"
    ds.to_parquet(tmp_path)
    os.replace(tmp_path, path)

    print(f"fixed {path}")

# walk through all shards
for root, _, files in os.walk(root1):
    for fname in files:
        if fname.endswith(".parquet"):
            fix_file(os.path.join(root, fname))

print("\n all parquet shards in smol-libero2/data have been rewritten with Sequence schema ;0")
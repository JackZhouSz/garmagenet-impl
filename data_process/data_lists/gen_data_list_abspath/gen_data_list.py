"""
export PYTHONPATH=/data/lsr/code/style3d_gen
python data_process/data_lists/gen_data_list_abspath/gen_data_list.py \
    --garmage_dirs /data/AIGP/brep_reso_256_edge_snap_with_caption /data/AIGP/Q4/brep_reso_256_edge_snap \
    --output_dir /data/lsr/code/style3d_gen/_LSR/gen_data_list/output
"""

import os
import pickle
import random
from tqdm import tqdm
from glob import glob
import argparse

from torch.utils.data import random_split


def keep_percentage(lst, r):
    n = max(1, int(len(lst) * r))  # 至少保留1个元素，防止为空
    return random.sample(lst, n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--garmage_dirs', type=str, nargs='+', default="/data/AIGP/brep_reso_256_edge_snap")
    parser.add_argument('--output_dir', type=str, default=["_LSR/gen_data_list/output"])
    args = parser.parse_args()

    garment_list = []
    if isinstance(args.garmage_dirs, str):
        args.garmage_dirs = [args.garmage_dirs]
    for garment_dir in args.garmage_dirs:
        garmage_list_ex = sorted(glob(os.path.join(garment_dir, "*.pkl")))
        garment_list.extend(garmage_list_ex)

    # 按批次分别存放
    Q_type = ["Q1", "Q2", "Q4"]  # 批次
    Q_range = [1, 0.1, 1]  # 每批数据采样的比例
    Q_list = {k:[] for k in Q_type}
    for garment_path in tqdm(garment_list):
        with open(garment_path, "rb") as f:
            data = pickle.load(f)
            for Q in Q_type:
                if Q in data["data_fp"]:
                    Q_list[Q].append(garment_path)
                    break

    # 每个批次仅取一定百分比数量
    for i, Q in enumerate(Q_type):
        if len(Q_list[Q])>0:
            Q_list[Q] = keep_percentage(Q_list[Q], Q_range[i])

    garment_list = []
    for Q in Q_type:
        garment_list.extend(Q_list[Q])

    # 数据集划分
    split = [9., 1.]
    data_list = {"train": [], "val": []}
    split[0] = int(len(garment_list) * split[0]/sum(split))
    split[1] = len(garment_list) - split[0]

    idx_list = range(len(garment_list))
    train_dataset, val_dataset = random_split(idx_list, split)
    train_list, val_list = list(train_dataset), list(val_dataset)
    train_list = [garment_list[idx] for idx in train_list]
    val_list = [garment_list[idx] for idx in val_list]

    data_list["train"] = train_list
    data_list["val"] = val_list

    with open(os.path.join("data_process/data_lists", "stylexd_data_split_reso_256_Q1Q2Q4.pkl"), "wb") as f:
        pickle.dump(data_list, f)


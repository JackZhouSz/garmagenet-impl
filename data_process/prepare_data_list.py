import os
import pickle
import argparse
from glob import glob

from torch.utils.data import random_split


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--garmage_dir', type=str, default="resources/examples/garmages")
    parser.add_argument('--output_dir', type=str, default="resources/examples/datalist")
    parser.add_argument('--output_name', type=str, default="garmageset_split_9_1")
    args = parser.parse_args()

    garment_list = sorted(glob(os.path.join(args.garmage_dir, "*.pkl")))
    garment_list = [os.path.basename(fp) for fp in garment_list]

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

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f"{args.output_name}.pkl"), "wb") as f:
        pickle.dump(data_list, f)
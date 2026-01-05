import os
import pickle
import argparse
from glob import glob
from tqdm import tqdm

import numpy as np
from src.network import SketchEncoder


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default="/data/AIGP/GarmageSet_Opensource/images")
    parser.add_argument('--output_dir', type=str, default="/data/AIGP/GarmageSet_Opensource/feature_laion2b")
    args = parser.parse_args()

    root_dir = args.root_dir
    output_dir = args.output_dir

    encoder = SketchEncoder(encoder='LAION2B', device="cuda:0")

    exts = ['.png', '.jpg', '.jpeg']
    img_fp_list = []
    for ext in exts:
        img_fp_list.extend(glob(os.path.join(root_dir, "**", f"*_0{ext}"), recursive=True))

    for img_fp in tqdm(img_fp_list):
        sketch_feature = encoder.sketch_embedder_fn(img_fp)
        rel_fp = os.path.relpath(img_fp, root_dir)
        output_fp = os.path.join(output_dir, os.path.splitext(rel_fp)[0] + ".pkl")
        os.makedirs(os.path.dirname(output_fp), exist_ok=True)

        with open(output_fp, "wb") as f:
            pickle.dump(sketch_feature, f)


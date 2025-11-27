import os
import argparse
from glob import glob
from tqdm import tqdm

import numpy as np

from src.network import SketchEncoder


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    root_dir = args.root_dir
    output_dir = args.output_dir

    encoder = SketchEncoder(encoder='RADIO_V2.5-H', device="cuda:0")

    exts = ['.png', '.jpg', '.jpeg']
    img_fp_list = []
    for ext in exts:
        img_fp_list.extend(glob(os.path.join(root_dir, "**", f"*_0{ext}"), recursive=True))

    for img_fp in tqdm(img_fp_list):
        sketch_feature = encoder.sketch_embedder_fn(img_fp)
        sketch_feature = sketch_feature.detach().cpu().numpy()

        rel_fp = os.path.relpath(img_fp, root_dir)
        output_fp = os.path.join(output_dir, os.path.splitext(rel_fp)[0] + ".npy")
        os.makedirs(os.path.dirname(output_fp), exist_ok=True)
        np.save(output_fp, sketch_feature)

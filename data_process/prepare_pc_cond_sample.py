import os
import argparse
from tqdm import tqdm
from pathlib import Path

import torch
import numpy as np
import trimesh

from src.utils import ensure_directory
from src.pc_utils import garmageset_normalize


def sample_pointclouds_uniform(dataset_folder: str, output_folder: str, sample_num: int = 4096):

    obj_with_stitch_folder = Path(dataset_folder)
    obj_paths = [p for p in obj_with_stitch_folder.rglob('*.obj')]

    ensure_directory(output_folder)

    print("Processing sketches...")
    with torch.no_grad():
        for obj_path in tqdm(obj_paths):
            relative_path = Path(obj_path).relative_to(dataset_folder)

            uuid = os.path.splitext(os.path.basename(relative_path))[0]
            pc_save_path = os.path.join(output_folder, uuid, f"{uuid}.npy")
            print(pc_save_path)
            garment_mesh = trimesh.load(obj_path, force="mesh", process=False)
            vertices = np.array(garment_mesh.vertices)
            sample_indices = np.random.choice(len(vertices), sample_num, replace=False)
            pc_sampled = vertices[sample_indices]

            pc_sampled = garmageset_normalize(pc_sampled)

            ensure_directory(os.path.dirname(pc_save_path))
            np.save(pc_save_path, pc_sampled)
            print(f"Saved pointcloud to {pc_save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_folder', type=str,
                        default="resources/examples/raw")
    parser.add_argument('--pc_output_folder', type=str,
                        default="resources/examples/pc_cond_sample_uniformm")
    args = parser.parse_args()

    dataset_folder = args.dataset_folder
    pc_output_folder = args.pc_output_folder
    sample_pointclouds_uniform(dataset_folder, pc_output_folder, sample_num = 2048)
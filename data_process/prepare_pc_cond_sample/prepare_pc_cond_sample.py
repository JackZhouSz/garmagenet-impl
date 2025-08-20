"""
sample pointcloud (as condition) uniformlly

RUN ON 187

cd /data/lsr/code/style3d_gen/
export PYTHONPATH=/data/lsr/code/style3d_gen/
python data_process/prepare_pc_cond_sample/prepare_pc_cond_sample.py \
    --dataset_folder    /data/AIGP/objs_with_stitch \
    --pc_output_folder  /data/AIGP/pc_cond_sample_uniform

"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import numpy as np

import trimesh

from src.vis import pointcloud_visualize

def ensure_directory(path):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(path):
        os.makedirs(path)


# 对整个数据集进行的相同的 normalize
def styleXD_normalize(point):
    global_scale = 2000.0
    if point.shape[-1]==3:
        global_offset = (0., 1000., 0.)
    elif point.shape[-1]==2:
        global_offset = (0., 1000.)

    point_rtn = (point - global_offset) / (global_scale * 0.5)
    return point_rtn


def sample_pointclouds_uniform(dataset_folder: str, output_folder: str, sample_num: int = 4096):

    obj_with_stitch_folder = Path(dataset_folder)
    obj_paths = [p for p in obj_with_stitch_folder.rglob('*.obj')]

    ensure_directory(output_folder)

    """
    m = trimesh.load("/home/Ex1/Datasets/DeepFashion3D_V2/point_cloud/17/17-1.ply", process=False)
    pointcloud_visualize(np.array(m.vertices))
    肩带之间的宽度为0.185，女性肩宽约36-40，猜测DF的点云坐标乘以2可得米坐标系下的点云
    """

    print("Processing sketches...")
    with torch.no_grad():
        for obj_path in tqdm(obj_paths):
            relative_path = Path(obj_path).relative_to(dataset_folder)
            pc_save_path = Path(output_folder) / relative_path.with_suffix('.npy')

            garment_mesh = trimesh.load(obj_path, force="mesh", process=False)
            vertices = np.array(garment_mesh.vertices)
            sample_indices = np.random.choice(len(vertices), sample_num, replace=False)
            pc_sampled = vertices[sample_indices]

            pc_sampled = styleXD_normalize(pc_sampled)

            ensure_directory(pc_save_path.parent)
            np.save(pc_save_path, pc_sampled)
            print(f"Saved pointcloud to {pc_save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_folder', type=str,
                        default="data_process/prepare_pc_cond_sample/data")
    parser.add_argument('--pc_output_folder', type=str,
                        default="data_process/prepare_pc_cond_sample/output")
    args = parser.parse_args()

    dataset_folder = args.dataset_folder
    pc_output_folder = args.pc_output_folder
    sample_pointclouds_uniform(dataset_folder, pc_output_folder, sample_num = 2048)
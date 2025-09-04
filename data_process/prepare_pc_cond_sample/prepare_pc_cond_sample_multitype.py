"""
sample pointcloud in different ways

run on 187
cd /data/lsr/code/style3d_gen
export PYTHONPATH=/data/lsr/code/style3d_gen
python data_process/prepare_pc_cond_sample/prepare_pc_cond_sample_multitype.py \
    --obj_dataset_folder /data/AIGP/objs_with_stitch/ \
    --pkl_dataset_folders /data/AIGP/brep_reso_256_edge_snap/ /data/AIGP/Q4/ \
    --pc_output_folder /data/AIGP/pc_cond_sample_multitype/ \
    --sample_num 2048
"""

import os
import math
import argparse
import  pickle
from glob import glob
from tqdm import tqdm
from pathlib import Path

import torch
import numpy as np
import trimesh
from src.pc_utils import farthest_point_sample
# from src.vis import pointcloud_visualize


def pharse_data_fp(data_fp):
    data_fp = data_fp.replace("\\", "/")
    if "Q1" in data_fp or "Q2" in data_fp:
        data_fp = data_fp.split("工程数据")[1]
        data_fp = data_fp.replace("/objs/", "/")
    elif "Q4" in data_fp:
        # print(f"data_fp: {data_fp}")
        data_fp = data_fp.split("工程数据")[1]
        data_fp = data_fp.replace("_objs", "")
    return data_fp


def ensure_directory(path):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(path):
        os.makedirs(path)




def styleXD_normalize(point):
    global_scale = 2000.0
    if point.shape[-1]==3:
        global_offset = (0., 1000., 0.)
    elif point.shape[-1]==2:
        global_offset = (0., 1000.)

    point_rtn = (point - global_offset) / (global_scale * 0.5)
    return point_rtn

def process_one_obj(obj_path, obj_dataset_folder, output_folder, sample_num):
    try:
        relative_path = os.path.relpath(os.path.dirname(obj_path), obj_dataset_folder)
        sample_output_dir = os.path.join(output_folder, relative_path, "sampled_pc")
        os.makedirs(sample_output_dir, exist_ok=True)

        garment_mesh = None
        results = {}

        # surface uniform sampling
        pc_save_path = os.path.join(sample_output_dir, f"surface_uniform_{sample_num}.npy")
        if not os.path.exists(pc_save_path):
            garment_mesh = trimesh.load_mesh(obj_path, process=False)
            if isinstance(garment_mesh, trimesh.Trimesh):
                pc_sampled_surface_uniform = np.array(
                    trimesh.sample.sample_surface(garment_mesh, sample_num)[0]
                )
                np.save(pc_save_path, pc_sampled_surface_uniform)
            else:
                # 如果 obj 是点云 Scene，直接采样顶点
                vertices = np.array(garment_mesh.vertices)
                if len(vertices) > sample_num:
                    idx = np.random.choice(len(vertices), sample_num, replace=False)
                    vertices = vertices[idx]
                np.save(pc_save_path, vertices)

        # farest point sampling
        pc_save_path_2 = os.path.join(sample_output_dir, f"fps_{sample_num}.npy")
        if not os.path.exists(pc_save_path_2):
            if garment_mesh is None:
                garment_mesh = trimesh.load_mesh(obj_path, process=False)

            vertices = np.array(garment_mesh.vertices)
            pc_sampled_fps = farthest_point_sample(vertices, 2048, max_npoint=40000)[0]
            np.save(pc_save_path_2, pc_sampled_fps)

        results["status"] = "ok"
        results["path"] = str(obj_path)
        return results

    except Exception as e:
        return {"status": "error", "path": str(obj_path), "error": str(e)}

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

def sample_pointclouds_uniform(obj_dataset_folder: str, output_folder: str, sample_num: int = 4096, num_workers=8):
    obj_with_stitch_folder = Path(obj_dataset_folder)
    obj_paths = [p for p in obj_with_stitch_folder.rglob('*.obj')]

    print("Processing...")
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_one_obj, obj_path, obj_dataset_folder, output_folder, sample_num): obj_path
            for obj_path in obj_paths
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            try:
                result = future.result()
                results.append(result)
                if result["status"] == "error":
                    print(f"❌ Error processing {result['path']}: {result['error']}")
            except Exception as e:
                print(f"❌ Exception: {e}")

    print("✅ Done")
    return results

    # for obj_path in tqdm(obj_paths, desc="non uniform"):
        # try:
        #     print(obj_path)
        #     relative_path = os.path.relpath(os.path.dirname(obj_path), obj_dataset_folder)
        #     sample_output_dir = os.path.join(output_folder, relative_path, "sampled_pc")
        #     os.makedirs(sample_output_dir, exist_ok=True)
        #
        #     garment_mesh = None
        #
        #     # surface uniform sampling
        #     pc_save_path = os.path.join(
        #         sample_output_dir,
        #         f"surface_uniform_{sample_num}.npy"
        #     )
        #     if not os.path.exists(pc_save_path):
        #         garment_mesh = trimesh.load_mesh(obj_path, process=False)
        #         pc_sampled_surface_uniform = np.array(trimesh.sample.sample_surface(garment_mesh, sample_num)[0])
        #         np.save(pc_save_path, pc_sampled_surface_uniform)
        #
        #     # farest point sampling
        #     # FPS 不进行多轮采样，
        #     pc_save_path_2 = os.path.join(
        #         sample_output_dir,
        #         f"fps_{sample_num}.npy"
        #     )
        #     if not os.path.exists(pc_save_path_2):
        #         if garment_mesh is None:
        #             garment_mesh = trimesh.load_mesh(obj_path, process=False)
        #         vertices = np.array(garment_mesh.vertices)
        #         pc_sampled_fps = farthest_point_sample(np.array(vertices), 2048, max_npoint=40000)
        #         np.save(pc_save_path_2, pc_sampled_fps)
        #
        # except Exception as e:
        #     print(e)
        #     continue


# def sample_pointclouds_non_uniform(pkl_dataset_folders: str, output_folder: str, sample_num: int = 4096):
#     """
#     非均匀采样
#     """
#
#     pkl_paths = []
#     for pkl_dataset_folder in tqdm(pkl_dataset_folders):
#         pkl_lsit = glob(os.path.join(pkl_dataset_folder, "*.pkl"))
#         pkl_paths.extend(pkl_lsit)
#         del pkl_lsit
#
#     print("Processing...")
#     for pkl_path in tqdm(pkl_paths, desc="non uniform"):
#         try:
#             print(pkl_path)
#             with open(pkl_path, "rb") as f:
#                 data = pickle.load(f)
#             data_fp = data["data_fp"]
#
#             relative_path = pharse_data_fp(data_fp)
#             sample_output_dir = os.path.join(output_folder, relative_path, "sampled_pc")
#             os.makedirs(sample_output_dir, exist_ok=True)
#
#             pc_save_path = os.path.join(
#                 sample_output_dir,
#                 f"surface_uniform_{sample_num}.npy"
#             )
#             if os.path.exists(pc_save_path):
#                 continue
#
#             n_surfs = len(data['surf_bbox_wcs'])
#             surf_wcs = data["surf_wcs"].reshape(n_surfs, -1, 3)
#             surf_mask = data["surf_mask"].reshape(n_surfs, -1)
#             valid_pts = surf_wcs[surf_mask]
#             pc_sampled = valid_pts[np.random.randint(0, len(valid_pts), size=2048)]
#
#             pc_save_path = os.path.join(
#                 sample_output_dir,
#                 f"non_uniform_{sample_num}.npy"
#             )
#             np.save(pc_save_path, pc_sampled)
#         except Exception as e:
#             print(e)

def process_one_pkl(pkl_path, output_folder, sample_num):
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        data_fp = data["data_fp"]

        relative_path = pharse_data_fp(data_fp)  # 这里假设你已经实现了
        sample_output_dir = os.path.join(output_folder, relative_path, "sampled_pc")
        os.makedirs(sample_output_dir, exist_ok=True)

        pc_save_path = os.path.join(sample_output_dir, f"non_uniform_{sample_num}.npy")
        if os.path.exists(pc_save_path):
            return {"status": "skip", "path": pkl_path}

        n_surfs = len(data['surf_bbox_wcs'])
        surf_wcs = data["surf_wcs"].reshape(n_surfs, -1, 3)
        surf_mask = data["surf_mask"].reshape(n_surfs, -1)
        valid_pts = surf_wcs[surf_mask]

        if len(valid_pts) < sample_num:
            return {"status": "error", "path": pkl_path, "error": "Not enough points"}

        pc_sampled = valid_pts[np.random.randint(0, len(valid_pts), size=sample_num)]
        np.save(pc_save_path, pc_sampled)

        return {"status": "ok", "path": pkl_path}
    except Exception as e:
        return {"status": "error", "path": pkl_path, "error": str(e)}


def sample_pointclouds_non_uniform(pkl_dataset_folders, output_folder: str, sample_num: int = 4096, num_workers: int = 8):
    """
    非均匀采样 (多进程版)
    """
    # 收集所有 pkl 文件
    pkl_paths = []
    for pkl_dataset_folder in tqdm(pkl_dataset_folders, desc="Scanning"):
        pkl_list = glob(os.path.join(pkl_dataset_folder,"**", "*.pkl"), recursive=True)
        pkl_paths.extend(pkl_list)

    print("Processing...")
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_one_pkl, pkl_path, output_folder, sample_num): pkl_path
            for pkl_path in pkl_paths
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="non uniform"):
            result = future.result()
            results.append(result)
            if result["status"] == "error":
                print(f"❌ Error processing {result['path']}: {result['error']}")
            elif result["status"] == "skip":
                pass  # 已存在，跳过
            else:
                pass  # 正常完成

    print("✅ Done")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_dataset_folder', type=str,
                        default="data_process/prepare_pc_cond_sample/data")
    parser.add_argument('--pkl_dataset_folders', type=str, nargs="+",
                        default=["data_process/prepare_pc_cond_sample/data_pkl"])
    parser.add_argument('--pc_output_folder', type=str,
                        default="data_process/prepare_pc_cond_sample/output_multitype")
    parser.add_argument('--sample_num', type=int, default=2048)
    args = parser.parse_args()

    obj_dataset_folder = args.obj_dataset_folder
    pkl_dataset_folders = args.pkl_dataset_folders
    pc_output_folder = args.pc_output_folder
    ensure_directory(pc_output_folder)
    sample_num = args.sample_num

    # # mesh表面均匀点云采样 + FPS点云采样
    # sample_pointclouds_uniform(obj_dataset_folder, pc_output_folder, sample_num = sample_num)

    # # 非均匀点云采样
    sample_pointclouds_non_uniform(pkl_dataset_folders, pc_output_folder, sample_num = sample_num)

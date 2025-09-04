"""
对于在inference过程中保存了去噪过程的data
可视化去噪过程
"""

import os
import pickle
import argparse
from glob import glob
from tqdm import tqdm

import torch
import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib.colors import to_hex
from torchvision.utils import make_grid
from PIL import Image
from matplotlib.colors import to_rgb
from diffusers import DDPMScheduler
from src.experiments.export_denoising_vid_byAddnoise.vis_utils import *
from src.vis import get_visualization_steps  #, draw_per_panel_geo_imgs
from src.network import AutoencoderKLFastDecode, AutoencoderKLFastEncode
from src.utils import randn_tensor, _denormalize_pts


def _pad_arr(arr, pad_size=10, pad_value=0):
    return np.pad(
        arr,
        ((pad_size, pad_size), (pad_size, pad_size), (0, 0)),   # pad size to each dimension, require tensor to have size (H,W, C)
        mode='constant',
        constant_values=pad_value)


def merge_images(image_list, pad_width=10):
    """
    将 Nx256x256x4 的图片列表合并成方形网格，图片间用透明 padding 分隔。

    参数:
        image_list: list of np.ndarray, 每个形状为 (256, 256, 4)，值范围 0-1
        pad_width: int, 图片间的透明 padding 宽度（像素）

    返回:
        merged_image: np.ndarray, 合并后的图像，形状为 (H, W, 4)，值范围 0-1
    """
    # 获取图片数量
    N = len(image_list)
    if N == 0:
        return np.zeros((256, 256, 4), dtype=np.float32)

    # 确定网格大小
    grid_size = math.ceil(math.sqrt(N))
    rows, cols = grid_size, grid_size

    # 计算输出图像尺寸
    img_h, img_w, channels = image_list[0].shape
    out_h = rows * img_h + (rows - 1) * pad_width
    out_w = cols * img_w + (cols - 1) * pad_width

    # 初始化全透明输出图像
    merged_image = np.zeros((out_h, out_w, channels), dtype=np.float32)

    # 放置每张图片
    for idx in range(min(N, rows * cols)):
        row = idx // cols
        col = idx % cols

        # 计算图片的起始坐标
        start_y = row * (img_h + pad_width)
        start_x = col * (img_w + pad_width)

        # 放置图片
        merged_image[start_y:start_y + img_h, start_x:start_x + img_w, :] = image_list[idx]

    return merged_image


def draw_per_panel_geo_imgs(surf_ncs, surf_mask, colors, pad_size=5, out_fp=''):
    n_surfs = surf_ncs.shape[0]
    reso = int(surf_ncs.shape[1] ** 0.5)

    framed_imgs = []

    _surf_ncs = surf_ncs.reshape(n_surfs, reso, reso, 3)
    _surf_mask = surf_mask.reshape(n_surfs, reso, reso, 1)

    # 限定图像范围防止越界
    _surf_ncs[_surf_ncs>1] = 1
    _surf_ncs[_surf_ncs<-1] = -1

    for idx in range(n_surfs):
        mask_img = _surf_mask[idx, ...].astype(np.float32)

        _inv_mask_img = 1.0 - mask_img

        _padded_mask = _pad_arr(_inv_mask_img * 0.33, pad_size=pad_size, pad_value=1.0)

        _cur_color = colors[idx]
        if type(_cur_color) is str: _cur_color = to_rgb(_cur_color)

        _bg_img = np.zeros_like(_padded_mask.repeat(3, axis=-1)) + np.asarray(_cur_color)[None, None, :3]
        _bg_img = np.concatenate([_bg_img * _padded_mask, _padded_mask], axis=-1)

        _fg_img = np.concatenate([(np.clip(_surf_ncs[idx, ...], -1.0, 1.0) + 1.0) * 0.5, _surf_mask[idx, ...]], axis=-1)
        _fg_img = _pad_arr(_fg_img, pad_size=pad_size, pad_value=0.0)

        fused_img = _bg_img + _fg_img

        framed_imgs.append(fused_img)

        # fused_pil_img = Image.fromarray((fused_img * 255).astype(np.uint8))
        #
        # fused_pil_img.save(os.path.join(os.path.dirname(out_fp),f"test_{idx}.png"))
    combined_images = merge_images(framed_imgs, pad_width=10)
    combined_pil_images = Image.fromarray((combined_images * 255).astype(np.uint8))
    combined_pil_images.save(out_fp)
    return framed_imgs


if __name__ == "__main__":
    raise NotImplementedError("This function as been merged into src/experiments/batch_inference/batch_inference.py")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default="/home/Ex1/ProjectFiles/Pycharm_MyPaperWork/style3d_gen/src/experiments/export_denoising_vid/pkl_files")
    parser.add_argument("--vae", type=str,
                        default="/data/lsr/models/style3d_gen/surf_vae/stylexd_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e800.pt")
    # parser.add_argument("--data_dir", type=str,
    #                     default="/home/Ex1/data/resources/其它/2025_06_11_伙伴大会的视频素材/挑选出的2_做数据/SketchCond/generated")
    # parser.add_argument("--vae", type=str,
    #                     default="/home/Ex1/data/models/style3d_gen/surf_vae/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e0800.pt")
    args = parser.parse_args()

    data_dir = args.data_dir
    vae = args.vae

    # for data_dir, vae in [
    #     [
    #         "/home/Ex1/data/resources/其它/2025_06_11_伙伴大会的视频素材/挑选出的2_做数据/CaptionCond/generated",
    #         "/data/lsr/models/style3d_gen/surf_vae/stylexd_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e800.pt"
    #     ],
    #     [
    #         "/home/Ex1/data/resources/其它/2025_06_11_伙伴大会的视频素材/挑选出的2_做数据/PointcloudCond/generated",
    #         "/home/Ex1/data/models/style3d_gen/surf_vae/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e0800.pt"
    #     ],
    #     [
    #         "/home/Ex1/data/resources/其它/2025_06_11_伙伴大会的视频素材/挑选出的2_做数据/SketchCond/generated",
    #         "/home/Ex1/data/models/style3d_gen/surf_vae/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e0800.pt"
    #     ]
    # ]:
    pkl_list = sorted(glob(os.path.join(data_dir, "*.pkl")))

    output_dir = data_dir
    os.makedirs(output_dir, exist_ok=True)

    # init models ---------------------------------------------------------------
    ddpm_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule='linear',
        prediction_type='epsilon',
        beta_start=0.0001,
        beta_end=0.02,
        clip_sample=False,
    )
    latent_channels, latent_size = 1, 8
    block_dims = [16,32,32,64,64,128]
    reso = 256
    surf_vae_decoder = AutoencoderKLFastDecode( in_channels=6,
                                                out_channels=6,
                                                down_block_types=['DownEncoderBlock2D']*len(block_dims),
                                                up_block_types=['UpDecoderBlock2D']*len(block_dims),
                                                block_out_channels=block_dims,
                                                layers_per_block=2,
                                                act_fn='silu',
                                                latent_channels=1,
                                                norm_num_groups=8,
                                                sample_size=reso
                                                )
    surf_vae_decoder.load_state_dict(torch.load(vae), strict=False)
    surf_vae_decoder.to("cuda").eval()

    surf_vae_encoder = AutoencoderKLFastEncode(
        in_channels=6,
        out_channels=6,
        down_block_types=['DownEncoderBlock2D']*len(block_dims),
        up_block_types=['UpDecoderBlock2D']*len(block_dims),
        block_out_channels=block_dims,
        layers_per_block=2,
        act_fn='silu',
        latent_channels=latent_channels,
        norm_num_groups=8,
        sample_size=reso,
    )
    surf_vae_encoder.load_state_dict(torch.load(args.vae), strict=False)
    surf_vae_encoder.to("cuda").eval()

    # denoising ---------------------------------------------------------------
    for idx, pkl_fp in enumerate(pkl_list):
        sub_basename = os.path.basename(pkl_fp).replace(".pkl", "")
        sub_output_dir = os.path.join(output_dir, sub_basename+"_denoising")
        os.makedirs(sub_output_dir, exist_ok=True)

        with open(pkl_fp, "rb") as f:
            data = pickle.load(f)
        n_surfs = len(data["surf_bbox"])

        denoising_data = data["denoising"]["denoising_data"]

        img_idx = 0
        per_panel_denoising_dir = os.path.join(sub_output_dir, "per_panel_denoising")
        denoiseing_3D_dir = os.path.join(sub_output_dir, "denoising_3D")
        denoiseing_2D_dir = os.path.join(sub_output_dir, "denoising_2D")
        os.makedirs(per_panel_denoising_dir, exist_ok=True)
        os.makedirs(denoiseing_3D_dir, exist_ok=True)
        os.makedirs(denoiseing_2D_dir, exist_ok=True)

        for step in tqdm(denoising_data):
            # Save per panel denoising ===
            surf_z_denoising = torch.tensor(denoising_data[step]["surf_z_latent"]).to("cuda")
            with torch.no_grad():
                decoded_surf_pos_denoising = surf_vae_decoder(surf_z_denoising.view(-1, latent_channels, latent_size, latent_size))
            decoded_surf_pos_denoising = decoded_surf_pos_denoising.permute(0, 2, 3, 1).detach().cpu().numpy()
            geo_denoising = decoded_surf_pos_denoising[..., :3]
            mask_denoising = decoded_surf_pos_denoising[..., -1]>0.
            colors = [to_hex(plt.cm.coolwarm(i)) for i in np.linspace(0, 1, n_surfs)]
            draw_per_panel_geo_imgs(
                geo_denoising.reshape(n_surfs, -1, 3),
                mask_denoising.reshape(n_surfs, -1), colors, pad_size=5,
                out_fp=os.path.join(per_panel_denoising_dir, f"{img_idx}".zfill(4)+"_t="+f"{step}".zfill(4)+".png"))

            # Save 3D 2D denoising ===
            surf_pos = denoising_data[step]["surf_pos"]
            _BBox3D_ = surf_pos[:, :6]
            _BBox2D_ = surf_pos[:, 6:]

            _surf_uv_bbox_wcs_ = np.zeros((n_surfs, 6))
            _surf_uv_bbox_wcs_[:, [0, 1, 3, 4]] = _BBox2D_
            _BBox2D_ = _surf_uv_bbox_wcs_

            _surf_ncs_ = decoded_surf_pos_denoising[..., :3].reshape(n_surfs, -1, 3)
            _surf_uv_ncs_ = decoded_surf_pos_denoising[..., 3:5].reshape(n_surfs, -1, 2)
            _surf_mask_ = decoded_surf_pos_denoising[..., -1].reshape(n_surfs, -1) > 0.0

            _Point_3D_ = _denormalize_pts(_surf_ncs_, _BBox3D_)
            _Point_2D_ = _denormalize_pts(_surf_uv_ncs_, _BBox2D_[..., [0, 1, 3, 4]])
            _Point_2D_ = np.concatenate([_Point_2D_, np.zeros((*_Point_2D_.shape[:2], 1))], axis=-1)

            # 可视化3D点+bbox
            draw_bbox_geometry(
                bboxes=_BBox3D_,
                bbox_colors=colors,
                points=_Point_3D_,
                point_masks=_surf_mask_,
                point_colors=colors,
                num_point_samples=2000,
                all_bboxes=_BBox3D_,
                output_fp=os.path.join(denoiseing_3D_dir, f"{img_idx}".zfill(4) + f"geometry_denoising_t=" + f"{step}".zfill(4) + ".png"),
                visatt_dict={
                    "bboxmesh_opacity": 0.12,
                    "point_size": 10,
                    "point_opacity": 0.8,
                    "bboxline_width": 8
                },
            )
            # 可视化2D点+bbox
            draw_bbox_geometry(
                bboxes=_BBox2D_,
                bbox_colors=colors,
                points=_Point_2D_,
                point_masks=_surf_mask_,
                point_colors=colors,
                num_point_samples=2000,
                all_bboxes=_BBox2D_,
                output_fp=os.path.join(denoiseing_2D_dir, f"{img_idx}".zfill(4) + f"geometry_denoising_t=" + f"{step}".zfill(4) + ".png"),
                visatt_dict={
                    "bboxmesh_opacity": 0.15,
                    "point_size": 10,
                    "point_opacity": 0.8,
                    "bboxline_width": 8,
                    "camera_eye_z": 1.5
                },
            )
            img_idx += 1


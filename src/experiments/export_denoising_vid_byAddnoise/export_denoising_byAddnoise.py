"""
对于没有在inference保存去噪过程的生成数据，
我们通过加噪还原去噪的过程
"""

import os
import pickle
import argparse
from glob import glob
from tqdm import tqdm

import torch
import numpy as np
from diffusers import DDPMScheduler
from matplotlib import pyplot as plt
from matplotlib.colors import to_hex


from src.experiments.export_denoising_vid_byAddnoise.vis_utils import *
from src.vis import get_visualization_steps
from src.network import AutoencoderKLFastDecode, AutoencoderKLFastEncode
from src.utils import randn_tensor, _denormalize_pts

from src.experiments.export_denoising_vid.export_denoising import draw_per_panel_geo_imgs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default="/home/Ex1/data/resources/Garmage_SigAisia2025/旧Teasor的/vae_noised")
    parser.add_argument("--output_root", type=str,
                        default="/home/Ex1/data/resources/Garmage_SigAisia2025_Revision/旧Teasor的_denoising/denoising_vid")
    parser.add_argument("--vae_ckpt_fp", type=str,
                        default='/data/lsr/models/style3d_gen/surf_vae/stylexd_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e800.pt')
    args = parser.parse_args()

    data_dir = args.data_dir
    pkl_list = sorted(glob(os.path.join(data_dir, "*.pkl")))

    output_dir = args.output_root
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
    vae_ckpt_fp = args.vae_ckpt_fp
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
    surf_vae_decoder.load_state_dict(torch.load(vae_ckpt_fp), strict=False)
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
    surf_vae_encoder.load_state_dict(torch.load(vae_ckpt_fp), strict=False)
    surf_vae_encoder.eval()

    # # data statistic ---------------------------------------------------------------
    # n_surf_list = []    # 用于计算操作过程中出现过的最大panel数量
    # all_bboxes3D = []   # 用于可视化时，让相机位置不变
    # _all_bboxes2D = []  # 用于可视化时，让相机位置不变
    # for pkl_fp in tqdm(pkl_list):
    #     with open(pkl_fp, "rb") as f:
    #         data = pickle.load(f)
    #     n_surf_list.append(len(data["surf_bbox"]))
    #     all_bboxes3D.append(data["surf_bbox"])
    #     _all_bboxes2D.append(data["surf_uv_bbox"])
    # max_surf = max(n_surf_list)
    # all_bboxes3D = np.concatenate(all_bboxes3D)
    # _all_bboxes2D = np.concatenate(_all_bboxes2D)
    # all_bboxes2D = np.zeros((_all_bboxes2D.shape[0],6))
    # all_bboxes2D[:, [0, 1, 3, 4]] = _all_bboxes2D

    # denoising ---------------------------------------------------------------
    for idx, pkl_fp in enumerate(pkl_list):
        sub_basename = os.path.basename(pkl_fp).replace(".pkl", "")
        sub_output_dir = os.path.join(output_dir, sub_basename)
        os.makedirs(sub_output_dir, exist_ok=True)

        with open(pkl_fp, "rb") as f:
            data = pickle.load(f)

        surf_bbox = data.get("surf_bbox") or data.get("surf_bbox_wcs")
        surf_uv_bbox = data.get("surf_uv_bbox") or data.get("surf_uv_bbox_wcs")
        surf_ncs = data["surf_ncs"].reshape(-1, reso, reso, 3)
        surf_uv_ncs = data["surf_uv_ncs"].reshape(-1, reso, reso, 2)
        surf_mask = data["surf_mask"].reshape(-1, reso, reso, 1).astype(np.float32) *2 -1
        # caption = data["caption"]
        n_surfs = len(surf_bbox)
        edited_mask = np.zeros(32, dtype=np.bool)
        edited_mask[:n_surfs] = True
        # Denoising ===
        sub_sub_output_dir1 = os.path.join(sub_output_dir, "denoising_3D")
        sub_sub_output_dir2 = os.path.join(sub_output_dir, "denoising_2D")
        per_panel_denoising_dir = os.path.join(sub_output_dir, "per_panel_denoising")
        os.makedirs(sub_sub_output_dir1, exist_ok=True)
        os.makedirs(sub_sub_output_dir2, exist_ok=True)
        os.makedirs(per_panel_denoising_dir, exist_ok=True)

        colors = [to_hex(plt.cm.coolwarm(i)) for i in np.linspace(0, 1, np.sum(edited_mask))]
        ddpm_scheduler.set_timesteps(1000)
        surf_pos_orig = torch.tensor(np.concatenate([surf_bbox, surf_uv_bbox], axis=-1))
        visualization_steps = get_visualization_steps()
        # visualization_steps = [999, 10, 0]
        surf_pos_noise = torch.randn(surf_pos_orig.shape)/2

        garmage_orig = torch.tensor(np.concatenate([surf_ncs, surf_uv_ncs, surf_mask], axis=-1))
        surf_z_orig = surf_vae_encoder(torch.tensor(garmage_orig).permute(0,3,1,2))
        surf_z_noise = randn_tensor((n_surfs, latent_channels, latent_size, latent_size))

        img_idx = 0

        for t in tqdm(ddpm_scheduler.timesteps, desc="Surf-Pos+Z Denoising"):
            if t in visualization_steps:
            # if t in [1, 0]:
                if t > 0:
                    surf_pos_noised = ddpm_scheduler.add_noise(surf_pos_orig, surf_pos_noise, t)
                    surf_pos_noise2 = torch.randn(surf_pos_orig.shape)/20
                    surf_pos_noised = ddpm_scheduler.add_noise(surf_pos_noised, surf_pos_noise2, t)
                else:
                    surf_pos_noised = surf_pos_orig

                # latentcode 按照t加噪，decode后作的为噪声加给garmage_orig
                surf_z_noised = ddpm_scheduler.add_noise(surf_z_orig, surf_z_noise, t).to("cuda")
                with torch.no_grad():
                    garmage_noised = surf_vae_decoder(surf_z_noised).permute(0, 2, 3, 1).detach().cpu()

                if t > 0:
                    garmage_noised = ddpm_scheduler.add_noise(garmage_orig, garmage_noised, t)
                else:
                    garmage_noised = garmage_orig

                # visualize
                _edited_mask = edited_mask[:n_surfs]

                # 所有panel的bbox
                _surf_bbox = surf_pos_noised.numpy()[..., :6]
                _surf_uv_bbox = surf_pos_noised.numpy()[..., 6:]
                _surf_uv_bbox_wcs_ = np.zeros((n_surfs, 6))
                _surf_uv_bbox_wcs_[:, [0, 1, 3, 4]] = surf_pos_noised[:, 6:].numpy()

                # 编辑过的panel的BBox
                _BBox3D_ = _surf_bbox[:, :6]
                _BBox2D_ = _surf_uv_bbox_wcs_

                # 所有panel的有噪声的Garmage
                garmage_noised = garmage_noised.numpy()
                _surf_ncs_noised_ = garmage_noised[..., :3].reshape(n_surfs, -1, 3)
                _surf_uv_ncs_noised_ = garmage_noised[..., 3:5].reshape(n_surfs, -1, 2)
                _surf_mask_noised_ = garmage_noised[..., -1:].reshape(n_surfs, -1) > 0.0

                # 编辑过的panel的有噪声的Garmage
                _surf_ncs_noised_ = _surf_ncs_noised_[_edited_mask]
                _surf_uv_ncs_noised_ = _surf_uv_ncs_noised_[_edited_mask]
                _surf_mask_noised_ = _surf_mask_noised_[_edited_mask]

                # 编辑过的panel的3d&2d的denormalized的点
                _Point_3D_ = _denormalize_pts(_surf_ncs_noised_, _BBox3D_)
                _Point_2D_ = _denormalize_pts(_surf_uv_ncs_noised_, _BBox2D_[...,[0, 1, 3, 4]])
                _Point_2D_ = np.concatenate([_Point_2D_, np.zeros((*_Point_2D_.shape[:2], 1))], axis=-1)

                # 可视化3D点+bbox ===
                draw_bbox_geometry(
                    bboxes=_BBox3D_,
                    bbox_colors=colors,
                    points=_Point_3D_,
                    point_masks=_surf_mask_noised_,
                    point_colors=colors,
                    num_point_samples=2000,
                    all_bboxes=_BBox3D_,
                    output_fp=os.path.join(sub_sub_output_dir1, f"{img_idx}".zfill(4) +  f"geometry_denoising_t="+f"{t}".zfill(4)+".png"),
                    visatt_dict = {
                        "bboxmesh_opacity": 0.12,
                        # "point_size": 10,
                        "point_opacity": 0.8,
                        "bboxline_width": 8
                    },
                )
                # 可视化2D点+bbox ===
                draw_bbox_geometry(
                    bboxes=_BBox2D_,
                    bbox_colors=colors,
                    points=_Point_2D_,
                    point_masks=_surf_mask_noised_,
                    point_colors=colors,
                    num_point_samples=2000,
                    all_bboxes=_BBox2D_,
                    output_fp=os.path.join(sub_sub_output_dir2, f"{img_idx}".zfill(4) +  f"geometry_denoising_t="+f"{t}".zfill(4)+".png"),
                    visatt_dict = {
                        "bboxmesh_opacity": 0.15,
                        "point_size": 10,
                        "point_opacity": 0.8,
                        "bboxline_width": 8,
                        "camera_eye_z": 1.5
                    },
                )

                # 可视化几何图 ===
                geo_denoising = _surf_ncs_noised_
                mask_denoising = _surf_mask_noised_
                colors = [to_hex(plt.cm.coolwarm(i)) for i in np.linspace(0, 1, n_surfs)]
                draw_per_panel_geo_imgs(
                    geo_denoising.reshape(n_surfs, -1, 3),
                    mask_denoising.reshape(n_surfs, -1),
                    colors,
                    pad_size=5,
                    out_fp=os.path.join(per_panel_denoising_dir, f"{img_idx}".zfill(4) + "_t=" + f"{t}".zfill(4) + ".png"))

                img_idx+=1
                # import sys
                # sys.exit(0)
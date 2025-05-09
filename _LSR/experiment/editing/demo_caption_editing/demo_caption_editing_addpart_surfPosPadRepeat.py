import os
import json
import pickle
import random
import argparse
from glob import glob
from tqdm import tqdm

import torch
import numpy as np
from torchvision.utils import make_grid

from matplotlib import pyplot as plt
from matplotlib.colors import to_hex

from src.network import AutoencoderKLFastDecode, AutoencoderKLFastEncode, SurfZNet, SurfPosNet, TextEncoder
from diffusers import DDPMScheduler  # , PNDMScheduler
from src.utils import randn_tensor
from src.vis import draw_bbox_geometry, draw_bbox_geometry_3D2D


def _denormalize_pts(pts, bbox):
    pos_dim =  pts.shape[-1]
    bbox_min = bbox[..., :pos_dim][:, None, ...]
    bbox_max = bbox[..., pos_dim:][:, None, ...]
    bbox_scale = np.max(bbox_max - bbox_min, axis=-1, keepdims=True) * 0.5
    bbox_offset = (bbox_max + bbox_min) / 2.0
    return pts * bbox_scale + bbox_offset


def init_models(args):

    block_dims = args.block_dims
    latent_channels = args.latent_channels
    sample_size = args.reso
    latent_size = sample_size//(2**(len(block_dims)-1))

    surf_vae_decoder = AutoencoderKLFastDecode( in_channels=6,
                                                out_channels=6,
                                                down_block_types=['DownEncoderBlock2D']*len(block_dims),
                                                up_block_types=['UpDecoderBlock2D']*len(block_dims),
                                                block_out_channels=block_dims,
                                                layers_per_block=2,
                                                act_fn='silu',
                                                latent_channels=latent_channels,
                                                norm_num_groups=8,
                                                sample_size=sample_size
                                                )
    surf_vae_decoder.load_state_dict(torch.load('/data/lsr/models/style3d_gen/surf_vae/stylexd_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e800.pt'), strict=False)
    surf_vae_decoder.to('cuda').eval()

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
        sample_size=sample_size,
    )
    surf_vae_encoder.load_state_dict(torch.load('/data/lsr/models/style3d_gen/surf_vae/stylexd_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e800.pt'), strict=False)
    surf_vae_encoder.to('cuda').eval()

    # pndm_scheduler = PNDMScheduler(
    #     num_train_timesteps=1000,
    #     beta_schedule='linear',
    #     prediction_type='epsilon',
    #     beta_start=0.0001,
    #     beta_end=0.02,
    # )

    ddpm_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule='linear',
        prediction_type='epsilon',
        beta_start=0.0001,
        beta_end=0.02,
        clip_sample=False,
    )

    # Load conditioning model
    if "text_encoder" in args: text_enc = TextEncoder(args.text_encoder, device='cuda')
    else: text_enc = None

    # Load SurfPos Net
    surfpos_model = SurfPosNet(
        p_dim=10,
        condition_dim=text_enc.text_emb_dim if "text_encoder" in args else -1,
        num_cf=-1
    )

    # Load SurfZ Net
    surfz_model = SurfZNet(
        p_dim=10,
        z_dim=latent_size**2*latent_channels,
        num_heads=12,
        condition_dim=text_enc.text_emb_dim if "text_encoder" in args else -1,
        num_cf=-1
    )

    if "text_encoder" in args:
        surfpos_model.load_state_dict(
            torch.load('/home/Ex1/data/models/style3d_gen/surf_pos/stylexd_surfpos_xyzuv_pad_repeat_cond_clip/ckpts/surfpos_e60000.pt')['model_state_dict'])
        surfz_model.load_state_dict(
            torch.load('/home/Ex1/data/models/style3d_gen/surf_z/stylexd_surfz_xyzuv_mask_latent1_mode_with_caption/ckpts/surfz_e100000.pt')['model_state_dict'])
    else:
        surfpos_model.load_state_dict(
            torch.load('/home/Ex1/data/models/style3d_gen/surf_pos/stylexd_surfpos_xyzuv_pad_repeat_uncond/ckpts/surfpos_e5000.pt')['model_state_dict'])
        surfz_model.load_state_dict(
            torch.load('/home/Ex1/data/models/style3d_gen/surf_z/stylexd_surfz_xyzuv_mask_latent1_mode/ckpts/surfz_e230000.pt')['model_state_dict'])

    surfpos_model.to('cuda').eval()
    surfz_model.to('cuda').eval()
    print('[DONE] Models initialized.')

    return {
        'surf_vae_decoder': surf_vae_decoder,
        'surf_vae_encoder': surf_vae_encoder,
        'ddpm_scheduler': ddpm_scheduler,
        'surfpos_model': surfpos_model,
        'surfz_model': surfz_model,
        'text_enc': text_enc,
        'latent_channels': latent_channels,
        'latent_size': latent_size
    }


def init_inference(models,
                  caption='',
                  output_fp='',
                  dedup=True,
                  vis=False,):
    batch_size = 1
    device = 'cuda'
    max_surf = 32
    bbox_dim = 10
    # caption="dress, fited, cinched waist, stand collar, cap sleeves"
    ddpm_scheduler = models['ddpm_scheduler']
    surfz_model = models['surfz_model']
    surfpos_model = models['surfpos_model']
    surf_vae_decoder = models['surf_vae_decoder']
    surf_vae_encoder = models['surf_vae_encoder']
    text_enc = models.get('text_enc', None)

    latent_channels = models['latent_channels']
    latent_size = models['latent_size']

    # text embedding
    text_embeding = text_enc(caption) \
        if caption is not None and text_enc is not None else None

    # SurfPos ===
    # generate bbox by denoising
    _surf_pos = randn_tensor((batch_size, max_surf, bbox_dim)).to(device)
    ddpm_scheduler.set_timesteps(1000)

    with torch.no_grad():
        _surf_pos_gt_noise = torch.randn(_surf_pos.shape).to(device)
        for t in tqdm(ddpm_scheduler.timesteps, desc="Surf-Pos Denoising"):
            timesteps = t.reshape(-1).to(device)
            pred = surfpos_model(_surf_pos, timesteps, condition=text_embeding)
            _surf_pos = ddpm_scheduler.step(pred, t, _surf_pos).prev_sample

    # [test] vis BBox
    colors = [to_hex(plt.cm.coolwarm(i)) for i in np.linspace(0, 1, max_surf)]
    _surf_pos_ = _surf_pos[:, :]
    _surf_uv_bbox_wcs_ = np.zeros((_surf_pos_.shape[-2], 6))
    _surf_uv_bbox_wcs_[:, [0, 1, 3, 4]] = _surf_pos_[0, :, 6:].detach().cpu().numpy()
    fig = draw_bbox_geometry_3D2D(
        bboxes=[_surf_pos_[0, :, :6].detach().cpu().numpy(), _surf_uv_bbox_wcs_],
        bbox_colors=colors,
        title=f"{caption}",
        # output_fp=output_fp.replace('.pkl', '_pointcloud.png'),
        show_num=True,
        fig_show="browser"
    )

    if dedup:
        for ii in range(batch_size):
            # bboxes = np.round(_surf_pos[ii].unflatten(-1, torch.Size([2, 5])).detach().cpu().numpy(), 4)  # [...,-2:]  # mask维度
            bboxes = np.round(
                torch.concatenate(
                    [_surf_pos[ii][:, :6].unflatten(-1, torch.Size([2, 3])), _surf_pos[ii][:, 6:].unflatten(-1, torch.Size([2, 2]))]
                    , dim=-1).detach().cpu().numpy(), 4)
            non_repeat = bboxes[:1]

            # _surf_pos_mask
            for bbox_idx in range(1, len(bboxes)):
                bbox = bboxes[bbox_idx]

                # （2D）判断BBox的大小是否非0
                bbox_threshold = 2e-6
                assert bbox_threshold >= 0. and bbox_threshold <= 1.
                v = 1
                is_thin = False  # 判断是否是细长型（但面积太小）
                for h in (bbox[1] - bbox[0])[3:]:
                    v *= h
                    if h > 0.02:
                        is_thin = True
                if v < bbox_threshold and not is_thin:
                    continue

                # （2D）和现有BBox是否重复
                bbox_threshold = 0.08
                assert bbox_threshold >= 0. and bbox_threshold <= 1.
                diff = np.max(np.max(np.abs(non_repeat - bbox)[..., -2:], -1), -1)
                same = diff < bbox_threshold
                bbox_rev = bbox[::-1]
                diff_rev = np.max(np.max(np.abs(non_repeat - bbox_rev)[..., -2:], -1), -1)
                same_rev = diff_rev < bbox_threshold
                if same.sum() >= 1 or same_rev.sum() >= 1:
                    continue

                # （3D）和现有BBox是否重复
                bbox_threshold = 0.08
                assert bbox_threshold >= 0. and bbox_threshold <= 1.
                diff = np.max(np.max(np.abs(non_repeat - bbox)[..., :-2], -1), -1)
                same = diff < bbox_threshold
                bbox_rev = bbox[::-1]
                diff_rev = np.max(np.max(np.abs(non_repeat - bbox_rev)[..., :-2], -1), -1)
                same_rev = diff_rev < bbox_threshold
                if same.sum() >= 1 or same_rev.sum() >= 1:
                    continue

                # （2D）和现有BBox重合了多少（蒙特卡洛方法）
                bbox_threshold = 0.5
                assert bbox_threshold >= 0. and bbox_threshold <= 1.
                sample_num_points = 10000
                b = bbox[:, -2:]
                x_min, y_min = b[1]
                x_max, y_max = b[0]
                xs = np.random.uniform(x_min, x_max, sample_num_points)
                ys = np.random.uniform(y_min, y_max, sample_num_points)
                sampled_points = np.stack([xs, ys], axis=1)
                x_in = (sampled_points[:, 0][:, None] >= non_repeat[:, 0, -2]) & (sampled_points[:, 0][:, None] <= non_repeat[:, 1, -2])
                y_in = (sampled_points[:, 1][:, None] >= non_repeat[:, 0, -1]) & (sampled_points[:, 1][:, None] <= non_repeat[:, 1, -1])
                in_mask = x_in & y_in
                overlap_persent = np.sum(np.any(in_mask, axis=1)) / sample_num_points
                if overlap_persent > bbox_threshold:
                    continue

                non_repeat = np.concatenate([non_repeat, bbox[np.newaxis, :, :]], 0)

            # bboxes = non_repeat.reshape(len(non_repeat), -1)
            bboxes = np.concatenate([non_repeat[:, :, :3].reshape(len(non_repeat), -1), non_repeat[:, :, 3:].reshape(len(non_repeat), -1)], axis=-1)
            surf_panel_mask = torch.zeros((1, len(bboxes))) == 1
            _surf_pos = torch.concat([torch.FloatTensor(bboxes), torch.zeros(max_surf - len(bboxes), 10)]).unsqueeze(0)
            _surf_panel_mask = torch.concat([surf_panel_mask, torch.zeros(1, max_surf - len(bboxes)) == 0], -1)
    else:
        _surf_panel_mask = torch.zeros((1, max_surf), dtype=torch.bool, device=device)

    n_surfs = torch.sum(~_surf_panel_mask)

    # [test] vis BBox deduped
    colors = [to_hex(plt.cm.coolwarm(i)) for i in np.linspace(0, 1, n_surfs)]
    _surf_pos_ = _surf_pos[:, :n_surfs]
    _surf_uv_bbox_wcs_ = np.zeros((_surf_pos_.shape[-2], 6))
    _surf_uv_bbox_wcs_[:, [0, 1, 3, 4]] = _surf_pos_[0, :, 6:].detach().cpu().numpy()
    fig = draw_bbox_geometry_3D2D(
        bboxes=[_surf_pos_[0, :, :6].detach().cpu().numpy(), _surf_uv_bbox_wcs_],
        bbox_colors=colors,
        title=f"{caption}",
        # output_fp=output_fp.replace('.pkl', '_pointcloud.png'),
        show_num=True,
        fig_show="browser"
    )

    # SurfZ ===
    _surf_z = randn_tensor((1, 32, latent_channels * latent_size * latent_size), device='cuda')
    ddpm_scheduler.set_timesteps(1000)

    with torch.no_grad():
        for t in tqdm(ddpm_scheduler.timesteps, desc="Surf-Z Denoising"):
            timesteps = t.reshape(-1).to('cuda')
            pred = surfz_model(_surf_z, timesteps, _surf_pos.to('cuda'),
                               _surf_panel_mask.to('cuda'), class_label=None, condition=text_embeding, is_train=False)
            _surf_z = ddpm_scheduler.step(pred, t, _surf_z).prev_sample
    surf_z = _surf_z[:, :n_surfs]

    # VAE Decoding ===
    with torch.no_grad():
        decoded_surf_pos = surf_vae_decoder(surf_z.view(-1, latent_channels, latent_size, latent_size))

    # [test] vis geoimg
    if vis:
        pred_img = make_grid(decoded_surf_pos, nrow=6, normalize=True, value_range=(-1, 1))
        fig, ax = plt.subplots(3, 1, figsize=(40, 40))
        ax[0].imshow(pred_img[:3, ...].permute(1, 2, 0).detach().cpu().numpy())
        ax[1].imshow(pred_img[3:, ...].permute(1, 2, 0).detach().cpu().numpy())
        ax[2].imshow(pred_img[-1:, ...].permute(1, 2, 0).detach().cpu().numpy())

        ax[0].set_title('Geometry Images')
        ax[1].set_title('UV Images')
        ax[2].set_title('Mask Images')

        plt.tight_layout()
        plt.axis('off')

        if output_fp:
            plt.savefig(output_fp.replace(".pkl", "_geo_img.png"), transparent=True, dpi=72)
        else:
            plt.show()
        plt.close()

    # plotly visualization
    colormap = plt.cm.coolwarm

    _surf_bbox = _surf_pos.squeeze(0)[~_surf_panel_mask.squeeze(0), :].detach().cpu().numpy()
    _decoded_surf_pos = decoded_surf_pos.permute(0, 2, 3, 1).detach().cpu().numpy()
    _surf_ncs_mask = _decoded_surf_pos[..., -1:].reshape(n_surfs, -1) > 0.0
    _surf_ncs = _decoded_surf_pos[..., :3].reshape(n_surfs, -1, 3)
    _surf_uv_ncs = _decoded_surf_pos[..., 3:5].reshape(n_surfs, -1, 2)

    _surf_uv_bbox = _surf_bbox[..., 6:]
    _surf_bbox = _surf_bbox[..., :6]

    if vis:
        colors = [to_hex(colormap(i)) for i in np.linspace(0, 1, n_surfs)]
        _surf_wcs_ = _denormalize_pts(_surf_ncs, _surf_bbox)
        # _surf_uv_wcs = _denormalize_pts(_surf_uv_ncs, _surf_uv_bbox)
        draw_bbox_geometry(  # [todo] 看
            bboxes=_surf_bbox,
            bbox_colors=colors,
            points=_surf_wcs_,
            point_masks=_surf_ncs_mask,
            point_colors=colors,
            num_point_samples=5000,
            title=caption,
            output_fp=output_fp.replace(".pkl", "_pointcloud.png")
            # show_num=True
        )

        _surf_uv_bbox_wcs_ = np.zeros((n_surfs, 6))
        _surf_uv_bbox_wcs_[:, [0, 1, 3, 4]] = _surf_uv_bbox
        _surf_uv_wcs_ = _denormalize_pts(_surf_uv_ncs, _surf_uv_bbox).reshape(n_surfs, -1, 2)
        _surf_uv_wcs_ = np.concatenate([_surf_uv_wcs_, np.zeros((n_surfs, _surf_uv_wcs_.shape[-2], 1), dtype=np.float32)], axis=-1)

        fig = draw_bbox_geometry_3D2D(
            bboxes=[_surf_bbox, _surf_uv_bbox_wcs_],
            bbox_colors=colors,
            points=[_surf_wcs_, _surf_uv_wcs_],
            point_masks=_surf_ncs_mask,
            point_colors=colors,
            num_point_samples=1000,
            title=f"{caption}",
            # output_fp=output_fp.replace('.pkl', '_pointcloud.png'),
            show_num=True,
            fig_show="browser"
        )

    _edited_mask = np.zeros((max_surf), dtype=np.bool)  # 用于标注哪些panel被编辑过了
    _edited_mask[:n_surfs] = True


    result = {
        'surf_bbox': _surf_bbox,        # (N, 6)
        'surf_uv_bbox': _surf_uv_bbox,  # (N, 4)
        'surf_ncs': _surf_ncs,          # (N, 256*256, 3)
        'surf_uv_ncs': _surf_uv_ncs,    # (N, 256*256, 2)
        'surf_mask': _surf_ncs_mask,    # (N, 256*256) => bool
        'caption': caption,             # str
        'edited_mask': _edited_mask,    # (max_surfs)
        "latent_code": surf_z           # (N, 64)
    }
    if output_fp:
        with open(output_fp, 'wb') as f: pickle.dump(result, f)
    print('[DONE] save to:', output_fp)

    return result


# [TODO] 输入data caption panel_mask , 进行局部编辑
def inference_one(models,
                  data,
                  caption='',
                  output_fp='',
                  dedup=True,
                  vis=False,):
    batch_size = 1
    device = 'cuda'
    max_surf = 32
    bbox_dim = 10
    RESO = 256

    # get data ===
    surf_ncs = data["surf_ncs"].reshape(-1, RESO, RESO, 3)
    surf_uv_ncs = data["surf_uv_ncs"].reshape(-1, RESO, RESO, 2)
    surf_mask = data["surf_mask"].reshape(-1, RESO, RESO, 1)
    surf_bbox_wcs = data["surf_bbox"]
    surf_uv_bbox_wcs = data["surf_uv_bbox"]
    n_surfs_orig = len(surf_ncs)

    # caption="dress, fited, cinched waist, stand collar, cap sleeves"
    # get models ===
    ddpm_scheduler = models['ddpm_scheduler']
    surfz_model = models['surfz_model']
    surfpos_model = models['surfpos_model']
    surf_vae_decoder = models['surf_vae_decoder']
    surf_vae_encoder = models['surf_vae_encoder']
    text_enc = models.get('text_enc', None)

    latent_channels = models['latent_channels']
    latent_size = models['latent_size']

    mask_type = input("choose mask type:\n"
                        "  1.  add part\n"
                        "  2.  change part\n")
    if "1" in mask_type:
        _surf_pos_mask = torch.ones((batch_size, max_surf, bbox_dim), dtype=torch.bool, device=device)
        _surf_pos_mask[0][n_surfs_orig:] = False
        _surf_z_mask = torch.ones((max_surf), dtype=torch.bool, device=device)
        _surf_z_mask[n_surfs_orig:] = False
    else:
        raise NotImplementedError

    # _surf_pos_mask[0,:] = False
    # _surf_z_mask[:] = False

    # text embedding
    text_embeding = text_enc(caption) \
        if caption is not None and text_enc is not None else None

    # SurfPos ===
    surf_pos_gt = torch.tensor(np.concatenate([surf_bbox_wcs, surf_uv_bbox_wcs], axis=-1), device=device)
    n_pads = max_surf-n_surfs_orig
    pad_idx = torch.arange(n_surfs_orig)
    _surf_pos_gt = torch.cat([
        surf_pos_gt[pad_idx, ...], torch.zeros((n_pads, *surf_pos_gt.shape[1:]), dtype=surf_pos_gt.dtype, device=surf_pos_gt.device)
    ], dim=0)[None, ...]

    # generate bbox by denoising
    _surf_pos = randn_tensor((batch_size, max_surf, bbox_dim)).to(device)
    ddpm_scheduler.set_timesteps(1000)

    with torch.no_grad():
        _surf_pos_gt_noise = torch.randn(_surf_pos.shape).to(device)
        for t in tqdm(ddpm_scheduler.timesteps, desc="Surf-Pos Denoising"):
            _surf_pos_gt_noised = ddpm_scheduler.add_noise(_surf_pos_gt, _surf_pos_gt_noise, t)
            _surf_pos[_surf_pos_mask] = _surf_pos_gt_noised[_surf_pos_mask]

            timesteps = t.reshape(-1).to(device)
            pred = surfpos_model(_surf_pos, timesteps, condition=text_embeding)
            _surf_pos = ddpm_scheduler.step(pred, t, _surf_pos).prev_sample

    _surf_pos[_surf_pos_mask] = _surf_pos_gt[_surf_pos_mask]

    # [test] vis BBox
    colors = [to_hex(plt.cm.coolwarm(i)) for i in np.linspace(0, 1, max_surf)]
    _surf_pos_ = _surf_pos[:, :]
    _surf_uv_bbox_wcs_ = np.zeros((_surf_pos_.shape[-2], 6))
    _surf_uv_bbox_wcs_[:,[0,1,3,4]] = _surf_pos_[0,:,6:].detach().cpu().numpy()
    fig = draw_bbox_geometry_3D2D(
        bboxes=[_surf_pos_[0,:,:6].detach().cpu().numpy(), _surf_uv_bbox_wcs_],
        bbox_colors=colors,
        title=f"{caption}",
        # output_fp=output_fp.replace('.pkl', '_pointcloud.png'),
        show_num=True,
        fig_show="browser"
    )

    if dedup:
        for ii in range(batch_size):
            # bboxes = np.round(_surf_pos[ii].unflatten(-1, torch.Size([2, 5])).detach().cpu().numpy(), 4)  # [...,-2:]  # mask维度
            bboxes = np.round(
                torch.concatenate(
                    [_surf_pos[ii][:, :6].unflatten(-1, torch.Size([2, 3])), _surf_pos[ii][:, 6:].unflatten(-1, torch.Size([2, 2]))]
                    , dim=-1).detach().cpu().numpy(), 4)
            non_repeat = bboxes[_surf_z_mask.detach().cpu().numpy()]

            # _surf_pos_mask
            for bbox_idx in range(0, len(bboxes)):
                if _surf_z_mask[bbox_idx]:
                    continue

                bbox = bboxes[bbox_idx]

                # （2D）判断BBox的大小是否非0
                bbox_threshold = 2e-6
                assert bbox_threshold >= 0. and bbox_threshold <= 1.
                v=1
                is_thin=False  # 判断是否是细长型（但面积太小）
                for h in (bbox[1]-bbox[0])[3:]:
                    v*=h
                    if h>0.02:
                        is_thin=True
                if v<bbox_threshold and not is_thin:
                    continue

                # （2D）和现有BBox是否重复
                bbox_threshold = 0.08
                assert bbox_threshold >= 0. and bbox_threshold <= 1.
                diff = np.max(np.max(np.abs(non_repeat - bbox)[..., -2:], -1), -1)
                same = diff < bbox_threshold
                bbox_rev = bbox[::-1]
                diff_rev = np.max(np.max(np.abs(non_repeat - bbox_rev)[..., -2:], -1), -1)
                same_rev = diff_rev < bbox_threshold
                if same.sum() >= 1 or same_rev.sum() >= 1:
                    continue

                # （3D）和现有BBox是否重复
                bbox_threshold = 0.08
                assert bbox_threshold >= 0. and bbox_threshold <= 1.
                diff = np.max(np.max(np.abs(non_repeat - bbox)[..., :-2], -1), -1)
                same = diff < bbox_threshold
                bbox_rev = bbox[::-1]
                diff_rev = np.max(np.max(np.abs(non_repeat - bbox_rev)[..., :-2], -1), -1)
                same_rev = diff_rev < bbox_threshold
                if same.sum() >= 1 or same_rev.sum() >= 1:
                    continue

                # （2D）和现有BBox重合了多少（蒙特卡洛方法）
                bbox_threshold = 0.5
                assert bbox_threshold>=0. and bbox_threshold<=1.
                sample_num_points = 10000
                b = bbox[:,-2:]
                x_min, y_min = b[1]
                x_max, y_max = b[0]
                xs = np.random.uniform(x_min, x_max, sample_num_points)
                ys = np.random.uniform(y_min, y_max, sample_num_points)
                sampled_points = np.stack([xs, ys], axis=1)
                x_in = (sampled_points[:, 0][:, None] >= non_repeat[:, 0, -2]) & (sampled_points[:, 0][:, None] <= non_repeat[:, 1, -2])
                y_in = (sampled_points[:, 1][:, None] >= non_repeat[:, 0, -1]) & (sampled_points[:, 1][:, None] <= non_repeat[:, 1, -1])
                in_mask = x_in & y_in
                overlap_persent = np.sum(np.any(in_mask, axis=1)) / sample_num_points
                if overlap_persent > bbox_threshold:
                    continue

                non_repeat = np.concatenate([non_repeat, bbox[np.newaxis, :, :]], 0)

            # bboxes = non_repeat.reshape(len(non_repeat), -1)
            bboxes = np.concatenate([non_repeat[:, :, :3].reshape(len(non_repeat), -1), non_repeat[:, :, 3:].reshape(len(non_repeat), -1)], axis=-1)
            surf_panel_mask = torch.zeros((1, len(bboxes))) == 1
            _surf_pos = torch.concat([torch.FloatTensor(bboxes), torch.zeros(max_surf - len(bboxes), 10)]).unsqueeze(0)
            _surf_panel_mask = torch.concat([surf_panel_mask, torch.zeros(1, max_surf - len(bboxes)) == 0], -1)
    else:
        _surf_panel_mask = torch.zeros((1, max_surf), dtype=torch.bool, device=device)

    n_surfs = torch.sum(~_surf_panel_mask)
    # [test] vis BBox
    colors = [to_hex(plt.cm.coolwarm(i)) for i in np.linspace(0, 1, n_surfs)]
    _surf_pos_ = _surf_pos[:, :n_surfs]
    _surf_uv_bbox_wcs_ = np.zeros((_surf_pos_.shape[-2], 6))
    _surf_uv_bbox_wcs_[:,[0,1,3,4]] = _surf_pos_[0,:,6:].detach().cpu().numpy()
    fig = draw_bbox_geometry_3D2D(
        bboxes=[_surf_pos_[0,:,:6].detach().cpu().numpy(), _surf_uv_bbox_wcs_],
        bbox_colors=colors,
        title=f"{caption}",
        # output_fp=output_fp.replace('.pkl', '_pointcloud.png'),
        show_num=True,
        fig_show="browser"
    )

    # SurfZ ===
    surf_gt = torch.tensor(np.concatenate([surf_ncs, surf_uv_ncs, surf_mask * 2. - 1], axis=-1), device=device, dtype=torch.float32)
    # pad zero
    _surf_gt = torch.cat([
        surf_gt[pad_idx, ...], torch.zeros((n_pads, *surf_gt.shape[1:]), dtype=surf_gt.dtype, device=surf_gt.device)
    ], dim=0)[None, ...]

    with torch.no_grad():
        _surf_z_gt = (surf_vae_encoder(_surf_gt[0].permute(0, 3, 1, 2))
                      ).reshape(1, 32, latent_channels * latent_size * latent_size)

    _surf_z = randn_tensor((1, 32, latent_channels * latent_size * latent_size), device='cuda')
    ddpm_scheduler.set_timesteps(1000)

    with torch.no_grad():
        for t in tqdm(ddpm_scheduler.timesteps, desc="Surf-Z Denoising"):
            _surf_z_gt_noise = torch.randn(_surf_z.shape).to(device)
            _surf_z_gt_noised = ddpm_scheduler.add_noise(_surf_z_gt, _surf_z_gt_noise, t)

            _surf_z[0][_surf_z_mask] = _surf_z_gt_noised[0][_surf_z_mask]

            timesteps = t.reshape(-1).to('cuda')
            pred = surfz_model(_surf_z, timesteps, _surf_pos.to('cuda'),
                               _surf_panel_mask.to('cuda'), class_label=None, condition=text_embeding, is_train=False)
            _surf_z = ddpm_scheduler.step(pred, t, _surf_z).prev_sample
    _surf_z[0][_surf_z_mask] = _surf_z_gt[0][_surf_z_mask]
    surf_z = _surf_z[:,:n_surfs]


    # VAE Decoding ===
    with torch.no_grad(): decoded_surf_pos = surf_vae_decoder(surf_z.view(-1, latent_channels, latent_size, latent_size))
    decoded_surf_pos[_surf_z_mask[:n_surfs]] = torch.tensor(np.concatenate([surf_ncs,surf_uv_ncs,surf_mask],axis=-1).transpose(0,3,1,2),device=device)[_surf_z_mask[:n_surfs_orig]]
    # [test] vis geoimg
    if vis:
        pred_img = make_grid(decoded_surf_pos, nrow=6, normalize=True, value_range=(-1,1))
        fig, ax = plt.subplots(3, 1, figsize=(40, 40))
        ax[0].imshow(pred_img[:3, ...].permute(1, 2, 0).detach().cpu().numpy())
        ax[1].imshow(pred_img[3:, ...].permute(1, 2, 0).detach().cpu().numpy())
        ax[2].imshow(pred_img[-1:, ...].permute(1, 2, 0).detach().cpu().numpy())

        ax[0].set_title('Geometry Images')
        ax[1].set_title('UV Images')
        ax[2].set_title('Mask Images')

        plt.tight_layout()
        plt.axis('off')

        if output_fp:
            plt.savefig(output_fp.replace(".pkl", "_geo_img.png"), transparent=True, dpi=72)
        else:
            plt.show()
        plt.close()

    # plotly visualization
    colormap = plt.cm.coolwarm

    _surf_bbox = _surf_pos.squeeze(0)[~_surf_panel_mask.squeeze(0), :].detach().cpu().numpy()
    _decoded_surf_pos = decoded_surf_pos.permute(0, 2, 3, 1).detach().cpu().numpy()
    _surf_ncs_mask = _decoded_surf_pos[..., -1:].reshape(n_surfs, -1) > 0.0
    _surf_ncs = _decoded_surf_pos[..., :3].reshape(n_surfs, -1, 3)
    _surf_uv_ncs = _decoded_surf_pos[..., 3:5].reshape(n_surfs, -1, 2)

    _surf_uv_bbox = _surf_bbox[..., 6:]
    _surf_bbox = _surf_bbox[..., :6]

    if vis:
        colors = [to_hex(colormap(i)) for i in np.linspace(0, 1, n_surfs)]
        _surf_wcs_ = _denormalize_pts(_surf_ncs, _surf_bbox)
        # _surf_uv_wcs = _denormalize_pts(_surf_uv_ncs, _surf_uv_bbox)
        draw_bbox_geometry( # [todo] 看
            bboxes = _surf_bbox,
            bbox_colors = colors,
            points = _surf_wcs_,
            point_masks = _surf_ncs_mask,
            point_colors = colors,
            num_point_samples = 5000,
            title = caption,
            output_fp=output_fp.replace(".pkl", "_pointcloud.png")
            # show_num=True
        )

        _surf_uv_bbox_wcs_ = np.zeros((n_surfs, 6))
        _surf_uv_bbox_wcs_[:, [0, 1, 3, 4]] = _surf_uv_bbox
        _surf_uv_wcs_ = _denormalize_pts(_surf_uv_ncs, _surf_uv_bbox).reshape(n_surfs, -1, 2)
        _surf_uv_wcs_ = np.concatenate([_surf_uv_wcs_, np.zeros((n_surfs,_surf_uv_wcs_.shape[-2], 1), dtype=np.float32)],axis=-1)

        fig = draw_bbox_geometry_3D2D(
            bboxes=[_surf_bbox, _surf_uv_bbox_wcs_],
            bbox_colors=colors,
            points=[_surf_wcs_, _surf_uv_wcs_],
            point_masks=_surf_ncs_mask,
            point_colors=colors,
            num_point_samples=1000,
            title=f"{caption}",
            # output_fp=output_fp.replace('.pkl', '_pointcloud.png'),
            show_num=True,
            fig_show="browser"
        )

    if "1" in mask_type:
        _edited_mask = np.ones((max_surf), dtype=np.bool)  # 用于标注哪些panel被编辑过了
        _edited_mask[:n_surfs_orig] = False
        _edited_mask[n_surfs:] = False
    else:
        raise NotImplementedError


    result = {
        'surf_bbox': _surf_bbox,        # (N, 6)
        'surf_uv_bbox': _surf_uv_bbox,  # (N, 4)
        'surf_ncs': _surf_ncs,          # (N, 256*256, 3)
        'surf_uv_ncs': _surf_uv_ncs,    # (N, 256*256, 2)
        'surf_mask': _surf_ncs_mask,    # (N, 256*256) => bool
        'caption': caption,             # str
        'edited_mask': _edited_mask,    # (max_surfs)
        "latent_code": surf_z           # (N, 64)
    }

    if output_fp:
        with open(output_fp, 'wb') as f: pickle.dump(result, f)
    print('[DONE] save to:', output_fp)

    return result


def run(args):
    """
    Operations：
        1.完全重新生成一遍
        2.更换caption后，保留原有panels，添加新的panels
        #r 生成结果不满意，退回上一步
        #n 切换到下一条caption
        #cc 手动编辑当前提示词

    使用方法：
        使用1重复生成得到比较好的初始服装
        循环每个提示词：
            使用#n切换提示词
            循环至满意：
                使用2进行impainting
                如果满意
                    #n break（切换至下一个提示词 ）
                如果不满意：
                    #r 回到上一状态

    """
    models = init_models(args)
    os.makedirs(args.output_root, exist_ok=True)
    # caption_editing_list_dir = os.path.join(args.data_root, 'caption_editing_list')
    # list_fp_list = sorted(glob(os.path.join(caption_editing_list_dir, '*.txt')))
    # for list_fp in list_fp_list:
    list_fp = "/home/Ex1/ProjectFiles/Pycharm_MyPaperWork/style3d_gen/_LSR/experiment/editing/demo_caption_editing/data/caption_editing_list/000_dress, sleeveless, round-neck.txt"

    with open(list_fp, 'r', encoding='utf-8') as f:
        caption_list = [line.strip() for line in f]

    data = None
    history = []
    for idx, caption in tqdm(enumerate(caption_list)):
        # fp
        if idx == 0:
            number = len(os.listdir(args.output_root)) if os.path.exists(args.output_root) else 0
            output_root = os.path.join(args.output_root, f"{number}".zfill(3)+f"_{caption}")
            output_dir = os.path.join(output_root, "generated")
        else:
            output_dir = os.path.join(output_root, "generated")
        output_fp = os.path.join(output_dir, f"{idx}".zfill(3)+f"_{caption}.pkl")
        os.makedirs(output_dir, exist_ok=True)

        while True:
            print(f"Current caption: {caption}")
            user_input = input("Operates:\n"
                               "  1:   regenerate\n"
                               "  2:   generate with mask\n"
                               "  cc:  change caption\n")

            # [TODO] 保存去噪过程

            # regenerate
            if "1" in user_input:
                if data is not None:
                    history.append(data)
                data = init_inference(models, caption, output_fp=output_fp, vis=True)
            # editing with mask
            elif "2" in user_input:
                if data is None:
                    data = init_inference(models, caption, output_fp=output_fp, vis=True)
                else:
                    history.append(data)
                    data = inference_one(models, data, caption, output_fp=output_fp, vis=True)

            elif "cc" in user_input:
                new_caption = input("Input new caption:\n")
                caption = new_caption
            elif "#s" in user_input:
                raise NotImplementedError
            elif "#n" in user_input:
                break
            elif "#r" in user_input:
                if len(history) > 0:
                    data = history[-1]
                    del history[-1]
            else:
                continue

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--exp', type=str,
        default='caption',  # bbox_seg    bbox_compose    caption
        help='Experiment type')

    parser.add_argument(
        '--text_encoder', type=str,
        default='CLIP',
        help='Path to SurfZ model')

    parser.add_argument(
        '--data_root', type=str,
        default='_LSR/experiment/editing/demo_caption_editing/data',
        help='Path to cache file')

    parser.add_argument(
        '--output_root', type=str, default='_LSR/experiment/editing/demo_caption_editing/output',
        help='Path to output directory')

    parser.add_argument('--block_dims', nargs='+', type=int, default=[16,32,32,64,64,128], help='Latent dimension of each block of the UNet model.')
    parser.add_argument('--latent_channels', type=int, default=1, help='Latent channels of the vae model.')
    parser.add_argument('--reso', type=int, default=256, help='Sample size of the vae model.')

    parser.add_argument('--num_samples', type=int, default=-1, help='Number of samples to inference.')

    args = parser.parse_args()

    run(args)
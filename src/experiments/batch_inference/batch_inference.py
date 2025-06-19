import os
import random
from glob import glob
import argparse
import pickle
import numpy as np

from tqdm import tqdm

import torch
from torchvision.utils import make_grid

from matplotlib import pyplot as plt
from matplotlib.colors import to_hex

from src.network import AutoencoderKLFastDecode, SurfZNet, SurfPosNet, TextEncoder, PointcloudEncoder, SketchEncoder
from diffusers import DDPMScheduler  # , PNDMScheduler
from src.utils import randn_tensor
from src.vis import draw_bbox_geometry, draw_bbox_geometry_3D2D, get_visualization_steps

import plotly.graph_objects as go


# 可视化作为 condition 的 PointCloud
def pointcloud_condition_visualize(vertices: np.ndarray, output_fp=None):
    assert vertices.ndim == 2 and vertices.shape[1] == 3, "vertices 应为 (N, 3) 的 numpy 数组"

    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    color = "#717388"  # 用 z 来着色
    xrange = x.max() - x.min()
    yrange = y.max() - y.min()
    zrange = z.max() - z.min()
    fig = go.Figure(data=[
        go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=2,
                color=color,
                colorscale='Viridis',
                opacity=1,
                showscale=False  # 不显示 colorbar
            ),
            showlegend=False  # 不显示图例
        )
    ])

    # 隐藏坐标轴、网格、背景等
    axis_style = dict(
        showbackground=False,
        showgrid=False,
        zeroline=False,
        showline=False,
        ticks='',
        showticklabels=False,
        visible=False  # 最直接隐藏整个轴
    )
    camera = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0, y=0, z=2)
    )
    fig.update_layout(
        scene=dict(
            xaxis=axis_style,
            yaxis=axis_style,
            zaxis=axis_style,
            aspectmode='manual',
            aspectratio=dict(
                x=xrange,
                y=yrange,
                z=zrange
            )
        ),
        scene_camera=camera,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    RESO = 800
    if output_fp:
        # fig.write_html(output_fp.replace(".pkl", "") + "_pcCond_vis.html")
        fig.write_image(output_fp.replace(".pkl", "_pcCondOrig.png"), width=RESO, height=RESO, scale=2.5)


def _denormalize_pts(pts, bbox):
    pos_dim =  pts.shape[-1]
    bbox_min = bbox[..., :pos_dim][:, None, ...]
    bbox_max = bbox[..., pos_dim:][:, None, ...]
    bbox_scale = np.max(bbox_max - bbox_min, axis=-1, keepdims=True) * 0.5
    bbox_offset = (bbox_max + bbox_min) / 2.0
    return pts * bbox_scale + bbox_offset


def init_models(args):

    device = args.device

    block_dims = args.block_dims
    sample_size = args.reso
    latent_channels = args.latent_channels
    latent_size = sample_size//(2**(len(block_dims)-1))

    surf_vae = AutoencoderKLFastDecode( in_channels=6,
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
    surf_vae.load_state_dict(torch.load(args.vae), strict=False)
    surf_vae.to(device).eval()

    ddpm_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule='linear',
        prediction_type='epsilon',
        beta_start=0.0001,
        beta_end=0.02,
        clip_sample=False,
    )

    # Load conditioning model
    if args.text_encoder is not None: text_enc = TextEncoder(args.text_encoder, device=device)
    else: text_enc = None
    if args.pointcloud_encoder is not None: pointcloud_enc = PointcloudEncoder(args.pointcloud_encoder, device=device)
    else: pointcloud_enc = None
    if args.sketch_encoder is not None: sketch_enc = SketchEncoder(args.sketch_encoder, device="cpu")
    else: sketch_enc = None
    # condition dimention
    if args.text_encoder is not None:
        condition_dim = text_enc.text_emb_dim
    elif args.pointcloud_encoder is not None:
        condition_dim = pointcloud_enc.pointcloud_emb_dim
    elif args.sketch_encoder is not None:
        condition_dim = sketch_enc.sketch_emb_dim
    else:
        condition_dim = -1

    # Load SurfPos Net
    surfpos_model = SurfPosNet(
        p_dim=10,
        condition_dim=condition_dim,
        num_cf=-1
    )
    surfpos_model.load_state_dict(torch.load(args.surfpos)['model_state_dict'])
    surfpos_model.to(device).eval()
    # Load SurfZ Net
    surfz_model = SurfZNet(
        p_dim=10,
        z_dim=latent_size**2*latent_channels,
        num_heads=12,
        condition_dim=condition_dim,
        num_cf=-1
        )
    surfz_model.load_state_dict(torch.load(args.surfz)['model_state_dict'])
    surfz_model.to(device).eval()

    print('[DONE] Models initialized.')

    return {
        'surf_vae': surf_vae,
        'ddpm_scheduler': ddpm_scheduler,
        'surfpos_model': surfpos_model,
        'surfz_model': surfz_model,
        'text_enc': text_enc,
        'pointcloud_enc': pointcloud_enc,
        'sketch_enc': sketch_enc,
        'latent_channels': latent_channels,
        'latent_size': latent_size
    }


def inference_one(
        models,
        args,
        caption='',
        pointcloud_feature=None,
        sampled_pc_cond=None,
        sketch_features=None,
        surf_pos_orig=None,
        output_fp='',
        dedup=True,
        vis=False,
        data_fp=None,
        data_id_trainval=None,
        save_denoising=False,
):
    batch_size = 1
    max_surf = 32
    device = args.device


    if save_denoising:
        assert surf_pos_orig is None and dedup
        denoising_dict = {}
        vis_steps = get_visualization_steps()
        denoising_dict["denoising_data"] = {k:{} for k in vis_steps}
        denoising_dict["dedup_mask"] = np.zeros((max_surf), dtype=np.bool)  # 记录哪些BBox (为True的部分) 是被dedup掉的
    else:
        denoising_dict = None

    ddpm_scheduler = models['ddpm_scheduler']
    surfz_model = models['surfz_model']
    surfpos_model = models['surfpos_model']
    surf_vae = models['surf_vae']
    text_enc = models['text_enc']

    latent_channels = models['latent_channels']
    latent_size = models['latent_size']

    # get condition embedding
    if args.text_encoder is not None:
        condition_emb = text_enc(caption)
    elif args.pointcloud_encoder is not None:
        condition_emb = pointcloud_feature.reshape(1,-1)
    elif args.sketch_encoder is not None:
        condition_emb = sketch_features.reshape(1,-1)
    else:
        condition_emb = None
    if condition_emb is not None:
        condition_emb = condition_emb.to(device)

    # GET BBOX ---------------------------------------------------------------
    # 如果不使用原有的BBox
    if surf_pos_orig is None:
        # SurfPos Denoising
        surfPos = randn_tensor((batch_size, max_surf, 10)).to(device)
        ddpm_scheduler.set_timesteps(1000)
        with torch.no_grad():
            for t in tqdm(ddpm_scheduler.timesteps, desc="Surf-Pos Denoising"):
                timesteps = t.reshape(-1).to(device)
                pred = surfpos_model(surfPos, timesteps, condition=condition_emb)
                surfPos = ddpm_scheduler.step(pred, t, surfPos).prev_sample

                # 保存去噪过程
                if save_denoising:
                    if t.item() in denoising_dict["denoising_data"].keys():
                        denoising_dict["denoising_data"][t.item()]["surf_pos"] = surfPos.squeeze(0).detach().cpu().numpy()

        # # [test] vis BBox
        # colors = [to_hex(plt.cm.coolwarm(i)) for i in np.linspace(0, 1, max_surf)]
        # _surf_pos_ = surfPos[:, :]
        # _surf_uv_bbox_wcs_ = np.zeros((_surf_pos_.shape[-2], 6))
        # _surf_uv_bbox_wcs_[:, [0, 1, 3, 4]] = _surf_pos_[0, :, 6:].detach().cpu().numpy()
        # fig = draw_bbox_geometry_3D2D(
        #     bboxes=[_surf_pos_[0, :, :6].detach().cpu().numpy(), _surf_uv_bbox_wcs_],
        #     bbox_colors=colors,
        #     title=f"{caption}",
        #     # output_fp=output_fp.replace('.pkl', '_pointcloud.png'),
        #     show_num=True,
        #     fig_show="browser"
        # )

        if dedup:
            if args.padding == "repeat":
                bbox_threshold = 0.08
            elif args.padding == "zero":
                bbox_threshold = 2e-4

            _surf_pos = surfPos

            bboxes = torch.concatenate(
                    [_surf_pos[0][:, :6].unflatten(-1, torch.Size([2, 3])), _surf_pos[0][:, 6:].unflatten(-1, torch.Size([2, 2]))]
                        , dim=-1).detach().cpu().numpy()

            non_repeat = None
            for bbox_idx, bbox in enumerate(bboxes):
                if args.padding=="repeat":
                    if non_repeat is None:
                        non_repeat = bbox[np.newaxis, :, :]
                    else:
                        diff = np.max(np.max(np.abs(non_repeat - bbox)[..., -2:], -1), -1)  #
                        same = diff < bbox_threshold
                        bbox_rev = bbox[::-1]  # also test reverse bbox for matching
                        diff_rev = np.max(np.max(np.abs(non_repeat - bbox_rev)[..., -2:], -1), -1)  # [...,-2:]
                        same_rev = diff_rev < bbox_threshold
                        if same.sum() >= 1 or same_rev.sum() >= 1:
                            is_deduped = True # 当前BBox是否被去重了
                        else:
                            is_deduped = False
                            non_repeat = np.concatenate([non_repeat, bbox[np.newaxis, :, :]], 0)

                if args.padding=="zero":
                    # （2D）判断BBox的大小是否非0
                    v = 1
                    for h in (bbox[1] - bbox[0])[3:]:
                        v *= h
                    if v < bbox_threshold:
                        is_deduped = True
                    else:
                        is_deduped = False
                        if non_repeat is None:
                            non_repeat = bbox[np.newaxis, :, :]
                        else:
                            non_repeat = np.concatenate([non_repeat, bbox[np.newaxis, :, :]], 0)

                if save_denoising:
                    denoising_dict["dedup_mask"][bbox_idx] = is_deduped

            bboxes = np.concatenate([non_repeat[:, :, :3].reshape(len(non_repeat), -1), non_repeat[:, :, 3:].reshape(len(non_repeat), -1)], axis=-1)
            surf_mask = torch.zeros((1, len(bboxes))) == 1
            _surf_pos = torch.concat([torch.FloatTensor(bboxes), torch.zeros(max_surf - len(bboxes), 10)]).unsqueeze(0)
            _surf_mask = torch.concat([surf_mask, torch.zeros(1, max_surf - len(bboxes)) == 0], -1)
            n_surfs = torch.sum(~_surf_mask)

            if save_denoising:
                """
                将同样的去重应用到去噪过程中保存的每一步中
                然后保存
                """
                assert n_surfs == sum(~denoising_dict["dedup_mask"])
                for step in denoising_dict["denoising_data"].keys():
                    denoising_dict["denoising_data"][step]["surf_pos"] = (
                        denoising_dict["denoising_data"][step]["surf_pos"][~denoising_dict["dedup_mask"]])
                del denoising_dict['dedup_mask']

    # 如果使用原有的BBox
    else:
        n_surfs = torch.tensor(len(surf_pos_orig))
        surf_mask = torch.zeros((1, n_surfs)) == 1
        _surf_pos = torch.concat([surf_pos_orig, torch.zeros((max_surf - n_surfs, 10))]).to(device)
        _surf_mask = torch.concat([surf_mask, torch.zeros(1, max_surf - n_surfs) == 0], -1).to(device)

    # SurfZ Denoising
    _surf_z = randn_tensor((1, 32, latent_channels * latent_size * latent_size), device=device)
    ddpm_scheduler.set_timesteps(1000)
    with torch.no_grad():
        for t in tqdm(ddpm_scheduler.timesteps, desc="Surf-Z Denoising"):
            timesteps = t.reshape(-1).to(device)
            pred = surfz_model(
                _surf_z, timesteps, _surf_pos.to(device), _surf_mask.to(device), condition_emb, is_train=False)
            _surf_z = ddpm_scheduler.step(pred, t, _surf_z).prev_sample

            # 保存去噪过程
            if save_denoising:
                if t.item() in denoising_dict["denoising_data"].keys():
                    denoising_dict["denoising_data"][t.item()]["surf_z_latent"] = _surf_z[0,:n_surfs].detach().cpu().numpy()

    _surf_z = _surf_z[:,:n_surfs]

    # VAE Decoding
    with torch.no_grad(): decoded_surf_pos = surf_vae(_surf_z.view(-1, latent_channels, latent_size, latent_size))
    pred_img = make_grid(decoded_surf_pos, nrow=6, normalize=True, value_range=(-1,1))

    if vis:
        fig, ax = plt.subplots(3, 1, figsize=(40, 40))
        ax[0].imshow(pred_img[:3, ...].permute(1, 2, 0).detach().cpu().numpy())
        ax[1].imshow(pred_img[3:, ...].permute(1, 2, 0).detach().cpu().numpy())
        ax[2].imshow(pred_img[-1:, ...].permute(1, 2, 0).detach().cpu().numpy())

        ax[0].set_title('Geometry Images')
        ax[1].set_title('UV Images')
        ax[2].set_title('Mask Images')

        plt.tight_layout()
        plt.axis('off')

        if output_fp: plt.savefig(output_fp.replace('.pkl', '_geo_img.png'), transparent=True, dpi=72)
        else: plt.show()
        plt.close()

    # plotly visualization
    colormap = plt.cm.coolwarm

    _surf_bbox = _surf_pos.squeeze(0)[~_surf_mask.squeeze(0), :].detach().cpu().numpy()
    _decoded_surf = decoded_surf_pos.permute(0, 2, 3, 1).detach().cpu().numpy()
    _surf_ncs = _decoded_surf[..., :3].reshape(n_surfs, -1, 3)
    _surf_uv_ncs = _decoded_surf[..., 3:5].reshape(n_surfs, -1, 2)
    _surf_ncs_mask = _decoded_surf[..., -1:].reshape(n_surfs, -1) > 0.0

    _surf_uv_bbox = _surf_bbox[..., 6:]
    _surf_bbox = _surf_bbox[..., :6]

    if vis:
        colors = [to_hex(colormap(i)) for i in np.linspace(0, 1, n_surfs)]
        _surf_wcs = _denormalize_pts(_surf_ncs, _surf_bbox)
        # _surf_uv_wcs = _denormalize_pts(_surf_uv_ncs, _surf_uv_bbox)
        draw_bbox_geometry(
            bboxes = _surf_bbox,
            bbox_colors = colors,
            points = _surf_wcs,
            point_masks = _surf_ncs_mask,
            point_colors = colors,
            num_point_samples = 5000,
            title = caption,
            output_fp = output_fp.replace('.pkl', '_pointcloud.png'),
            # show_num=True
            )

    # 如果没有原本的路径data_fp, 尝试根据 data_id 找回 data_fp
    if data_fp is None and data_id_trainval is not None:
        try:
            data_id_tr = os.path.basename(data_id_trainval)
            data_id_tr = os.path.join("/data/AIGP/brep_reso_256_edge_snap_with_caption/", data_id_tr)
            data_fp = pickle.load(open(data_id_tr, "rb"))["data_fp"]
        except FileNotFoundError:
            try:
                data_id_tr = os.path.basename(data_id_trainval)
                data_id_tr = os.path.join("/data/AIGP/Q4/", data_id_tr)
                data_fp = pickle.load(open(data_id_tr, "rb"))["data_fp"]
            except FileNotFoundError:
                data_fp = None
                print("Without data_fp")

    result = {
        'surf_bbox': _surf_bbox,        # (N, 6)
        'surf_uv_bbox': _surf_uv_bbox,  # (N, 4)
        'surf_ncs': _surf_ncs,          # (N, 256*256, 3)
        'surf_uv_ncs': _surf_uv_ncs,    # (N, 256*256, 2)
        'surf_mask': _surf_ncs_mask,    # (N, 256*256) => bool
        'caption': caption,             # str
        'data_fp': data_fp,
        'denoising': denoising_dict,
        'args': vars(args)
    }

    # 如果点云condition，渲染个点云图
    if args.pointcloud_encoder is not None:
        pointcloud_condition_visualize(sampled_pc_cond, output_fp)
        result["sampled_pc_cond"] = sampled_pc_cond

    if output_fp:
        with open(output_fp, 'wb') as f: pickle.dump(result, f)

    torch.cuda.empty_cache()
    print('[DONE] save to:', output_fp)


def run(args):
    # 保证最多只有一种 condition embedding
    not_none_count = sum(x is not None for x in [args.text_encoder, args.pointcloud_encoder, args.sketch_encoder])
    assert not_none_count in (0, 1)

    models = init_models(args)

    os.makedirs(args.output, exist_ok=True)

    with open(args.cache, 'rb') as f: data_cache = pickle.load(f)

    if args.num_samples == -1: sample_idxs = range(len(data_cache['item_idx']))
    else: sample_idxs = random.sample(range(len(data_cache['item_idx'])), k=args.num_samples)

    for sample_data_idx in tqdm(sample_idxs):
        surf_pos_orig = None
        if args.use_original_pos:
            start_idx, end_idx = data_cache['item_idx'][sample_data_idx]
            surf_pos_orig = data_cache['surf_pos'][start_idx:end_idx]
            # surf_cls = data_cache['surf_cls'][start_idx:end_idx].to('cuda')

        # data_cache 中通常有caption，可以存下来
        if "caption" in data_cache:
            caption = data_cache['caption'][sample_data_idx]
        else:
            print("No caption in cache.")
            caption = None
        if args.pointcloud_encoder is not None:
            pointcloud_features = data_cache["pointcloud_feature"][sample_data_idx]
            sampled_pc_cond = data_cache["sampled_pc_cond"][sample_data_idx]
        else:
            pointcloud_features = None
            sampled_pc_cond = None

        if args.sketch_encoder is not None:
            sketch_features = data_cache["sketch_feature"][sample_data_idx]
        else:
            sketch_features = None

        output_fp = os.path.join(args.output, f'{sample_data_idx:04d}.pkl')

        data_fp = data_cache.get('data_fp', None)
        if data_fp is not None:
            data_fp = data_fp[sample_data_idx]
        data_id_trainval = data_cache.get('data_id', None)
        if data_id_trainval is not None:
            data_id_trainval = data_id_trainval[sample_data_idx]

        inference_one(
            models,
            args=args,
            caption = caption,
            pointcloud_feature=pointcloud_features,
            sampled_pc_cond = sampled_pc_cond,
            sketch_features=sketch_features,
            surf_pos_orig = surf_pos_orig,
            dedup=True,
            output_fp = output_fp,
            vis=True,
            data_fp=data_fp,
            data_id_trainval=data_id_trainval,
            save_denoising=args.save_denoising,
        )

    print('[DONE]')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # path configuration
    parser.add_argument(
        '--vae', type=str,
        default='/data/lsr/models/style3d_gen/surf_vae/stylexd_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e550.pt',
        help='Path to VAE model')
    parser.add_argument(
        '--surfpos', type=str,
        default='/data/lsr/models/style3d_gen/surf_pos/stylexd_surfpos_xyzuv_pad_repeat_uncond/ckpts/surfpos_e3000.pt',
        help='Path to SurfZ model')
    parser.add_argument(
        '--surfz', type=str,
        default='/data/lsr/models/style3d_gen/surf_z/stylexd_surfz_xyzuv_mask_latent1_mode/ckpts/surfz_e230000.pt',
        help='Path to SurfZ model')
    parser.add_argument(
        '--cache', type=str,
        default='/data/lsr/models/style3d_gen/surf_vae/stylexd_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e550/encoder_mode/surfpos_validate.pkl',
        help='Path to cache file')
    parser.add_argument('--use_original_pos',
                        action='store_true', help='If using pos information in training/validating data.')

    parser.add_argument(
        '--output', type=str, default='generated/surfz_e230000',
        help='Path to output directory')
    parser.add_argument('--save_denoising', action="store_true", help='Save garmage during denoising.')

    parser.add_argument('--block_dims', nargs='+', type=int, default=[16,32,32,64,64,128], help='Latent dimension of each block of the UNet model.')
    parser.add_argument('--latent_channels', type=int, default=1, help='Latent channels of the vae model.')
    parser.add_argument('--reso', type=int, default=256, help='Sample size of the vae model.')
    parser.add_argument("--padding", type=str, default="repeat", choices=["repeat", "zero"], help='Padding type during surfPos training.')
    parser.add_argument('--num_samples', type=int, default=-1, help='Number of samples to inference.')

    parser.add_argument('--text_encoder', type=str, default=None, choices=[None, 'CLIP', 'T5', 'GME'], help='Text encoder type.')
    parser.add_argument('--pointcloud_encoder', type=str, default=None, choices=[None, 'POINT_E'], help='Pointcloud encoder type.')
    parser.add_argument('--sketch_encoder', type=str, default=None, choices=[None, 'LAION2B', "RADIO_V2.5-G"], help='Sketch encoder type.')

    parser.add_argument('--device', type=str, default="cuda", help='')

    args = parser.parse_args()
    run(args)
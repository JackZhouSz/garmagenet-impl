import os
import yaml
import argparse

import numpy as np
import torch
from torchvision.utils import make_grid

from tqdm import tqdm
from network import *
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt

from diffusers import DDPMScheduler, PNDMScheduler

from utils import randn_tensor
from vis import draw_bbox_geometry
from matplotlib.colors import to_hex

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


import pickle

text2int = {'uncond': 0,
            'class1': 1,
            'bed': 2,
            'bench': 3,
            'bookshelf': 4,
            'cabinet': 5,
            'chair': 6,
            'couch': 7,
            'lamp': 8,
            'sofa': 9,
            'bathtub': 10
            }

    
def sample(eval_args, out_dir, vis=True, dedup=True):
    torch.cuda.empty_cache()

    # Inference configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")
    batch_size = eval_args['batch_size']
    bbox_threshold = eval_args['bbox_threshold']
    save_folder = eval_args['save_folder']
    num_surfaces = eval_args['num_surfaces']
    block_dims = eval_args['block_dims']
    z_scale = eval_args['z_scale']
    num_classes = eval_args['num_classes']

    if eval_args['use_cf']:
        class_label = torch.LongTensor([text2int[eval_args['class_label']]] * batch_size + \
                                       [text2int['uncond']] * batch_size).to(device).reshape(-1, 1)
        w = 0.6
    else:
        class_label = None

    if not os.path.exists(save_folder): os.makedirs(save_folder)

    surfPos_model = SurfPosNet(p_dim=10, num_cf=11)
    surfPos_model.load_state_dict(torch.load(eval_args['surfpos_weight'])['model_state_dict'])
    surfPos_model = surfPos_model.to(device).eval()
    print('[DONE] Load SurfPosNet model.')

    surfZ_model = SurfZNet(p_dim=10, z_dim=8*8*8, num_cf=-1)
    state_dict = torch.load(eval_args['surfz_weight'])['model']
    surfZ_model.load_state_dict(state_dict)
    surfZ_model = surfZ_model.to(device).eval()
    print('[DONE] Load SurfZ model.')
    

    surf_vae = AutoencoderKLFastDecode(in_channels=6,
                                       out_channels=6,
                                       down_block_types=['DownEncoderBlock2D']*len(block_dims),
                                       up_block_types=['UpDecoderBlock2D']*len(block_dims),
                                       block_out_channels=block_dims,
                                       layers_per_block=2,
                                       act_fn='silu',
                                       latent_channels=8,
                                       norm_num_groups=8,
                                       sample_size=256
                                       )
        
    surf_vae.load_state_dict(torch.load(eval_args['surfvae_weight']), strict=False)
    # surf_vae = nn.DataParallel(surf_vae)  # distributed inference
    surf_vae = surf_vae.to(device).eval()
    print('[DONE] Load SurfVAE model.')

    pndm_scheduler = PNDMScheduler(
        num_train_timesteps=1000,
        beta_schedule='linear',
        prediction_type='epsilon',
        beta_start=0.0001,
        beta_end=0.02,
    )

    ddpm_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule='linear',
        prediction_type='epsilon',
        beta_start=0.0001,
        beta_end=0.02,
        clip_sample=True,
        clip_sample_range=8
    )

    with (torch.no_grad()):
        with torch.cuda.amp.autocast():
            ###########################################
            # STEP 1-1: generate the surface position #
            ###########################################
            surfPos = randn_tensor((batch_size, num_surfaces, 10)).to(device)

            pndm_scheduler.set_timesteps(200)
            for t in tqdm(pndm_scheduler.timesteps[:158], desc=os.path.basename(out_dir)):  #
                timesteps = t.reshape(-1).to(device)
                if class_label is not None:
                    _surfPos_ = surfPos.repeat(2, 1, 1)
                    pred = surfPos_model(_surfPos_, timesteps, class_label)
                    pred = pred[:batch_size] * (1 + w) - pred[batch_size:] * w  # 线性插值
                else:
                    pred = surfPos_model(surfPos, timesteps, class_label)
                surfPos = pndm_scheduler.step(pred, t, surfPos).prev_sample
                # # [test]
                # if t%50==0:
                #     draw_bbox_geometry(surfPos[0][:,:6].detach().cpu().numpy(),
                #                        [to_hex(plt.cm.coolwarm(i)) for i in np.linspace(0, 1, 32)])
            # Late increase for ABC/DeepCAD (slightly more efficient)
            if not eval_args['use_cf']:
                surfPos = surfPos.repeat(1, 2, 1)
                num_surfaces *= 2

            ddpm_scheduler.set_timesteps(1000)
            for t in tqdm(ddpm_scheduler.timesteps[-250:]):
                timesteps = t.reshape(-1).to(device)
                if class_label is not None:
                    _surfPos_ = surfPos.repeat(2, 1, 1)
                    pred = surfPos_model(_surfPos_, timesteps, class_label)
                    pred = pred[:batch_size] * (1 + w) - pred[batch_size:] * w
                else:
                    pred = surfPos_model(surfPos, timesteps, class_label)
                surfPos = ddpm_scheduler.step(pred, t, surfPos).prev_sample
                # # [test]
                # if t%10-1==0:
                #     draw_bbox_geometry(surfPos[0][:,:6].detach().cpu().numpy(),
                #                        [to_hex(plt.cm.coolwarm(i)) for i in np.linspace(0, 1, 32)])
            # # [test]
            # draw_bbox_geometry(surfPos[0][:, :6].detach().cpu().numpy(),
            #                    [to_hex(plt.cm.coolwarm(i)) for i in np.linspace(0, 1, 32)])

            #######################################
            # STEP 1-2: remove duplicate surfaces #
            #######################################

            if dedup:
                surfPos_deduplicate = []
                surfMask_deduplicate = []
                for ii in range(batch_size):
                    bboxes = np.round(surfPos[ii].unflatten(-1, torch.Size([2, 5])).detach().cpu().numpy(),4)#[...,-2:]  # mask维度
                    non_repeat = bboxes[:1]
                    
                    for bbox_idx, bbox in enumerate(bboxes):
                        diff = np.max(np.max(np.abs(non_repeat - bbox)[...,-2:], -1), -1)#
                        same = diff < bbox_threshold
                        bbox_rev = bbox[::-1]  # also test reverse bbox for matching
                        diff_rev = np.max(np.max(np.abs(non_repeat - bbox_rev)[...,-2:], -1), -1)#[...,-2:]
                        same_rev = diff_rev < bbox_threshold
                        if same.sum() >= 1 or same_rev.sum() >= 1: continue  # repeat value
                        else: non_repeat = np.concatenate([non_repeat, bbox[np.newaxis, :, :]], 0)
                    
                    bboxes = non_repeat.reshape(len(non_repeat), -1)
                    surf_mask = torch.zeros((1, len(bboxes))) == 1
                    bbox_padded = torch.concat([torch.FloatTensor(bboxes), torch.zeros(num_surfaces - len(bboxes), 10)]) ###################################
                    mask_padded = torch.concat([surf_mask, torch.zeros(1, num_surfaces - len(bboxes)) == 0], -1)
                    surfPos_deduplicate.append(bbox_padded)
                    surfMask_deduplicate.append(mask_padded)

                surfPos = torch.stack(surfPos_deduplicate).to(device)
                surfMask = torch.vstack(surfMask_deduplicate).to(device)  # mask掉的直接删除
                            
            else:
                surfPos = surfPos.to(device)
                surfMask = (torch.zeros((batch_size, num_surfaces)) == 1).to(device)
                        
            #################################
            # STEP 1-3:  generate surface z #
            #################################
            surfZ = randn_tensor((batch_size, num_surfaces, 8*8*8)).to(device)  # 1 30 192

            pndm_scheduler.set_timesteps(200)
            for t in tqdm(pndm_scheduler.timesteps):
                timesteps = t.reshape(-1).to(device)
                # if class_label is not None:
                #     _surfZ_ = surfZ.repeat(2, 1, 1)
                #     _surfPos_ = surfPos.repeat(2, 1, 1)
                #     _surfMask_ = surfMask.repeat(2, 1)
                #     pred = surfZ_model(_surfZ_, timesteps, _surfPos_, _surfMask_, class_label)
                #     pred = pred[:batch_size] * (1 + w) - pred[batch_size:] * w
                # else:

                # surfZ[:, known_idx, :] = pndm_scheduler.add_noise(surfZ[:, known_idx, :], 200-t)
                pred = surfZ_model(surfZ, timesteps, surfPos, surfMask, None)
                surfZ = pndm_scheduler.step(pred, t, surfZ).prev_sample

            surfZ = surfZ / z_scale
            surf_ncs = surf_vae(
                surfZ.unflatten(-1, torch.Size([8*8, 8])).flatten(0, 1).permute(0, 2, 1).unflatten(-1,torch.Size([8, 8]))) # 8 * 8 or 16 * 4 ？  in 120 3 8 8

            # remove duplicate surfaces (i.e. masked)
            surfMask = surfMask.squeeze(0)
            surfPos = surfPos.squeeze(0)[~surfMask].cpu().numpy() / 3.0  # (N_surfs, 10)            
            surf_ncs = surf_ncs[~surfMask].cpu().numpy()                 # (N_surfs, 6, 64, 64), xyzuvm
            surfMask = surfMask.cpu().numpy()                           # (N_surfs, )            
            
            # extract point mask
            point_mask = np.clip(surf_ncs[:, -1:, :, :], -50.0, 50.0)
            point_mask = 1.0 / (1.0 + np.exp(-np.clip(surf_ncs[:, -1:, :, :], -50, 50)))

            # normalized coordinate to global coordinate (uv)
            surfPos_uv = surfPos[..., -4:]      # (N_surfs, 4)
            surf_ncs_uv = surf_ncs[:, 3:5, :, :]  # (N_surfs, 2, 64, 64)
            bbox_scale_uv = np.max(surfPos_uv[:, 2:] - surfPos_uv[:, :2], axis=1)[:, None, None, None]
            bbox_offset_uv = ((surfPos_uv[:, 2:] + surfPos_uv[:, :2]) * 0.5)[:, :, None, None]
            surf_pnts_wcs_uv = surf_ncs_uv * (bbox_scale_uv * 0.5) + bbox_offset_uv
            
            # normalized coordinate to global coordinate (xyz)
            surfPos = surfPos[:, :6]          # (N_surfs, 6)
            surf_ncs = surf_ncs[:, :3, :, :]  # (N_surfs, 3, 64, 64)
            bbox_scale = np.max(surfPos[:, 3:] - surfPos[:, :3], axis=1)[:, None, None, None]
            bbox_offset = ((surfPos[:, 3:] + surfPos[:, :3]) * 0.5)[:, :, None, None]            
            surf_pnts_wcs = surf_ncs * (bbox_scale * 0.5) + bbox_offset
                        
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, 'result.pkl'), 'wb') as f:
                pickle.dump({'xyz': surf_pnts_wcs, 'uv': surf_pnts_wcs_uv, 'mask': point_mask}, f)          
                                               
            if vis:
                n_surfs = surfPos.shape[0]
                colors = plt.get_cmap('rainbow', n_surfs)

                fig = make_subplots(
                    rows=3, cols=5, subplot_titles=("XYZ", "UV", "Geometry Images"),
                    specs=[
                        [{"type": "scene", "colspan": 2, "rowspan": 2}, None, {"type": "xy", "colspan": 3, "rowspan": 2}, None, None],
                        [None, None, None, None, None],
                        [{"colspan": 5, "type": "xy"}, None, None, None, None]
                    ])
                                
                # geometry images and mask
                grid_imgs = make_grid(
                    torch.cat([torch.FloatTensor((surf_ncs+1.0)*0.5), torch.FloatTensor(point_mask)], dim=1), nrow=n_surfs, padding=5)
                grid_imgs = grid_imgs.permute(1, 2, 0).cpu().numpy()                
                grid_imgs = np.concatenate([grid_imgs[:, :, :3], np.repeat(grid_imgs[:, :, -1:], 3, axis=-1)], axis=0)
                
                for s_idx in range(n_surfs):
                    
                    surf_color = mcolors.to_hex(colors(s_idx))
                    
                    valid_pts = surf_pnts_wcs[s_idx].reshape(3, -1)[:, point_mask[s_idx].reshape(-1) > 0.5]                    
                    fig.add_trace(go.Scatter3d(
                        x=valid_pts[0, :],
                        y=valid_pts[1, :],
                        z=valid_pts[2, :],
                        mode='markers',
                        name='s%02d_xyz' % s_idx,
                        marker=dict(
                            size=2,
                            color=surf_color,
                            opacity=0.8
                        )
                    ), row=1, col=1)

                    valid_pts_uv = surf_pnts_wcs_uv[s_idx].reshape(2, -1)[:, point_mask[s_idx].reshape(-1) > 0.5]                    
                    fig.add_trace(go.Scatter(
                        x=valid_pts_uv[0, :],
                        y=valid_pts_uv[1, :],
                        mode='markers',
                        name='s%02d_uv' % s_idx,
                        marker=dict(
                            size=2,
                            color=surf_color,
                            opacity=0.8
                        )
                    ), row=1, col=3)
                    
                    fig.add_trace(px.imshow(grid_imgs).data[0], row=3, col=1)
                
                fig.update_layout(
                    height=1200, width=2400, title_text="Generated Garment", 
                    scene_camera=dict(
                        up=dict(x=0, y=1, z=0), center=dict(x=0, y=0, z=0), eye=dict(x=0, y=0, z=1.5)
                    ),
                    scene=dict(
                        xaxis=dict(range=[-1.5, 1.5]), 
                        yaxis=dict(range=[-1.5, 1.5]), 
                        zaxis=dict(range=[-1.5, 1.5]),
                        aspectmode='manual',
                        aspectratio=dict(x=1, y=1, z=1)
                        )
                )
                fig.update_xaxes(title_text="u", range=[-1.75, 1.0], row=1, col=2)
                fig.update_yaxes(title_text="v", range=[-1.5,  1.25], row=1, col=2)

                fig.write_html(f"{out_dir}/vis.html")


if __name__ == "__main__":
    raise Exception("This file is not used anymore.")


    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=['abc', 'deepcad', 'furniture', 'sxd'], default='sxd',
                        help="Choose between evaluation mode [abc/deepcad/furniture] (default: abc)")
    parser.add_argument('-c', '--config', type=str, default='configs/eval_config.yaml', help='Evaluation configuration file')
    parser.add_argument('-o', '--out_dir', type=str, default='generated', help='Output directory (default: generated)')
    parser.add_argument("-n", "--num_samples", type=int, default=1, help="Number of samples to generate (default: 1)")
    parser.add_argument("-v", "--vis", action='store_true', help="Visualize the generated samples (default: False)")
    parser.add_argument("--dedupe", action='store_true', help="Whether to remove duplicated surfaces (default: False)")

    args = parser.parse_args()

    # Load evaluation config
    with open(args.config, 'rb') as file: eval_args = yaml.safe_load(file)[args.mode]
    for i in range(args.num_samples): sample(eval_args, out_dir=f"{args.out_dir}/{i:04d}", vis=args.vis, dedup=args.dedupe)

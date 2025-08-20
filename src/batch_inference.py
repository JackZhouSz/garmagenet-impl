import os
import random
import pickle
import numpy as np
import argparse
from tqdm import tqdm

import torch
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
from matplotlib.colors import to_hex

from network import AutoencoderKLFastDecode, SurfZNet, SurfPosNet, TextEncoder
from diffusers import DDPMScheduler, PNDMScheduler
from utils import randn_tensor
from vis import draw_bbox_geometry



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
    surf_vae.to('cuda').eval()

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
        clip_sample=False,
    )

    # Load conditioning model
    if args.text_encoder is not None: text_enc = TextEncoder(args.text_encoder, device='cuda')
    else: text_enc = None

    # Load SurfPos Net
    surfpos_model = SurfPosNet(
        p_dim=10,
        condition_dim=text_enc.text_emb_dim if args.text_encoder is not None else -1,
        num_cf=-1
        )

    # Load SurfZ Net
    surfz_model = SurfZNet(
        p_dim=10, 
        z_dim=latent_size**2*latent_channels, 
        num_heads=12, 
        condition_dim=text_enc.text_emb_dim if args.text_encoder is not None else -1,
        num_cf=-1
        )
    surfz_model.load_state_dict(torch.load(args.surfz)['model_state_dict'])
    surfz_model.to('cuda').eval()
    
    print('[DONE] Models initialized.')

    return {
        'surf_vae': surf_vae,
        'ddpm_scheduler': ddpm_scheduler,
        'surfpos_model': surfpos_model,
        'surfz_model': surfz_model,
        'text_enc': text_enc,
        'latent_channels': latent_channels,
        'latent_size': latent_size
    }


def inference_one(models, surf_pos, surf_cls=None, caption='', output_fp='', vis=False):
    
    ddpm_scheduler = models['ddpm_scheduler']
    surfz_model = models['surfz_model']
    surf_vae = models['surf_vae']
    text_enc = models['text_enc']

    latent_channels = models['latent_channels']
    latent_size = models['latent_size']
    
    n_surfs, n_pads = surf_pos.shape[0], 32-surf_pos.shape[0]
    # # pad zero
    pad_idx = torch.randperm(n_surfs)
    _surf_mask = torch.cat([
        torch.zeros(n_surfs, dtype=bool), torch.ones(n_pads, dtype=bool)
    ], dim=0)[None, ...]
    _surf_pos = torch.cat([
        surf_pos[pad_idx, ...], torch.zeros((n_pads, *surf_pos.shape[1:]), dtype=surf_pos.dtype, device=surf_pos.device)
    ], dim=0)[None, ...]

    # encode text
    text_emb = text_enc(caption) if caption and text_enc is not None else None

    # Diffusion Generation
    _surf_z = randn_tensor((1, 32, latent_channels*latent_size*latent_size), device='cuda')
    ddpm_scheduler.set_timesteps(1000)
    for t in ddpm_scheduler.timesteps:
        timesteps = t.reshape(-1).to('cuda')
        pred = surfz_model(
            _surf_z, timesteps, _surf_pos.to('cuda'), _surf_mask.to('cuda'), text_emb, is_train=False)
        _surf_z = ddpm_scheduler.step(pred, t, _surf_z).prev_sample
        
    _surf_z = _surf_z.squeeze(0)[~_surf_mask.squeeze(0), ...]

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
    n_surfs = decoded_surf_pos.shape[0]
    colormap = plt.cm.coolwarm

    _surf_bbox = _surf_pos.squeeze(0)[~_surf_mask.squeeze(0), :].detach().cpu().numpy()
    _decoded_surf_pos = decoded_surf_pos.permute(0, 2, 3, 1).detach().cpu().numpy()
    _surf_ncs_mask = _decoded_surf_pos[..., -1:].reshape(n_surfs, -1) > 0.0
    _surf_ncs = _decoded_surf_pos[..., :3].reshape(n_surfs, -1, 3)
    _surf_uv_ncs = _decoded_surf_pos[..., 3:5].reshape(n_surfs, -1, 2)

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
            output_fp = output_fp.replace('.pkl', '_pointcloud.png')
            )

    result = {
        'surf_bbox': _surf_bbox,        # (N, 6)
        'surf_uv_bbox': _surf_uv_bbox,  # (N, 4)
        'surf_ncs': _surf_ncs,          # (N, 256*256, 3)
        'surf_uv_ncs': _surf_uv_ncs,    # (N, 256*256, 2)
        'surf_mask': _surf_ncs_mask,    # (N, 256*256) => bool
        'caption': caption              # str
    }

    if output_fp: 
        with open(output_fp, 'wb') as f: pickle.dump(result, f)
    
    # print('[DONE] save to:', output_fp)
    
    
def run(args):
    
    models = init_models(args)
    
    os.makedirs(args.output, exist_ok=True)
    
    with open(args.cache, 'rb') as f: data_cache = pickle.load(f)
    
    if args.num_samples == -1: sample_idxs = range(len(data_cache['item_idx']))
    else: sample_idxs = random.sample(range(len(data_cache['item_idx'])), k=args.num_samples)
    
    for sample_data_idx in tqdm(sample_idxs):
        start_idx, end_idx = data_cache['item_idx'][sample_data_idx]

        surf_pos = data_cache['surf_pos'][start_idx:end_idx].to('cuda')
        surf_cls = data_cache['surf_cls'][start_idx:end_idx].to('cuda')
        caption = data_cache['caption'][sample_data_idx]
        
        output_fp = os.path.join(args.output, f'{sample_data_idx:04d}.pkl')
        inference_one(models, surf_pos, surf_cls, caption, output_fp, vis=True)    

    print('[DONE]')
    
    
if __name__ == "__main__":
    raise NotImplementedError("Please use src/experiments/batch_inference/batch_inference.py")

    parser = argparse.ArgumentParser()
    
    # path configuration
    parser.add_argument(
        '--vae', type=str, 
        default='log/stylexd_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e550.pt', 
        help='Path to VAE model')
    parser.add_argument(
        '--surfz', type=str, 
        default='log/stylexd_surfz_xyzuv_mask_latent1_mode/ckpts/surfz_e230000.pt', 
        help='Path to SurfZ model')
    parser.add_argument(
        '--cache', type=str, 
        default='log/stylexd_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e550/encoder_mode/surfpos_validate.pkl', 
        help='Path to cache file')    
    parser.add_argument(
        '--output', type=str, default='generated/surfz_e230000', 
        help='Path to output directory')
    
    parser.add_argument('--block_dims', nargs='+', type=int, default=[16,32,32,64,64,128], help='Latent dimension of each block of the UNet model.')
    parser.add_argument('--latent_channels', type=int, default=1, help='Latent channels of the vae model.')
    parser.add_argument('--reso', type=int, default=256, help='Sample size of the vae model.')
    
    parser.add_argument('--num_samples', type=int, default=-1, help='Number of samples to inference.')

    args = parser.parse_args()
    
    run(args)
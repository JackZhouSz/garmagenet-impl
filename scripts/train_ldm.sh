#!/bin/bash\

### Train the Latent Diffusion Model ###
# --data_aug is optional
# max_face 30, max_edge 20 for deepcad 
# max_face 50, max_edge 30 for abc/furniture
# --surfvae refer to the surface vae weights 
# --edgevae refer to the edge vae weights 

### StyleXD - SurfPos
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption \
    --list data_process/stylexd_data_split_reso_256.pkl --option surfpos \
    --cache_dir log/stylexd_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e550/encoder_mode \
    --padding repeat \
    --expr stylexd_surfpos_xyzuv_pad_repeat_uncond --train_nepoch 100000 --test_nepoch 100 --save_nepoch 1000 \
    --batch_size 512 --max_face 32 --bbox_scaled 1.0 \
    --data_fields surf_bbox_wcs surf_uv_bbox_wcs


python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption \
    --list data_process/stylexd_data_split_reso_256.pkl --option surfpos \
    --cache_dir log/stylexd_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e550/encoder_mode \
    --padding repeat --text_encoder CLIP \
    --expr stylexd_surfpos_xyzuv_pad_repeat_cond_clip --train_nepoch 100000 --test_nepoch 100 --save_nepoch 1000 \
    --batch_size 2048 --max_face 32 --bbox_scaled 1.0 \
    --data_fields surf_bbox_wcs surf_uv_bbox_wcs caption


### StyleXD - SurfZ
CUDA_VISIBLE_DEVICES=0,1 python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption \
    --list data_process/stylexd_data_split_reso_256.pkl --option surfz \
    --surfvae log/stylexd_vae_surf_256_xyz_uv_mask_unet6/ckpts/epoch_1800.pt \
    --cache_dir log/stylexd_vae_surf_256_xyz_uv_mask_unet6/cache/epoch_1800/encoder_mode \
    --expr stylexd_surfz_xyzuv_pad_zero_z_scale_1.0 --train_nepoch 500000 --test_nepoch 500 --save_nepoch 1000 \
    --batch_size 3072 --chunksize -1 --padding zero --z_scaled 1.0 --bbox_scaled 1.0 \
    --block_dims 16 32 32 64 64 128 --max_face 32 --sample_mode mode \
    --data_fields surf_ncs surf_uv_ncs surf_mask surf_bbox_wcs surf_uv_bbox_wcs \
    --finetune --weight log/stylexd_surfz_xyzuv_pad_zero_z_scale_1.0/ckpts/surfz_e50000.pt
    

CUDA_VISIBLE_DEVICES=0 python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption \
    --list data_process/stylexd_data_split_reso_256.pkl --option surfz \
    --surfvae log/stylexd_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e550.pt \
    --cache_dir log/stylexd_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e550/encoder_mode \
    --expr stylexd_surfz_xyzuv_mask_latent1_mode --train_nepoch 1000000 --test_nepoch 500 --save_nepoch 5000 \
    --batch_size 4096 --chunksize -1 --padding zero --bbox_scaled 1.0 --z_scaled 1.0 \
    --block_dims 16 32 32 64 64 128 --latent_channels 1 --max_face 32 --sample_mode mode \
    --data_fields surf_ncs surf_uv_ncs surf_mask surf_bbox_wcs surf_uv_bbox_wcs



CUDA_VISIBLE_DEVICES=0 python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption \
    --list data_process/stylexd_data_split_reso_256.pkl --option surfz \
    --surfvae log/stylexd_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e300.pt \
    --cache_dir log/stylexd_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e300/encoder_mode \
    --expr stylexd_surfz_xyzuv_latent1_mode --train_nepoch 50000 --test_nepoch 50 --save_nepoch 500 \
    --batch_size 1024 --chunksize -1 --padding zero --bbox_scaled 1.0 \
    --block_dims 16 32 32 64 64 128 --latent_channels 1 --max_face 32 --sample_mode mode \
    --data_fields surf_ncs surf_uv_ncs surf_mask surf_bbox_wcs surf_uv_bbox_wcs 


### Training DeepCAD Latent Diffusion Model ###  
python src/ldm.py --data data_process/deepcad_parsed \
    --list data_process/deepcad_data_split_6bit.pkl --option surfpos --gpu 0 1 \
    --expr deepcad_ldm_surfpos --train_nepoch 3000 --test_nepoch 200 --save_nepoch 200 \
    --max_face 30 --max_edge 20

python src/ldm.py --data data_process/deepcad_parsed \
    --list data_process/deepcad_data_split_6bit.pkl --option surfz \
    --surfvae log/deepcad_vae_surf.pt --gpu 0 1 \
    --expr deepcad_ldm_surfz --train_nepoch 3000 --batch_size 256 \
    --max_face 30 --max_edge 20

python src/ldm.py --data data_process/deepcad_parsed \
    --list data_process/deepcad_data_split_6bit.pkl --option edgepos \
    --surfvae log/deepcad_vae_surf.pt --gpu 0 1 \
    --expr deepcad_ldm_edgepos --train_nepoch 1000 --batch_size 128 \
    --max_face 30 --max_edge 20

python src/ldm.py --data data_process/deepcad_parsed \
    --list data_process/deepcad_data_split_6bit.pkl --option edgez \
    --surfvae log/deepcad_vae_surf.pt --edgevae log/deepcad_vae_edge.pt --gpu 0 1 \
    --expr deepcad_ldm_edgez --train_nepoch 1000 --batch_size 128 \
    --max_face 30 --max_edge 20


### Training ABC Latent Diffusion Model ###  
python src/ldm.py --data data_process/abc_parsed \
    --list data_process/abc_data_split_6bit.pkl --option surfpos --gpu 0 1 \
    --expr abc_ldm_surfpos --train_nepoch 1000 --test_nepoch 200 --save_nepoch 200 \
    --max_face 50 --max_edge 30

python src/ldm.py --data data_process/abc_parsed \
    --list data_process/abc_data_split_6bit.pkl --option surfz \
    --surfvae log/abc_vae_surf.pt --gpu 0 1 \
    --expr abc_ldm_surfz --train_nepoch 1000 --batch_size 256 \
    --max_face 50 --max_edge 30

python src/ldm.py --data data_process/abc_parsed \
    --list data_process/abc_data_split_6bit.pkl --option edgepos \
    --surfvae log/abc_vae_surf.pt --gpu 0 1 \
    --expr abc_ldm_edgepos --train_nepoch 300 --batch_size 64 \
    --max_face 50 --max_edge 30

python src/ldm.py --data data_process/abc_parsed \
    --list data_process/abc_data_split_6bit.pkl --option edgez \
    --surfvae log/abc_vae_surf.pt --edgevae log/abc_vae_edge.pt --gpu 0 1 \
    --expr abc_ldm_edgez --train_nepoch 300 --batch_size 64 \
    --max_face 50 --max_edge 30


### Training Furniture Latent Diffusion Model (classifier-free) ###  
python src/ldm.py --data data_process/furniture_parsed \
    --list data_process/furniture_data_split_6bit.pkl --option surfpos --gpu 0 1 \
    --expr furniture_ldm_surfpos --train_nepoch 3000 --test_nepoch 200 --save_nepoch 200 \
    --max_face 50 --max_edge 30 --use_cf

python src/ldm.py --data data_process/furniture_parsed \
    --list data_process/furniture_data_split_6bit.pkl --option surfz \
    --surfvae log/furniture_vae_surf.pt --gpu 0 1 \
    --expr furniture_ldm_surfz --train_nepoch 3000 --batch_size 256 \
    --max_face 50 --max_edge 30 --use_cf

python src/ldm.py --data data_process/furniture_parsed \
    --list data_process/furniture_data_split_6bit.pkl --option edgepos \
    --surfvae log/furniture_vae_surf.pt --gpu 0 1 \
    --expr furniture_ldm_edgepos --train_nepoch 1000 --batch_size 64 \
    --max_face 50 --max_edge 30 --use_cf

python src/ldm.py --data data_process/furniture_parsed \
    --list data_process/furniture_data_split_6bit.pkl --option edgez \
    --surfvae log/furniture_vae_surf.pt --edgevae log/furniture_vae_edge.pt --gpu 0 1 \
    --expr furniture_ldm_edgez --train_nepoch 1000 --batch_size 64 \
    --max_face 50 --max_edge 30 --use_cf
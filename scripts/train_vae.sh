#!/bin/bash\

# VAE mask Q124
# Bsize 187:64
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/vae.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed \
    --list data_process/data_lists/stylexd_data_split_reso_256_Q1Q2Q4.pkl \
    --expr stylexdQ1Q2Q4_vae_surf_256_mask_unet6_latent_1 \
    --batch_size 64 --block_dims 16 32 32 64 64 128 --latent_channels 1 \
    --test_nepoch 20 --save_nepoch 50 --train_nepoch 3000 \
    --data_fields surf_mask --chunksize 512

# VAE xyz-w normal mask Q124, latent 16x16x1
# Bsize 187:64
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/vae.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed \
    --list data_process/data_lists/stylexd_data_split_reso_256_Q1Q2Q4.pkl \
    --expr stylexdQ1Q2Q4_vae_surf_256_xyz-w_normal_mask_unet6_latent_16_16_1 \
    --batch_size 64 --block_dims 16 32 32 64 64 --latent_channels 1 \
    --test_nepoch 20 --save_nepoch 50 --train_nepoch 3000 \
    --data_fields surf_wcs surf_normals surf_mask --chunksize 512

# VAE xyz-w mask Q124, latent 16x16x1
# Bsize 187:80
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/vae.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed \
    --list data_process/data_lists/stylexd_data_split_reso_256_Q1Q2Q4.pkl \
    --expr stylexdQ1Q2Q4_vae_surf_256_xyz-w_mask_unet6_latent_16_16_1 \
    --batch_size 80 --block_dims 16 32 32 64 64 --latent_channels 1 \
    --test_nepoch 20 --save_nepoch 50 --train_nepoch 3000 \
    --data_fields surf_wcs surf_mask --chunksize 512

# VAE xyz Q124, latent 8x8x4
# Bsize 190:128
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/vae.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed \
    --list data_process/data_lists/stylexd_data_split_reso_256_Q1Q2Q4.pkl \
    --expr stylexdQ1Q2Q4_vae_surf_256_xyz_unet6_latent_4 \
    --batch_size 128 --block_dims 16 32 32 64 64 128 --latent_channels 4 \
    --test_nepoch 20 --save_nepoch 50 --train_nepoch 3000 \
    --data_fields surf_ncs surf_uv_ncs surf_mask --chunksize 1024

# VAE xyz_uv_mask Q124
# Bsize 188:
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/vae.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed \
    --list data_process/data_lists/stylexd_data_split_reso_256_Q1Q2Q4.pkl \
    --expr stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1 \
    --batch_size 64 --block_dims 16 32 32 64 64 128 --latent_channels 1 \
    --test_nepoch 10 --save_nepoch 50 --train_nepoch 2000 \
    --data_fields surf_ncs surf_uv_ncs surf_mask --chunksize 512

# 256 lry ===
python src/vae.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption \
    --list data_process/stylexd_data_split_reso_256.pkl \
    --expr stylexd_vae_surf_256_xyz_nrm_mask_unet6_latent_1 \
    --batch_size 64 --block_dims 16 32 32 64 64 128 --latent_channels 1 \
    --test_nepoch 10 --save_nepoch 50 --train_nepoch 2000 \
    --data_fields surf_ncs surf_normals surf_mask --chunksize 512

python src/vae.py --data /data/AIGP/brep_reso_256_edge_snap \
    --train_list data_process/stylexd_data_split_reso_256.pkl \
    --val_list data_process/stylexd_data_split_reso_256.pkl \
    --gpu 0 --expr stylexd_vae_surf_256_xyz_uv_mask \
    --batch_size 32 --train_nepoch 2000 --block_dims 32,64,64,128 \
    --data_fields surf_ncs surf_uv_ncs surf_mask --chunksize 128 \
    --finetune --weight log/stylexd_vae_surf_256_xyz_uv_mask/epoch_20.pt


CUDA_VISIBLE_DEVICES=0 python src/vae.py --data /data/AIGP/brep_reso_256_edge_snap \
    --train_list data_process/stylexd_data_split_reso_256.pkl \
    --val_list data_process/stylexd_data_split_reso_256.pkl \
    --expr stylexd_vae_surf_256_xyz_uv_mask_unet6_cont \
    --batch_size 64 --block_dims 16 32 32 64 64 128 \
    --test_nepoch 50 --save_nepoch 100 --train_nepoch 8000 \
    --data_fields surf_ncs surf_uv_ncs surf_mask --chunksize 256 \
    --finetune --weight log/stylexd_vae_surf_256_xyz_uv_mask_unet6/ckpts/epoch_1800.pt


### DeepCAD VAE Training ###
python src/vae.py --data data_process/deepcad_parsed \
    --train_list data_process/deepcad_data_split_6bit_surface.pkl \
    --val_list data_process/deepcad_data_split_6bit.pkl \
    --option surface --gpu 0 --expr deepcad_vae_surf --train_nepoch 400 --data_aug

python src/vae.py --data data_process/deepcad_parsed \
    --train_list data_process/deepcad_data_split_6bit_edge.pkl \
    --val_list data_process/deepcad_data_split_6bit.pkl \
    --option edge --gpu 0 --expr deepcad_vae_edge --train_nepoch 400 --data_aug


### ABC VAE Training ###
python src/vae.py --data data_process/abc_parsed \
    --train_list data_process/abc_data_split_6bit_surface.pkl \
    --val_list data_process/abc_data_split_6bit.pkl \
    --option surface --gpu 0 --expr abc_vae_surf --train_nepoch 200 --data_aug

python src/vae.py --data data_process/abc_parsed \
    --train_list data_process/abc_data_split_6bit_edge.pkl \
    --val_list data_process/abc_data_split_6bit.pkl \
    --option edge --gpu 0 --expr abc_vae_edge --train_nepoch 200 --data_aug


### Furniture VAE Training (fintune) ###
python src/vae.py --data data_process/furniture_parsed \
    --train_list data_process/furniture_data_split_6bit_surface.pkl \
    --val_list data_process/furniture_data_split_6bit.pkl \
    --option surface --gpu 0 --expr furniture_vae_surf --train_nepoch 200 --finetune \
    --weight log/deepcad_vae_surf.pt

python src/vae.py --data data_process/furniture_parsed \
    --train_list data_process/furniture_data_split_6bit_edge.pkl \
    --val_list data_process/furniture_data_split_6bit.pkl \
    --option edge --gpu 0 --expr furniture_vae_edge --train_nepoch 200 --finetune \
    --weight log/deepcad_vae_edge.pt


#!/bin/bash\

### StyleXD VAE Training ###
python src/vae.py --data /data/AIGP/brep_reso_64_edge_snap \
    --train_list data_process/stylexd_data_split_reso_64.pkl \
    --val_list data_process/stylexd_data_split_reso_64.pkl \
    --gpu 1 --expr stylexd_vae_surf_64_xyz_uv_mask \
    --batch_size 512 --train_nepoch 2000 --block_dims 32,64,64,128 \
    --data_fields surf_ncs surf_uv_ncs surf_mask 


python src/vae.py --data /data/AIGP/brep_reso_256_edge_snap \
    --train_list data_process/stylexd_data_split_reso_256.pkl \
    --val_list data_process/stylexd_data_split_reso_256.pkl \
    --gpu 0 --expr stylexd_vae_surf_256_xyz_uv_mask \
    --batch_size 32 --train_nepoch 2000 --block_dims 32,64,64,128 \
    --data_fields surf_ncs surf_uv_ncs surf_mask --chunksize 128 \
    --finetune --weight log/stylexd_vae_surf_256_xyz_uv_mask/epoch_20.pt


python src/vae.py --data /data/AIGP/brep_reso_256_edge_snap \
    --train_list data_process/stylexd_data_split_reso_256.pkl \
    --val_list data_process/stylexd_data_split_reso_256.pkl \
    --gpu 1 --expr stylexd_vae_surf_256_xyz_uv_mask_unet6 \
    --batch_size 64 --train_nepoch 2000 --block_dims 16,32,32,64,64,128 \
    --data_fields surf_ncs surf_uv_ncs surf_mask --chunksize 256 

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
## Requirements

### Environment (Tested)
- Linux
- Python 3.9
- CUDA 11.8 
- PyTorch 2.2 
- Diffusers 0.27


### Dependencies

Install PyTorch and other dependencies:
```
conda create --name garmage_env python=3.9 -y
conda activate garmage_env

pip install -r requirements.txt
pip install chamferdist
```

If `chamferdist` fails to install here are a few options to try:

- If there is a CUDA version mismatch error, then try setting the `CUDA_HOME` environment variable to point to CUDA installation folder. The CUDA version of this folder must match with PyTorch's version i.e. 11.8.

- Try [building from source](https://github.com/krrish94/chamferdist?tab=readme-ov-file#building-from-source).


## Data
The dataset for training consists of `*.pkl` files for each garment (e.g. `resources/examples/processes/00000.pkl`) containing the following fields:
```python
result = {
    # raw data path
    'data_fp': data_item,
    # comma-separated description for the garmage
    'caption': "dress, fitted, round neck, puff sleeves, empire waist", 
    
    # (3, ), global offset for all garments default to [0., 1000., 0.]
    'global_offset': global_offset.astype(np.float32),  
    # float, global scale for all garments default to 2000
    'global_scale': global_scale,                 
    # (2, ) global uv offset, default to [0., 1000.]
    'uv_offset': uv_offset.astype(np.float32),  
    # float, global uv scale, default to 3000.
    'uv_scale': uv_scale,         
    
    # xyz
    'surf_cls': np.array(panel_cls, dtype=np.int32),
    'surf_mask': surf_mask.astype(bool),  # (N, H, W, 1), mask for each panel, 
                                          # N refers to number of panels in the garment. 
                                          # By default H=W=256 refers to Garmage resolution.
    'surf_wcs': surfs_wcs.astype(np.float32),   # (N, H, W, 3), panel points in world coordinate
    'surf_ncs': surfs_ncs.astype(np.float32),   # (N, H, W, 3), panel points in normalized coordinate
    
    # uv
    'surf_uv_wcs': surfs_uv_wcs.astype(np.float32),  # (N, H, W, 2)
    'surf_uv_ncs': surfs_uv_ncs.astype(np.float32),  # (N, H, W, 2)               

    # normal
    'surf_normals': surf_norms.astype(np.float32),
    'corner_normals': corner_normals.astype(np.float32),
    
#     # optional edge-related fields
#     'edge_wcs': edges_wcs.astype(np.float32),
#     'edge_ncs': edges_ncs.astype(np.float32),
#     'corner_wcs': corner_wcs.astype(np.float32),
#     'edge_uv_wcs': edges_uv_wcs.astype(np.float32),
#     'edge_uv_ncs': edges_uv_ncs.astype(np.float32),
#     'corner_uv_wcs': corner_uv_wcs.astype(np.float32),
#     'faceEdge_adj': faceEdge_adj

}
```
The `*.pkl` files are generated from raw `*.obj` garment assets and their sewing patterns saved as `panel.json` files (e.g., resources/examples/objs/0000). We use the following script to convert raw assets to `*.pkl` format:

```bash
cd data_process
python process_sxd.py -i {OBJS_SOURCE_DIR} -o {PKL_OUTPUT_DIR} --range 0,16 --use_uni_norm --nf 256
```
where `--range` indicates the input range, `--use_uni_norm` is a boolean flag for universal normalization (i.e. all the garments in the dataset will share the same global offset and scale if `--use_uni_norm` is specified) and `--nf` refers to the Garmage resolution.

## Training 

Firstly, train the VAE encoder to compress Garmages. By default, all Garmages in the dataset have a resolution of $256\times256$. Each garment is represented as a set of per-panel Garmages, forming a tensor of shape $N\times256\times256\times C$, where $C$ depends on the desired encoded fields. For instance, the simplest Garmage has four channels: the first three encode geometric positions, while the fourth (alpha) outlines the sewing pattern. For example:

```bash
python src/vae.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption \
    --list data_process/stylexd_data_split_reso_256.pkl \
    --expr stylexd_vae_surf_256_xyz_nrm_mask_unet6_latent_1 \
    --batch_size 64 --block_dims 16 32 32 64 64 128 --latent_channels 1 \
    --test_nepoch 10 --save_nepoch 50 --train_nepoch 2000 \
    --data_fields surf_ncs surf_normals surf_mask --chunksize 512
```

Secondly, train the topology generator:

```bash
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption \
    --list data_process/stylexd_data_split_reso_256.pkl --option surfpos \
    --cache_dir log/stylexd_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e550/encoder_mode \
    --padding repeat \
    --expr stylexd_surfpos_xyzuv_pad_repeat_uncond --train_nepoch 100000 --test_nepoch 100 --save_nepoch 1000 \
    --batch_size 512 --max_face 32 --bbox_scaled 1.0 \
    --data_fields surf_bbox_wcs surf_uv_bbox_wcs
```

Finally, train the geometry generator:

```bash
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption \
    --list data_process/stylexd_data_split_reso_256.pkl --option surfz \
    --surfvae log/stylexd_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e800.pt \
    --cache_dir log/stylexd_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e800/encoder_mode \
    --expr stylexd_surfz_xyzuv_mask_latent1_mode_with_caption --train_nepoch 100000 --test_nepoch 200 --save_nepoch 5000 \
    --batch_size 2048 --chunksize -1 --padding zero --bbox_scaled 1.0 --z_scaled 1.0 --text_encoder CLIP \
    --block_dims 16 32 32 64 64 128 --latent_channels 1 --max_face 32 --sample_mode mode \
    --data_fields surf_ncs surf_uv_ncs surf_mask surf_bbox_wcs surf_uv_bbox_wcs caption
```

Training of the topology/geometry generators can run in parallel.







------

FOLLOWING ADD BY LSR



## Docker

现在我使用的Image放在187的**/home/lsr/docker/images/style3d_gen.tar**，现有的容器在187、188、190三台机器上，名字包含**style_gen_lsr**的全都是。

**创建容器**

```bash
docker run --gpus all --ipc=host -it -P -d --name <CONTAINER_NAME> \
    -v /data:/data \
    brep:lsr /bin/bash
```



## Training on Q124

### DataList 

**现在使用的 Q124 datalist**

data_process/data_lists/stylexd_data_split_reso_256_Q1Q2Q4.pkl

现在使用的datalist是绝对路径的，因为Q12、Q4训练数据存放在不同文件夹下

**生成绝对路径的datalist**

[data_process/data_lists/gen_data_list_abspath/gen_data_list.py](data_process/data_lists/gen_data_list_abspath/gen_data_list.py)

```bash
cd /data/lsr/code/style3d_gen
export PYTHONPATH=/data/lsr/code/style3d_gen
python data_process/data_lists/gen_data_list_abspath/gen_data_list.py \
    --garmage_dirs /data/AIGP/brep_reso_256_edge_snap_with_caption /data/AIGP/Q4/brep_reso_256_edge_snap \
    --output_dir /data/lsr/code/style3d_gen/_LSR/gen_data_list/output
```



### Prepare Sketch Features

准备好了的SketchFeature放在187、188、190的 **/data/AIGP/feature_laion2b/**

**提前线稿特征的方法**

```
python prepare_sketch.py \
    --dataset_folder /data/AIGP/silhouettes/ \
    --feature_output_folder /data/lsr/dataset/feature_laion2b
```



### Train VAE

#### **Q124 局部坐标XYZ + UV + Mask，Latent8x8x1**

```bash
# 192.168.31.188
cd /data/lsr/code/style3d_gen
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/vae.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed \
    --list data_process/data_lists/stylexd_data_split_reso_256_Q1Q2Q4.pkl \
    --expr stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1 \
    --batch_size 64 --block_dims 16 32 32 64 64 128 --latent_channels 1 \
    --test_nepoch 10 --save_nepoch 50 --train_nepoch 2000 \
    --data_fields surf_ncs surf_uv_ncs surf_mask --chunksize 512
```

#### **Q124 世界坐标XYZ + Mask，Latent16x16x1**

```bash
# 192.168.31.187
cd /data/lsr/code/style3d_gen
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/vae.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed \
    --list data_process/data_lists/stylexd_data_split_reso_256_Q1Q2Q4.pkl \
    --expr stylexdQ1Q2Q4_vae_surf_256_xyz-w_mask_unet6_latent_16_16_1 \
    --batch_size 80 --block_dims 16 32 32 64 64 --latent_channels 1 \
    --test_nepoch 20 --save_nepoch 50 --train_nepoch 3000 \
    --data_fields surf_wcs surf_mask --chunksize 512
```



### Train Pos+Z（模型TransformerEncoder）

#### **Q124 unCond 局部坐标XYZ + UV + Mask pad-zero**

**POS**

```bash
# 192.168.31.187
cd /data/lsr/code/style3d_gen
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed \
    --list data_process/data_lists/stylexd_data_split_reso_256_Q1Q2Q4.pkl --option surfpos \
    --cache_dir log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e0800/encoder_mode \
    --padding zero \
    --expr stylexdQ1Q2Q4_surfpos_xyzuv_pad_zero_uncond --train_nepoch 100000 --test_nepoch 100 --save_nepoch 1000 \
    --batch_size 512 --max_face 32 --bbox_scaled 1.0 \
    --data_fields surf_bbox_wcs surf_uv_bbox_wcs
```

**Z**

```bash
# 192.168.31.187
cd /data/lsr/code/style3d_gen
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed \
    --list data_process/data_lists/stylexd_data_split_reso_256_Q1Q2Q4.pkl --option surfz \
    --surfvae log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e0800.pt \
    --cache_dir log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e0800/encoder_mode \
    --expr stylexdQ1Q2Q4_surfz_xyzuv_pad_zero_uncond --train_nepoch 100000 --test_nepoch 200 --save_nepoch 5000 \
    --batch_size 2048 --chunksize -1 --padding zero --bbox_scaled 1.0 --z_scaled 1.0 \
    --block_dims 16 32 32 64 64 128 --latent_channels 1 --max_face 32 --sample_mode mode \
    --data_fields surf_ncs surf_uv_ncs surf_mask surf_bbox_wcs surf_uv_bbox_wcs
```

#### Q124  点云cond(PointE)  局部坐标XYZ + UV + Mask  pad-zero

**POS**

```bash
# 192.168.31.187
cd /data/lsr/code/style3d_gen
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed \
    --list data_process/data_lists/stylexd_data_split_reso_256_Q1Q2Q4.pkl --option surfpos \
    --cache_dir log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e0800_pcCond/encoder_mode \
    --padding zero \
    --expr stylexdQ1Q2Q4_surfpos_xyzuv_pad_zero_pcCond --train_nepoch 100000 --test_nepoch 100 --save_nepoch 1000 \
    --batch_size 3420 --max_face 32 --bbox_scaled 1.0 \
    --pointcloud_encoder POINT_E \
    --data_fields surf_bbox_wcs surf_uv_bbox_wcs pointcloud_feature
```

**Z**

```bash
# 192.168.31.188
cd /data/lsr/code/style3d_gen
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed \
    --list data_process/data_lists/stylexd_data_split_reso_256_Q1Q2Q4.pkl  --option surfz \
    --surfvae log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e0800.pt \
    --cache_dir log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e0800_pcCond_Q124/encoder_mode \
    --expr stylexdQ1Q2Q4_surfz_xyzuv_pad_zero_pcCond --train_nepoch 100000 --test_nepoch 200 --save_nepoch 5000 \
    --batch_size 3420 --chunksize -1 --padding zero --bbox_scaled 1.0 --z_scaled 1.0 \
    --block_dims 16 32 32 64 64 128 --latent_channels 1 --max_face 32 --sample_mode mode \
    --pointcloud_encoder POINT_E \
    --data_fields surf_ncs surf_uv_ncs surf_mask surf_bbox_wcs surf_uv_bbox_wcs pointcloud_feature
```

#### Q124  线稿cond(laion2b)  局部坐标XYZ + UV + Mask  pad-zero

**POS**

```bash
# 192.168.31.187
cd /data/lsr/code/style3d_gen
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed \
    --list data_process/data_lists/stylexd_data_split_reso_256_Q1Q2Q4.pkl --option surfpos \
    --cache_dir log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e0800_sketchCond/encoder_mode \
    --padding zero \
    --expr stylexdQ1Q2Q4_surfpos_xyzuv_pad_zero_sketchCond --train_nepoch 100000 --test_nepoch 100 --save_nepoch 1000 \
    --batch_size 2500 --max_face 32 --bbox_scaled 1.0 \
    --sketch_encoder LAION2B --sketch_feature_dir /data/AIGP/feature_laion2b \
    --data_fields surf_bbox_wcs surf_uv_bbox_wcs sketch_feature
```

**Z**

```bash
# 192.168.30.190
cd /data/lsr/code/style3d_gen
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed \
    --list data_process/data_lists/stylexd_data_split_reso_256_Q1Q2Q4.pkl  --option surfz \
    --surfvae log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e0800.pt \
    --cache_dir log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e0800_sketchCond_Q124/encoder_mode \
    --expr stylexdQ1Q2Q4_surfz_xyzuv_pad_zero_sketchCond --train_nepoch 100000 --test_nepoch 200 --save_nepoch 5000 \
    --batch_size 5000 --chunksize -1 --padding zero --bbox_scaled 1.0 --z_scaled 1.0 \
    --block_dims 16 32 32 64 64 128 --latent_channels 1 --max_face 32 --sample_mode mode \
    --sketch_encoder LAION2B --sketch_feature_dir /data/AIGP/feature_laion2b \
    --data_fields surf_ncs surf_uv_ncs surf_mask surf_bbox_wcs surf_uv_bbox_wcs sketch_feature
```





### Train Pos+Z（模型Hunyuan3D 2.0 Dit）

使用的模型修改自 [https://github.com/Tencent-Hunyuan/Hunyuan3D-2/blob/main/hy3dgen/shapegen/models/denoisers/hunyuan3ddit.py](https://github.com/Tencent-Hunyuan/Hunyuan3D-2/blob/main/hy3dgen/shapegen/models/denoisers/hunyuan3ddit.py)

#### Q124  线稿cond(laion2b)  局部坐标XYZ + UV + Mask  pad-zero

**POS**

```bash
# 192.168.31.188
cd /data/lsr/code/style3d_gen
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed \
    --list data_process/data_lists/stylexd_data_split_reso_256_Q1Q2Q4.pkl  --option surfpos --denoiser_type hunyuan_dit \
    --cache_dir log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e0800_sketchCond_Q124/encoder_mode \
    --expr stylexdQ1Q2Q4_surfpos_HYdit_L2+6_emb384_pad_repeat_sketchCond --train_nepoch 200000 --test_nepoch 1000 --save_nepoch 1000 \
    --batch_size 1650 --chunksize -1 --padding repeat --max_face 32 \
    --embed_dim 384 --num_layer 2 6 \
    --sketch_encoder LAION2B --sketch_feature_dir /data/AIGP/feature_laion2b \
    --data_fields surf_bbox_wcs surf_uv_bbox_wcs sketch_feature

```

**Z**

```bash
# 192.168.31.188
cd /data/lsr/code/style3d_gen
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed \
    --list data_process/data_lists/stylexd_data_split_reso_256_Q1Q2Q4.pkl  --option surfz --denoiser_type hunyuan_dit \
    --surfvae log/stylexdQ1Q2Q4_vae_surf_256_xyz_mask_unet6_latent_16_16_1/ckpts/vae_e0850.pt \
    --cache_dir log/stylexdQ1Q2Q4_vae_surf_256_xyz_mask_unet6_latent_16_16_1/cache/vae_e0850_sketchCond_Q124_latent_16_16_1/encoder_mode \
    --expr stylexdQ1Q2Q4_surfz_HYdit_Layer_2_6_xyz_mask_pad_zero_sketchCond_latent_16_16_1 --train_nepoch 100000 --test_nepoch 200 --save_nepoch 1000 \
    --batch_size 1650 --chunksize -1 --padding zero --bbox_scaled 1.0 --z_scaled 1.0 \
    --block_dims 16 32 32 64 64 --latent_channels 1 --max_face 32 --sample_mode mode \
    --embed_dim 768 --num_layer 3 9 \
    --sketch_encoder LAION2B --sketch_feature_dir /data/AIGP/feature_laion2b \
    --data_fields surf_ncs surf_mask surf_bbox_wcs surf_uv_bbox_wcs sketch_feature
```




## Generation and Evaluation

Test the trained model with:
```bash
python src/batch_inference.py
```



### **batch inference Q124 Uncond**

```bash
# 192.168.31.187
export PYTHONPATH=/data/lsr/code/style3d_gen
python _LSR/experiments/batch_inference/batch_inference.py \
        --vae /data/lsr/models/style3d_gen/surf_vae/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e0800.pt \
        --surfpos /data/lsr/models/style3d_gen/surf_pos/stylexdQ1Q2Q4_surfpos_xyzuv_pad_zero_uncond/ckpts/surfpos_e32000.pt \
        --surfz /data/lsr/models/style3d_gen/surf_z/stylexdQ1Q2Q4_surfz_xyzuv_pad_zero_uncond/ckpts/surfz_e380000.pt \
        --cache /data/lsr/models/style3d_gen/surf_vae/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e0800/encoder_mode/surfpos_validate.pkl \
        --output generated/surfz_e380000 \  # --use_original_pos \  # 使用 GT 的 POS
        --padding zero  # --save_denoising [optional]
```

### batch inference Q124 pcCond

```bash
# 192.168.31.188
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/experiments/batch_inference/batch_inference.py \
    --vae log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e0800.pt \
    --surfpos log/stylexdQ1Q2Q4_surfpos_xyzuv_pad_zero_pcCond/ckpts/surfpos_e93000.pt \
    --surfz log/stylexdQ1Q2Q4_surfz_xyzuv_pad_zero_pcCond/ckpts/surfz_e200000.pt \
    --cache log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e0800_pcCond_Q124/encoder_mode/surfz_validate.pkl \
    --pointcloud_encoder POINT_E \  # --use_original_pos \  # 使用 GT 的 POS
    --output generated/xyzuv_pad_zero_pcCond_surfz_e200000 \
    --padding zero  # --save_denoising [optional]
```

### batch inference Q124 sketchCond

```bash
# run on 192.168.30.190
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/experiments/batch_inference/batch_inference.py \
    --vae log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e0800.pt \
    --surfpos log/stylexdQ1Q2Q4_surfpos_xyzuv_pad_zero_sketchCond/ckpts/surfpos_e59000.pt \
    --surfz log/stylexdQ1Q2Q4_surfz_xyzuv_pad_zero_sketchCond/ckpts/surfz_e150000.pt \
    --cache log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e0800_sketchCond_Q124/encoder_mode/surfz_validate.pkl \
    --sketch_encoder LAION2B \  # --use_original_pos \  # 使用 GT 的 POS
    --output generated/xyzuv_pad_zero_sketchCond_surfz_e_e150000 \
    --padding zero  # --save_denoising [optional]
```

### HunyuanDit batch inference Q124 sketchCond

**使用DDPM的：**
```bash
# run on 192.168.31.188
cd /data/lsr/code/style3d_gen
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/experiments/batch_inference/batch_inference.py \
    --vae log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e4200.pt \
    --surfpos log/stylexdQ1Q2Q4_surfpos_xyzuv_pad_zero_sketchCond/ckpts/surfpos_e59000.pt \
    --surfz log/stylexdQ1Q2Q4_surfz_HYdit_xyzuv_pad_zero_sketchCond/ckpts/surfz_e5000.pt --surfz_type 'hunyuan_dit' \
    --cache log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e0800_sketchCond_Q124/encoder_mode/surfz_validate.pkl \
    --sketch_encoder LAION2B \  # --use_original_pos \  # 使用 GT 的 POS
    --output generated/xyzuv_pad_zero_sketchCond_surfz_HYdit_e5000_valiation \
    --padding zero  # --save_denoising [optional]
```

**使用flow matching的scheduler (参考)：**
```bash
cd /data/lsr/code/style3d_gen
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed --device cuda --lr 5e-5\
    --list data_process/data_lists/stylexd_data_split_reso_256_Q1Q2Q4.pkl  --option surfz --denoiser_type hunyuan_dit \
    --surfvae log/stylexdQ1Q2Q4_vae_surf_256_xyz_mask_unet6_latent_16_16_1/ckpts/vae_e0850.pt \
    --cache_dir log/stylexdQ1Q2Q4_vae_surf_256_xyz_mask_unet6_latent_16_16_1/cache/vae_e0850_stylexdQ1Q2Q4_surfz_HYdit_Layer_10_12_emb768_xyz_mask_pad_zero_sketchCond_latent_16_16_1_scheduler_HY_FMED_shift3/encoder_mode \
    --expr stylexdQ1Q2Q4_surfz_HYdit_Layer_5_15_emb768_xyz_mask_pad_zero_sketchCond_latent_16_16_1_scheduler_HY_FMED_shift5_lr5e-5 --train_nepoch 100000 --test_nepoch 200 --save_nepoch 500 \
    --batch_size 100 --chunksize -1 --padding zero --bbox_scaled 1.0 --z_scaled 1.0 \
    --block_dims 16 32 32 64 64 --latent_channels 1 --max_face 32 --sample_mode mode \
    --embed_dim 768 --num_layer 5 15 \
    --scheduler HY_FMED --scheduler_shift 5 \
    --sketch_encoder LAION2B --sketch_feature_dir /data/AIGP/feature_laion2b \
    --data_fields surf_ncs surf_mask surf_bbox_wcs surf_uv_bbox_wcs sketch_feature
```
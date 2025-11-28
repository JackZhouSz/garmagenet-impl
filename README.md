## GarmageNet: A Multimodal Generative Framework for Sewing Pattern Design and Generic Garment Modeling

[![arXiv](https://img.shields.io/badge/📃-arXiv%20-red.svg)](https://arxiv.org/abs/2504.01483)
[![Github](https://img.shields.io/badge/github-Homepage-blue?logo=github)](https://style3d.github.io/garmagenet/)
[![HuggingFace](https://img.shields.io/badge/huggingface-Dataset-yellow?logo=huggingface)](https://huggingface.co/datasets/Style3D/GarmageSet)

\\[$\text{Siran Li}^{1,2,*}$]()\\
[$\text{Ruiyang Liu}^{4,2,*†}$](https://github.com/walnut-REE?tab=overview&from=2021-12-01&to=2021-12-31) 
[$\text{Chen Liu}^{4,2,*}$]()
[$\text{Zhendong Wang}^{2}$](https://wangzhendong619.github.io/)
[$\text{Gaofeng He}^{2}$]()
[$\text{YongLu Li}^{2}$](https://dirtyharrylyl.github.io/)
[$\text{Xiaogang Jin}^{2}$](http://www.cad.zju.edu.cn/home/jin/)
[$\text{Huamin Wang}^{2}$](https://wanghmin.github.io/)

![](resources/assests/teaser.jpg)
> We introduce GarmageNet, a novel uniform generation framework capable of producing complex and simulation-ready garments from various input modalities, including text prompts, sketch images, raw sewing patterns, or unstructured point clouds.

![](resources/assests/overview.png)
> The image above is an overview of GarmageNet. This project mainly focuses on the parts from **(a)** to **(b)**.

## Requirements

### Environment (Tested)
- Linux
- Python 3.10
- CUDA 11.8 
- PyTorch 2.2

### Dependencies

Install PyTorch and other dependencies:
```
python3.10 -m venv .venv/garmagenet
source .venv/garmagenet/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install chamferdist
```

## Data Preparation

Download GarmageSet from [HuggingFace](https://huggingface.co/datasets/Style3D/GarmageSet/tree/main)

Prepare GarmageNet`s training data (Garmage):

``` bash
python data_process/process_garmage.py \
    -i <garmageset-root>/raw \
    -o <garmageset-root>/garmages
```

Prepare datalist:
``` bash
python data_process/prepare_data_list.py \
    --garmage_dir <garmageset-root>/garmages \
    --output_dir <garmageset-root>/datalist
```

## Training

### Train VAE
Firstly, train the VAE encoder to compress Garmages. By default, all Garmages in the dataset have a resolution of $256\times256$. Each garment is represented as a set of per-panel Garmages, forming a tensor of shape $N\times256\times256\times C$, where $C$ depends on the desired encoded fields. 
For instance, the simplest Garmage has four channels: the first three encode geometric positions, while the last (alpha) outlines the sewing pattern. 
**For example:**

```bash
python src/vae.py --data <garmageset-root>/garmages --use_data_root \
    --list <garmageset-root>/datalist/garmageset_split_9_1.pkl \
    --expr garmagenet_vae_surf_256_xyz_mask_unet6_latent_1 \
    --batch_size 64 --block_dims 16 32 32 64 64 128 --latent_channels 1 \
    --test_nepoch 10 --save_nepoch 50 --train_nepoch 2000 \
    --data_fields surf_ncs surf_mask --chunksize 512
```

### Train Diffusion Generation
Based on the learned Garmage latent space, 
we train model to map random samples from the standard normal distribution $\epsilon ∽ N (0, 1)$ to valid Garmages.

#### Unconditional generation model training:

```bash
python src/ldm.py --data <garmageset-root>/garmages --use_data_root \
    --list <datalist-path> --option onestage_gen \
    --surfvae <vae-checkpoint-path> \
    --cache_dir log/garmagenet_vae_surf_256_xyz_mask_unet6_latent_1/cache/Onestage_xyz_mask_uncond/encoder_mode \
    --expr Onestage_xyz_mask_pad_zero_uncond \
    --train_nepoch 300000 --test_nepoch 200 --save_nepoch 10000 --batch_size 1230 --chunksize -1 \
    --padding zero --bbox_scaled 1.0 --z_scaled 1.0 \
    --block_dims 16 32 32 64 64 128 --latent_channels 1 --max_face 32 \
    --embed_dim 768 --pos_dim -1 \
    --data_fields surf_ncs surf_mask surf_bbox_wcs surf_uv_bbox_wcs
    --gpu 0
```

#### **Pointcloud condition** generation model training:

Prepare pointcloud sampling (surface uniform sampling).

```bash
python data_process/prepare_pc_cond_sample.py \
	--dataset_folder <garmageset-root>/raw \
	--pc_output_folder <garmageset-root>/pc_cond_sample_uniform
```

Run training.

```bash
python src/ldm.py --data /data/AIGP/GarmageSet_Opensource/garmages --use_data_root \
    --list /data/AIGP/GarmageSet_Opensource/datalist/garmageset_split_9_1.pkl --option onestage_gen \
    --surfvae log/stylexd_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e0800.pt \
    --cache_dir log/garmagenet_vae_surf_256_xyz_mask_unet6_latent_1/cache/Onestage_xyz_mask_pccond/encoder_mode \
    --expr Onestage_xyz_mask_pad_zero_pccond \
    --train_nepoch 300000 --test_nepoch 200 --save_nepoch 10000 --batch_size 1230 --chunksize -1 \
    --padding zero --bbox_scaled 1.0 --z_scaled 1.0 \
    --block_dims 16 32 32 64 64 128 --latent_channels 1 --max_face 32 \
    --embed_dim 768 --num_layer 12 --pos_dim -1 --dropout 0.1 \
    --pointcloud_encoder POINT_E --pointcloud_sampled_dir /data/AIGP/GarmageSet_Opensource/pc_cond_sample_uniform \
    --data_fields surf_ncs surf_mask surf_bbox_wcs surf_uv_bbox_wcs pointcloud_feature \
    --gpu 0
```

#### **Sketch condition** generation model training:

Prepare sketch feature. 

```bash
python data_process/prepare_sketch_feature_radiov2.5h.py \
	--root_dir <garmageset-root>/images \
	--output_dir <garmageset-root>/feature_radio_v2.5-h
```

Run training.

```bash
python src/ldm.py --data <garmageset-root>/garmages --use_data_root \
    --list <datalist-path> --option onestage_gen \
    --surfvae <vae-checkpoint-path> \
    --cache_dir log/garmagenet_vae_surf_256_xyz_mask_unet6_latent_1/cache/Onestage_xyz_mask_sketchCond_radio_v2.5-h/encoder_mode \
    --expr Onestage_xyz_mask_pad_zero_sketchCond_radio_v2.5-h \
    --train_nepoch 200000 --test_nepoch 200 --save_nepoch 10000 --batch_size 1230 --chunksize -1 \
    --padding zero --bbox_scaled 1.0 --z_scaled 1.0 \
    --block_dims 16 32 32 64 64 128 --latent_channels 1 --max_face 32 \
    --embed_dim 768 --num_layer 12 --pos_dim -1 --dropout 0.1 \
    --sketch_encoder RADIO_V2.5-H --sketch_feature_dir <garmageset-root>/feature_radio_v2.5-h \
    --data_fields surf_ncs surf_mask surf_bbox_wcs surf_uv_bbox_wcs sketch_feature \
    --gpu 0
```

## Inference 

Run inference to generate garmage with condition in validation set (saved in the cache).

```bash
python src/experiments/batch_inference_onestage/batch_inference_onestage.py \
	--vae <vae-checkpoint-path> \
	--onestage_gen <one-stage-model-ckpt-path> \
	--cache <cache-path-for-inference> \
	--sketch_encoder RADIO_V2.5-H
	--output generated/
	--padding zero
	--block_dims 16 32 32 64 64 128
	--img_channels 4
	--pos_dim -1
	--garmage_data_fields surf_ncs surf_mask
	--latent_data_fields latent64 bbox3d scale2d
```
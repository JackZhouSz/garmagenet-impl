<h2 align="center">
<img src="assets/images/logo.png" width="35%" align="center"> 
<br/>
<x-small>A Multimodal Generative Framework for Sewing Pattern Design and Generic Garment Modeling</x-small>
</h2>

<p align="center">
    <a href="">Siran Li</a><sup>*</sup>,
    <a href="https://github.com/walnut-REE">Ruiyang Liu</a><sup>*&dagger;</sup>,
    <a href="">Chen Liu</a><sup>*</sup>,
    <a href="">Zhendong Wang</a>,
    <a href="">Gaofeng He</a>,
    <a href="https://dirtyharrylyl.github.io/">Yong-Lu Li</a>,
    <a href="http://www.cad.zju.edu.cn/home/jin/">Xiaogang Jin</a>,
    <a href="https://wanghmin.github.io/">Huamin Wang</a>
</p>

<p align="center">
<a href="https://arxiv.org/abs/2504.01483"><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
<a href='https://style3d.github.io/garmagenet'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white' alt='Project Page'></a>
<a href='https://huggingface.co/spaces'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live_Demo-blue'></a>
</p>
<p align="center"><img src="assets/images/teaser.png" width="100%"></p>

> **GarmageNet** is a unified generative framework that automates the creation of 2D sewing patterns, the construction of sewing relationships, and the synthesis of 3D garment initializations compatible with physics-based simulation. Leveraging Garmage (a structured geometry image representation), it uses a latent-diffusion transformer to synthesize panels and GarmageJigsaw to predict point-to-point stitching, effectively closing the gap between 2D patterns and 3D shapes.



## 💫 Updates

- **[November 18, 2025]** First release of [GarmageSet](https://huggingface.co/datasets/Style3D/GarmageSet) dataset 🥳

- **[January 05, 2025]** Release GarmageNet training code.



## 🔨 Installation

**Tested Environment:** `Ubuntu 22.04` + `CUDA 11.8` + `Python 3.10` + `PyTorch 2.6`

Clone the repo:

```bash
git clone https://github.com/Style3D/garmagenet-impl.git garmagenet
cd garmagenet
```

Install PyTorch and other dependencies:
```
python3.10 -m venv .venv/garmagenet
source .venv/garmagenet/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install chamferdist
```



## 🎡 GarmageNet Training

### Data Preparation

Download GarmageSet raw geometry from [HuggingFace](https://huggingface.co/datasets/Style3D/GarmageSet), and prepare training Garmages with:

``` bash
# Render Garmages from raw OBJ
python data_process/process_garmage.py \
    -i <garmageset-root>/raw \
    -o <garmageset-root>/garmages

# Prepare datalist
python data_process/prepare_data_list.py \
    --garmage_dir <garmageset-root>/garmages \
    --output_dir <garmageset-root>/datalist
```

### Train Geometry Encoding VAE
By default, dataset Garmages have a resolution of $`256\times 256`$. Each garment is represented as a tensor of shape $`N\times 256 \times 256 \times C`$, where $`N`$ denotes the number of panels and $`C`$ the channel depth. The standard configuration ($C=4$) consists of three channels for geometric positions and one alpha channel defining the sewing pattern outline. To train the VAE:

```bash
python src/vae.py --data <garmageset-root>/garmages --use_data_root \
    --list <datalist-path> \
    --expr garmagenet_vae_surf_256_xyz_mask_unet6_latent_1 \
    --batch_size 64 --block_dims 16 32 32 64 64 128 --latent_channels 1 \
    --test_nepoch 10 --save_nepoch 50 --train_nepoch 2000 \
    --data_fields surf_ncs surf_mask --chunksize 512
```

### Train Diffusion Generator

**Unconditional** generation:

```bash
python src/ldm.py \
    --data <garmageset-root>/garmages --use_data_root \
    --list <datalist-path> --option garmagenet --lr 5e-4 \
    --surfvae <vae-checkpoint-path> \
    --cache_dir log/garmagenet_vae_surf_256_xyz_mask_unet6_latent_1/cache/GarmageNet_xyz_mask_uncond/encoder_mode \
    --expr GarmageNet_xyz_mask_pad_zero_uncond \
    --train_nepoch 200000 --test_nepoch 200 --save_nepoch 10000 --batch_size 1230 --chunksize -1 \
    --padding zero --bbox_scaled 1.0 --z_scaled 1.0 \
    --block_dims 16 32 32 64 64 128 --latent_channels 1 --max_face 32 \
    --embed_dim 768 \
    --data_fields surf_ncs surf_mask surf_bbox_wcs surf_uv_bbox_wcs \
    --gpu 0
```

**Text prompt** conditioned generation:

```bash
python src/ldm.py --data <garmageset-root>/garmages --use_data_root \
    --list <datalist-path> --option garmagenet \
    --surfvae <vae-checkpoint-path> \
    --cache_dir log/garmagenet_vae_surf_256_xyz_mask_unet6_latent_1/cache/GarmageNet_xyz_mask_caption_cond/encoder_mode \
    --expr GarmageNet_xyz_mask_pad_zero_caption_cond \
    --train_nepoch 200000 --test_nepoch 200 --save_nepoch 10000 --batch_size 1230 --chunksize -1 \
    --padding zero --bbox_scaled 1.0 --z_scaled 1.0 \
    --block_dims 16 32 32 64 64 128 --latent_channels 1 --max_face 32 \
    --embed_dim 768 --num_layer 12 --dropout 0.1 \
    --text_encoder CLIP \
    --data_fields surf_ncs surf_mask surf_bbox_wcs surf_uv_bbox_wcs caption \
    --gpu 0
```

**Pointcloud** conditioned generation:

```bash
# Prepare pointcloud sampling (surface uniform sampling).
python data_process/prepare_pc_cond_sample.py \
	--dataset_folder <garmageset-root>/raw \
	--pc_output_folder <garmageset-root>/pc_cond_sample_uniform
```

```bash
# Run training
python src/ldm.py --data <garmageset-root>/garmages --use_data_root \
    --list <datalist-path> --option garmagenet \
    --surfvae <vae-checkpoint-path> \
    --cache_dir log/garmagenet_vae_surf_256_xyz_mask_unet6_latent_1/cache/GarmageNet_xyz_mask_pccond/encoder_mode \
    --expr GarmageNet_xyz_mask_pad_zero_pccond \
    --train_nepoch 200000 --test_nepoch 200 --save_nepoch 10000 --batch_size 1230 --chunksize -1 \
    --padding zero --bbox_scaled 1.0 --z_scaled 1.0 \
    --block_dims 16 32 32 64 64 128 --latent_channels 1 --max_face 32 \
    --embed_dim 768 --num_layer 12 --dropout 0.1 \
    --pointcloud_encoder POINT_E --pointcloud_sampled_dir /data/AIGP/GarmageSet_Opensource/pc_cond_sample_uniform \
    --data_fields surf_ncs surf_mask surf_bbox_wcs surf_uv_bbox_wcs pointcloud_feature \
    --gpu 0
```

**Line-art sketch** conditioned generation:

```bash
# Prepare sketch feature. 
python data_process/prepare_sketch_feature_vit.py  \
	--root_dir <garmageset-root>/images \
	--output_dir <garmageset-root>/feature_laion2b
```

```bash
# Run training
python src/ldm.py \
     --data <garmageset-root>/garmages --use_data_root \
    --list <datalist-path> --option garmagenet \
    --surfvae <vae-checkpoint-path> \
    --cache_dir log/garmagenet_vae_surf_256_xyz_mask_unet6_latent_1/cache/GarmageNet_xyz_mask_sketchCond_laion2b/encoder_mode \
    --expr GarmageNet_xyz_mask_pad_zero_sketchCond_laion2b \
    --train_nepoch 200000 --test_nepoch 200 --save_nepoch 10000 --batch_size 1230 --chunksize -1 \
    --padding zero --bbox_scaled 1.0 --z_scaled 1.0 \
    --block_dims 16 32 32 64 64 128 --latent_channels 1 --max_face 32 \
    --sketch_encoder LAION2B --sketch_feature_dir <garmageset-root>/feature_laion2b \
    --condition_type spatial --feature_kwd 0 \
    --data_fields surf_ncs surf_mask surf_bbox_wcs surf_uv_bbox_wcs sketch_feature \
    --gpu 0
```

## 🍭 Generate Garmages from Pre-trained Checkpoints 

**Unconditional** generation:

```bash
python src/experiments/batch_inference/batch_inference.py \
	--vae <vae-checkpoint-path> \
	--garmagenet <GarmageNet-model-ckpt-path> \
	--cache log/garmagenet_vae_surf_256_xyz_mask_unet6_latent_1/cache/GarmageNet_xyz_mask_uncond/encoder_mode/garmagenet_validate.pkl \
	--output generated/uncond \
	--padding zero \
	--block_dims 16 32 32 64 64 128 \
	--img_channels 4 \
	--garmage_data_fields surf_ncs surf_mask \
	--latent_data_fields latent64 bbox3d scale2d
```

Generate garmage with **text prompts**:

```bash
python src/experiments/batch_inference/batch_inference.py \
	--vae <vae-checkpoint-path> \
	--garmagenet <GarmageNet-model-ckpt-path> \
	--cache log/garmagenet_vae_surf_256_xyz_mask_unet6_latent_1/cache/GarmageNet_xyz_mask_caption_cond/encoder_mode/garmagenet_validate.pkl \
	--text_encoder CLIP \
	--output generated/caption_cond \
	--padding zero \
	--block_dims 16 32 32 64 64 128 \
	--img_channels 4 \
	--garmage_data_fields surf_ncs surf_mask \
	--latent_data_fields latent64
```

Generate garmage with **unstructured pointclouds**:

```bash
python src/experiments/batch_inference/batch_inference.py \
	--vae <vae-checkpoint-path> \
	--garmagenet <GarmageNet-model-ckpt-path> \
	--cache log/garmagenet_vae_surf_256_xyz_mask_unet6_latent_1/cache/GarmageNet_xyz_mask_pccond/encoder_mode/garmagenet_validate.pkl \
	--pointcloud_encoder POINT_E \
	--output generated/pc_cond_uniform \
	--padding zero \
	--block_dims 16 32 32 64 64 128 \
	--img_channels 4 \
	--garmage_data_fields surf_ncs surf_mask \
	--latent_data_fields latent64
```

Generate garmage with **line-art sketches** (prefer front-view):

```bash
python src/experiments/batch_inference/batch_inference.py \
	--vae <vae-checkpoint-path> \
	--garmagenet <GarmageNet-model-ckpt-path> \
	--cache log/garmagenet_vae_surf_256_xyz_mask_unet6_latent_1/cache/GarmageNet_xyz_mask_sketchCond_laion2b/encoder_mode/garmagenet_validate.pkl \
	--sketch_encoder LAION2B \
	--output generated/sketch_cond \
	--padding zero \
	--block_dims 16 32 32 64 64 128 \
	--img_channels 4 \
	--garmage_data_fields surf_ncs surf_mask \
	--latent_data_fields latent64
```



## ⭐ Acknowledgements

We extend our sincere gratitude to the following open-source projects and research initiatives, whose contributions laid the foundation for GarmageNet:

- [**BrepGen**](https://github.com/samxuxiang/BrepGen)
- [**Jigsaw**](https://github.com/Jiaxin-Lu/Jigsaw)

We are also deeply indebted to the broader research community for their pioneering exploration and invaluable insights into the field of garment modeling.

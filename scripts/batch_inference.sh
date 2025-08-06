# 之前vae 4200那一版，稍微改了改 ===
cd /data/lsr/code/style3d_gen
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/experiments/batch_inference/batch_inference.py \
    --vae log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e0800.pt \
    --surfpos log/stylexdQ1Q2Q4_surfpos_xyzuv_pad_zero_sketchCond/ckpts/surfpos_e59000.pt \
    --surfz log/stylexdQ1Q2Q4_surfz_xyzuv_pad_zero_sketchCond_finetune_vae_e4200/ckpts/surfz_e250000.pt \
    --cache log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e0800_sketchCond_Q124/encoder_mode/surfz_validate.pkl \
    --use_original_pos \
    --sketch_encoder LAION2B \
    --output generated/xyzuv_pad_zero_sketchCond_surfz_e250000_vae_e0800 \
    --padding zero


# 2025_07_07 使用Hunyuan2.0 DIT做ldm ===
# batch inference Q124 HYdit sketchCond
# run on 188
cd /data/lsr/code/style3d_gen
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/experiments/batch_inference/batch_inference.py \
    --vae log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e4200.pt \
    --surfpos log/stylexdQ1Q2Q4_surfpos_xyzuv_pad_zero_sketchCond/ckpts/surfpos_e59000.pt \
    --surfz log/stylexdQ1Q2Q4_surfz_HYdit_xyzuv_pad_zero_sketchCond/ckpts/surfz_e5000.pt --surfz_type 'hunyuan_dit' \
    --cache log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e0800_sketchCond_Q124/encoder_mode/surfz_validate.pkl \
    --use_original_pos \
    --sketch_encoder LAION2B \
    --output generated/xyzuv_pad_zero_sketchCond_surfz_HYdit_e5000_valiation \
    --padding zero
#    --save_denoising

python src/experiments/batch_inference/batch_inference.py \
    --vae log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e4200.pt \
    --surfpos log/stylexdQ1Q2Q4_surfpos_xyzuv_pad_zero_sketchCond/ckpts/surfpos_e59000.pt \
    --surfz log/stylexdQ1Q2Q4_surfz_xyzuv_pad_zero_sketchCond_finetune_vae_e4200/ckpts/surfz_e250000.pt \
    --cache log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e0800_sketchCond_Q124/encoder_mode/surfz_train.pkl \
    --sketch_encoder LAION2B \
    --output generated/xyzuv_pad_zero_sketchCond_surfz_e250000_train \
    --padding zero



# 2025_06_26 做Supplementary素材 ===
# 我重新训练了VAE，在新的VAE上微调了Sketch、PC conditiond 的SurfZ

# batch inference Q124 sketchCond
# run on 190
cd /data/lsr/code/style3d_gen
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/experiments/batch_inference/batch_inference.py \
    --vae log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e4200.pt \
    --surfpos log/stylexdQ1Q2Q4_surfpos_xyzuv_pad_zero_sketchCond/ckpts/surfpos_e59000.pt \
    --surfz log/stylexdQ1Q2Q4_surfz_xyzuv_pad_zero_sketchCond_finetune_vae_e4200/ckpts/surfz_e250000.pt \
    --cache log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e0800_sketchCond_Q124/encoder_mode/surfz_validate.pkl \
    --use_original_pos \
    --sketch_encoder LAION2B \
    --output generated/xyzuv_pad_zero_sketchCond_surfz_e250000_save_denoising \
    --padding zero
#    --save_denoising


# batch inference Q124 pcCond
# run on 188
cd /data/lsr/code/style3d_gen
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/experiments/batch_inference/batch_inference.py \
    --vae log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e4200.pt \
    --surfpos log/stylexdQ1Q2Q4_surfpos_xyzuv_pad_zero_pcCond/ckpts/surfpos_e93000.pt \
    --surfz log/stylexdQ1Q2Q4_surfz_xyzuv_pad_zero_pcCond_finetune_vae_e4200/ckpts/surfz_e330000.pt \
    --cache log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e0800_pcCond_Q124/encoder_mode/surfz_validate.pkl \
    --use_original_pos \
    --pointcloud_encoder POINT_E \
    --output generated/xyzuv_pad_zero_pcCond_surfz_e330000_save_denoising \
    --padding zero
#    --save_denoising \
#    --device cuda:1





# 2025_06_11 伙伴大会素材 ===
# batch inference Q124 sketchCond
# run on 190
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/experiments/batch_inference/batch_inference.py \
    --vae log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e0800.pt \
    --surfpos log/stylexdQ1Q2Q4_surfpos_xyzuv_pad_zero_sketchCond/ckpts/surfpos_e59000.pt \
    --surfz log/stylexdQ1Q2Q4_surfz_xyzuv_pad_zero_sketchCond/ckpts/surfz_e150000.pt \
    --cache log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e0800_sketchCond_Q124/encoder_mode/surfz_validate.pkl \
    --sketch_encoder LAION2B \
    --output generated/xyzuv_pad_zero_sketchCond_surfz_e_e150000_save_denoising \
    --padding zero \
    --save_denoising
# batch inference Q124 pcCond
# run on 188
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/experiments/batch_inference/batch_inference.py \
    --vae log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e0800.pt \
    --surfpos log/stylexdQ1Q2Q4_surfpos_xyzuv_pad_zero_pcCond/ckpts/surfpos_e93000.pt \
    --surfz log/stylexdQ1Q2Q4_surfz_xyzuv_pad_zero_pcCond/ckpts/surfz_e200000.pt \
    --cache log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e0800_pcCond_Q124/encoder_mode/surfz_validate.pkl \
    --pointcloud_encoder POINT_E \
    --output generated/xyzuv_pad_zero_pcCond_surfz_e200000_save_denoising \
    --padding zero \
    --save_denoising \
    --device cuda:1
# batch inference Q12 TextCond
# run on 187
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/experiments/batch_inference/batch_inference.py \
  --vae /data/lsr/models/style3d_gen/surf_vae/stylexd_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e800.pt \
  --surfpos /data/lsr/models/style3d_gen/surf_pos/stylexd_surfpos_xyzuv_pad_zero_cond_clip/ckpts/surfpos_e30000.pt \
  --surfz /data/lsr/models/style3d_gen/surf_z/stylexd_surfz_xyzuv_mask_latent1_mode_with_caption/ckpts/surfz_e90000.pt \
  --cache /data/lsr/models/style3d_gen/surf_vae/stylexd_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e800/encoder_mode/surfz_validate.pkl \
  --text_encoder CLIP \
  --output generated/surfz_pos_padzero_textcond_e90000_save_denoising \
  --padding zero \
  --save_denoising



# batch inference Q12 TextCond
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/experiment/batch_inference/batch_inference.py \
  --vae /data/lsr/models/style3d_gen/surf_vae/stylexd_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e800.pt \
  --surfpos /data/lsr/models/style3d_gen/surf_pos/stylexd_surfpos_xyzuv_pad_zero_cond_clip/ckpts/surfpos_e30000.pt \
  --surfz /data/lsr/models/style3d_gen/surf_z/stylexd_surfz_xyzuv_mask_latent1_mode_with_caption/ckpts/surfz_e90000.pt \
  --cache /data/lsr/models/style3d_gen/surf_vae/stylexd_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e800/encoder_mode/surfz_validate.pkl \
  --use_original_pos \
  --text_encoder CLIP \
  --output generated/surfz_pos_padzero_textcond_e90000 \
  --padding zero



## 2025_06_04 version ===
# batch inference Q124 pcCond
# run on 188
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/experiments/batch_inference/batch_inference.py \
    --vae log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e0800.pt \
    --surfpos log/stylexdQ1Q2Q4_surfpos_xyzuv_pad_zero_pcCond/ckpts/surfpos_e93000.pt \
    --surfz log/stylexdQ1Q2Q4_surfz_xyzuv_pad_zero_pcCond/ckpts/surfz_e200000.pt \
    --cache log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e0800_pcCond_Q124/encoder_mode/surfz_validate.pkl \
    --use_original_pos \
    --pointcloud_encoder POINT_E \
    --output generated/xyzuv_pad_zero_pcCond_surfz_e200000 \
    --padding zero  # --save_denoising [optional]
# batch inference Q124 sketchCond
# run on 190
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/experiments/batch_inference/batch_inference.py \
    --vae log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e0800.pt \
    --surfpos log/stylexdQ1Q2Q4_surfpos_xyzuv_pad_zero_sketchCond/ckpts/surfpos_e59000.pt \
    --surfz log/stylexdQ1Q2Q4_surfz_xyzuv_pad_zero_sketchCond/ckpts/surfz_e150000.pt \
    --cache log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e0800_sketchCond_Q124/encoder_mode/surfz_validate.pkl \
    --use_original_pos \
    --sketch_encoder LAION2B \
    --output generated/xyzuv_pad_zero_sketchCond_surfz_e_e150000 \
    --padding zero  # --save_denoising [optional]


## 2025_05_20 version, ckpt has some problem ! ===
## batch inference Q124 pcCond
## run on 188
#export PYTHONPATH=/data/lsr/code/style3d_gen
#python _LSR/experiments/batch_inference/batch_inference.py \
#    --vae log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e0800.pt \
#    --surfpos log/stylexdQ1Q2Q4_surfpos_xyzuv_pad_zero_pcCond/ckpts/surfpos_e93000.pt \
#    --surfz log/stylexdQ1Q2Q4_surfz_xyzuv_pad_zero_pcCond/ckpts/surfz_e95000.pt \
#    --cache log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e0800_pcCond_Q124/encoder_mode/surfz_validate.pkl \
#    --use_original_pos \
#    --pointcloud_encoder POINT_E \
#    --output generated/xyzuv_pad_zero_pcCond_surfz_e95000 \
#    --padding zero
## batch inference Q124 sketchCond
## run on 190
#export PYTHONPATH=/data/lsr/code/style3d_gen
#python _LSR/experiments/batch_inference/batch_inference.py \
#    --vae log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e0800.pt \
#    --surfpos log/stylexdQ1Q2Q4_surfpos_xyzuv_pad_zero_sketchCond/ckpts/surfpos_e59000.pt \
#    --surfz log/stylexdQ1Q2Q4_surfz_xyzuv_pad_zero_sketchCond/ckpts/surfz_e45000.pt \
#    --cache log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e0800_sketchCond_Q124/encoder_mode/surfz_validate.pkl \
#    --use_original_pos \
#    --sketch_encoder LAION2B \
#    --output generated/xyzuv_pad_zero_sketchCond_surfz_e_e45000 \
#    --padding zero



# batch inference with Q124ckpt xyzuv zero padding uncond
# run on 187
export PYTHONPATH=/data/lsr/code/style3d_gen
python _LSR/experiments/batch_inference/batch_inference.py \
        --vae /data/lsr/models/style3d_gen/surf_vae/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e0800.pt \
        --surfpos /data/lsr/models/style3d_gen/surf_pos/stylexdQ1Q2Q4_surfpos_xyzuv_pad_zero_uncond/ckpts/surfpos_e32000.pt \
        --surfz /data/lsr/models/style3d_gen/surf_z/stylexdQ1Q2Q4_surfz_xyzuv_pad_zero_uncond/ckpts/surfz_e380000.pt \
        --cache /data/lsr/models/style3d_gen/surf_vae/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e0800/encoder_mode/surfpos_validate.pkl \
        --use_original_pos \
        --output generated/surfz_e380000 \
        --padding zero
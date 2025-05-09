
# batch inference with Q124ckpt xyzuv zero padding
# run on 187
export PYTHONPATH=/data/lsr/code/style3d_gen
python _LSR/experiment/batch_inference/batch_inference.py \
        --vae /data/lsr/models/style3d_gen/surf_vae/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e0800.pt \
        --surfpos /data/lsr/models/style3d_gen/surf_pos/stylexdQ1Q2Q4_surfpos_xyzuv_pad_zero_uncond/ckpts/surfpos_e32000.pt \
        --surfz /data/lsr/models/style3d_gen/surf_z/stylexdQ1Q2Q4_surfz_xyzuv_pad_zero_uncond/ckpts/surfz_e380000.pt \
        --cache /data/lsr/models/style3d_gen/surf_vae/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e0800/encoder_mode/surfpos_validate.pkl \
        --use_original_pos True \
        --output generated/surfz_e380000 \
        --padding zero
### Train the Latent Diffusion Model ###

# Radio-v2.5-h 测试泛化性能否提升 ===
# Z 190:2460 182:1230
cd /data/lsr/code/style3d_gen
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed \
    --list data_process/data_lists/stylexd_data_split_reso_256_Q1Q2Q4.pkl  --option surfz \
    --surfvae log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e0800.pt \
    --cache_dir log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e0800_sketchCond_radio_v2.5-h_Q124/encoder_mode \
    --expr stylexdQ1Q2Q4_surfz_xyzuv_pad_zero_sketchCond_radio_v2.5-h --train_nepoch 600000 --test_nepoch 200 --save_nepoch 20000 \
    --batch_size 2460 --chunksize -1 --padding zero --bbox_scaled 1.0 --z_scaled 1.0 \
    --block_dims 16 32 32 64 64 128 --latent_channels 1 --max_face 32 --sample_mode mode \
    --sketch_encoder RADIO_V2.5-H --sketch_feature_dir /data/AIGP/feature_radio_v2.5-h \
    --data_fields surf_ncs surf_uv_ncs surf_mask surf_bbox_wcs surf_uv_bbox_wcs sketch_feature

# POS 188:1230
cd /data/lsr/code/style3d_gen
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed \
    --list data_process/data_lists/stylexd_data_split_reso_256_Q1Q2Q4.pkl --option surfpos \
    --cache_dir log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e0800_sketchCond_radio_v2.5-h_Q124/encoder_mode \
    --padding zero \
    --expr stylexdQ1Q2Q4_surfpos_xyzuv_pad_zero_sketchCond_radio_v2.5-h --train_nepoch 600000 --test_nepoch 1000 --save_nepoch 20000 \
    --batch_size 1230 --max_face 32 --bbox_scaled 1.0 \
    --sketch_encoder RADIO_V2.5-H --sketch_feature_dir /data/AIGP/feature_radio_v2.5-h \
    --data_fields surf_bbox_wcs surf_uv_bbox_wcs sketch_feature --gpu 0





# Revision 多种采样方式的点云条件 ===
# zero padding xyz uv mask Q1-4 pcCond_uniform_nonuniform_fps
# POS Bsize 182:1230 190：2460
cd /data/lsr/code/style3d_gen
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed \
    --list data_process/data_lists/stylexd_data_split_reso_256_Q1Q2Q4.pkl --option surfpos \
    --cache_dir log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e0800_pcCond_multitype/encoder_mode \
    --padding zero \
    --expr stylexdQ1Q2Q4_surfpos_xyzuv_pad_zero_pcCond_multitype --train_nepoch 300000 --test_nepoch 1000 --save_nepoch 5000 \
    --batch_size 1230 --max_face 32 --bbox_scaled 1.0 \
    --pointcloud_encoder POINT_E --pointcloud_feature_dir /data/AIGP/pc_cond_sample_multitype/ \
    --data_fields surf_bbox_wcs surf_uv_bbox_wcs pointcloud_feature --gpu 0

# Z Bsize 188:1230
cd /data/lsr/code/style3d_gen
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed \
    --list data_process/data_lists/stylexd_data_split_reso_256_Q1Q2Q4.pkl  --option surfz \
    --surfvae log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e0800.pt \
    --cache_dir log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e0800_pcCond_multitype/encoder_mode \
    --expr stylexdQ1Q2Q4_surfz_xyzuv_pad_zero_pcCond_multitype --train_nepoch 300000 --test_nepoch 200 --save_nepoch 5000 \
    --batch_size 2460 --chunksize -1 --padding zero --bbox_scaled 1.0 --z_scaled 1.0 \
    --block_dims 16 32 32 64 64 128 --latent_channels 1 --max_face 32 --sample_mode mode \
    --pointcloud_encoder POINT_E --pointcloud_feature_dir /data/AIGP/pc_cond_sample_multitype/\
    --data_fields surf_ncs surf_uv_ncs surf_mask surf_bbox_wcs surf_uv_bbox_wcs pointcloud_feature


# Revision mesh表面均匀采样的点云条件 ===
# zero padding xyz uv mask Q1-4 pcCond_uniform
# POS Bsize 182:1230
cd /data/lsr/code/style3d_gen
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed \
    --list data_process/data_lists/stylexd_data_split_reso_256_Q1Q2Q4.pkl --option surfpos \
    --cache_dir log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e0800_pcCond_uniform/encoder_mode \
    --padding zero \
    --expr stylexdQ1Q2Q4_surfpos_xyzuv_pad_zero_pcCond_uniform --train_nepoch 100000 --test_nepoch 1000 --save_nepoch 1000 \
    --batch_size 1640 --max_face 32 --bbox_scaled 1.0 \
    --pointcloud_encoder POINT_E --pointcloud_sampled_dir /data/AIGP/pc_cond_sample/uniform_2048 \
    --data_fields surf_bbox_wcs surf_uv_bbox_wcs pointcloud_feature
# Z Bsize 188:1230
cd /data/lsr/code/style3d_gen
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed \
    --list data_process/data_lists/stylexd_data_split_reso_256_Q1Q2Q4.pkl  --option surfz \
    --surfvae log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e0800.pt \
    --cache_dir log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e0800_pcCond_uniform/encoder_mode \
    --expr stylexdQ1Q2Q4_surfz_xyzuv_pad_zero_pcCond_uniform --train_nepoch 100000 --test_nepoch 200 --save_nepoch 5000 \
    --batch_size 1230 --chunksize -1 --padding zero --bbox_scaled 1.0 --z_scaled 1.0 \
    --block_dims 16 32 32 64 64 128 --latent_channels 1 --max_face 32 --sample_mode mode \
    --pointcloud_encoder POINT_E --pointcloud_sampled_dir /data/AIGP/pc_cond_sample/uniform_2048 \
    --data_fields surf_ncs surf_uv_ncs surf_mask surf_bbox_wcs surf_uv_bbox_wcs pointcloud_feature



# DIT Z  scheduler 用 hunyuan 的 FlowMatchEulerDiscreteScheduler ===
cd /data/lsr/code/style3d_gen
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed --device cuda:1\
    --list data_process/data_lists/stylexd_data_split_reso_256_Q1Q2Q4.pkl  --option surfz --denoiser_type hunyuan_dit \
    --surfvae log/stylexdQ1Q2Q4_vae_surf_256_xyz_mask_unet6_latent_16_16_1/ckpts/vae_e0850.pt \
    --cache_dir log/stylexdQ1Q2Q4_vae_surf_256_xyz_mask_unet6_latent_16_16_1/cache/vae_e0850_stylexdQ1Q2Q4_surfz_HYdit_Layer_10_12_emb768_xyz_mask_pad_zero_sketchCond_latent_16_16_1_scheduler_HY_FMED_shift3/encoder_mode \
    --expr stylexdQ1Q2Q4_surfz_HYdit_Layer_10_12_emb768_xyz_mask_pad_zero_sketchCond_latent_16_16_1_scheduler_HY_FMED_shift3 --train_nepoch 100000 --test_nepoch 200 --save_nepoch 500 \
    --batch_size 100 --chunksize -1 --padding zero --bbox_scaled 1.0 --z_scaled 1.0 \
    --block_dims 16 32 32 64 64 --latent_channels 1 --max_face 32 --sample_mode mode \
    --embed_dim 768 --num_layer 10 12 \
    --scheduler HY_FMED --scheduler_shift 3 \
    --sketch_encoder LAION2B --sketch_feature_dir /data/AIGP/feature_laion2b \
    --data_fields surf_ncs surf_mask surf_bbox_wcs surf_uv_bbox_wcs sketch_feature


# Dit Pos 测试 (测试不同的padding) ===
# layer 2+6  embed_dim 384 zero padding
# 188: 2475
cd /data/lsr/code/style3d_gen
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed \
    --list data_process/data_lists/stylexd_data_split_reso_256_Q1Q2Q4.pkl  --option surfpos --denoiser_type hunyuan_dit \
    --cache_dir log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e0800_sketchCond_Q124/encoder_mode \
    --expr stylexdQ1Q2Q4_surfpos_HYdit_L2+6_emb384_pad_zero_sketchCond --train_nepoch 200000 --test_nepoch 1000 --save_nepoch 1000 \
    --batch_size 2475 --chunksize -1 --padding zero --max_face 32 \
    --embed_dim 384 --num_layer 2 6 \
    --sketch_encoder LAION2B --sketch_feature_dir /data/AIGP/feature_laion2b \
    --data_fields surf_bbox_wcs surf_uv_bbox_wcs sketch_feature

# layer 2+6  embed_dim 384 repeat padding
# 188: 1650
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




# 依旧是DIT的ldm，减少几层试试 ===
# SurfZ_HYdit Layer 2+4 latent256 zero padding xyz mask Q1-4 sketchCond(laion2b)
# Z Bsize 188:1650
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed \
    --list data_process/data_lists/stylexd_data_split_reso_256_Q1Q2Q4.pkl  --option surfz --denoiser_type hunyuan_dit \
    --surfvae log/stylexdQ1Q2Q4_vae_surf_256_xyz_mask_unet6_latent_16_16_1/ckpts/vae_e0850.pt \
    --cache_dir log/stylexdQ1Q2Q4_vae_surf_256_xyz_mask_unet6_latent_16_16_1/cache/vae_e0850_sketchCond_Q124_latent_16_16_1/encoder_mode \
    --expr stylexdQ1Q2Q4_surfz_HYdit_Layer_2_6_xyz_mask_pad_zero_sketchCond_latent_16_16_1 --train_nepoch 100000 --test_nepoch 200 --save_nepoch 1000 \
    --batch_size 1650 --chunksize -1 --padding zero --bbox_scaled 1.0 --z_scaled 1.0 \
    --block_dims 16 32 32 64 64 --latent_channels 1 --max_face 32 --sample_mode mode \
    --embed_dim 768 --num_layer 2 6 \
    --sketch_encoder LAION2B --sketch_feature_dir /data/AIGP/feature_laion2b \
    --data_fields surf_ncs surf_mask surf_bbox_wcs surf_uv_bbox_wcs sketch_feature



# 更换了 Hunyuan2.0 DIT 进行一些实验 ===
# SurfZ_HYdit zero padding xyz uv mask Q1-4 sketchCond(laion2b)
# Z Bsize 188:1650
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed \
    --list data_process/data_lists/stylexd_data_split_reso_256_Q1Q2Q4.pkl  --option surfz --denoiser_type hunyuan_dit \
    --surfvae log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e4200.pt \
    --cache_dir log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e0800_sketchCond_Q124/encoder_mode \
    --expr stylexdQ1Q2Q4_surfz_HYdit_xyzuv_pad_zero_sketchCond --train_nepoch 100000 --test_nepoch 200 --save_nepoch 1000 \
    --batch_size 1650 --chunksize -1 --padding zero --bbox_scaled 1.0 --z_scaled 1.0 \
    --block_dims 16 32 32 64 64 128 --latent_channels 1 --max_face 32 --sample_mode mode \
    --sketch_encoder LAION2B --sketch_feature_dir /data/AIGP/feature_laion2b \
    --data_fields surf_ncs surf_uv_ncs surf_mask surf_bbox_wcs surf_uv_bbox_wcs sketch_feature
# SurfZ_HYdit latent256 zero padding xyz mask Q1-4 sketchCond(laion2b)
# Z Bsize 188:1650
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed \
    --list data_process/data_lists/stylexd_data_split_reso_256_Q1Q2Q4.pkl  --option surfz --denoiser_type hunyuan_dit \
    --surfvae log/stylexdQ1Q2Q4_vae_surf_256_xyz_mask_unet6_latent_16_16_1/ckpts/vae_e0850.pt \
    --cache_dir log/stylexdQ1Q2Q4_vae_surf_256_xyz_mask_unet6_latent_16_16_1/cache/vae_e0850_sketchCond_Q124_latent_16_16_1/encoder_mode \
    --expr stylexdQ1Q2Q4_surfz_HYdit_xyz_pad_zero_sketchCond_latent_16_16_1 --train_nepoch 100000 --test_nepoch 200 --save_nepoch 1000 \
    --batch_size 1650 --chunksize -1 --padding zero --bbox_scaled 1.0 --z_scaled 1.0 \
    --block_dims 16 32 32 64 64 --latent_channels 1 --max_face 32 --sample_mode mode \
    --sketch_encoder LAION2B --sketch_feature_dir /data/AIGP/feature_laion2b \
    --data_fields surf_ncs surf_mask surf_bbox_wcs surf_uv_bbox_wcs sketch_feature
# SurfZ_HYdit latent256 embed1536 zero padding xyz mask Q1-4 sketchCond(laion2b) [TODO]
# Z Bsize 188:1650
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed \
    --list data_process/data_lists/stylexd_data_split_reso_256_Q1Q2Q4.pkl  --option surfz --denoiser_type hunyuan_dit \
    --surfvae log/stylexdQ1Q2Q4_vae_surf_256_xyz_mask_unet6_latent_16_16_1/ckpts/vae_e0850.pt \
    --cache_dir log/stylexdQ1Q2Q4_vae_surf_256_xyz_mask_unet6_latent_1/cache/vae_e0850_sketchCond_Q124_latent_16_16_1/encoder_mode \
    --expr stylexdQ1Q2Q4_surfz_HYdit_xyzuv_pad_zero_sketchCond_latent_16_16_1 --train_nepoch 100000 --test_nepoch 200 --save_nepoch 1000 \
    --batch_size 1650 --chunksize -1 --padding zero --bbox_scaled 1.0 --z_scaled 1.0 \
    --block_dims 16 32 32 64 64 --embed_dim 1536 --latent_channels 1 --max_face 32 --sample_mode mode \
    --sketch_encoder LAION2B --sketch_feature_dir /data/AIGP/feature_laion2b \
    --data_fields surf_ncs surf_mask surf_bbox_wcs surf_uv_bbox_wcs sketch_feature



# 对Q124的VAE进一步训练后，我在这个VAE上对之前训的两个SurfZ进行微调 ===
# zero padding xyz uv mask Q1-4 sketchCond(laion2b) finetune on 4200 ckpt ===
# Z Bsize 190:2500
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed \
    --list data_process/data_lists/stylexd_data_split_reso_256_Q1Q2Q4.pkl  --option surfz \
    --surfvae log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e4200.pt \
    --cache_dir log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e0800_sketchCond_Q124/encoder_mode \
    --expr stylexdQ1Q2Q4_surfz_xyzuv_pad_zero_sketchCond_finetune_vae_e4200 --train_nepoch 300000 --test_nepoch 200 --save_nepoch 5000 \
    --batch_size 2500 --chunksize -1 --padding zero --bbox_scaled 1.0 --z_scaled 1.0 \
    --block_dims 16 32 32 64 64 128 --latent_channels 1 --max_face 32 --sample_mode mode \
    --sketch_encoder LAION2B --sketch_feature_dir /data/AIGP/feature_laion2b \
    --data_fields surf_ncs surf_uv_ncs surf_mask surf_bbox_wcs surf_uv_bbox_wcs sketch_feature \
    --finetune --weight log/stylexdQ1Q2Q4_surfz_xyzuv_pad_zero_sketchCond/ckpts/surfz_e150000.pt
# zero padding xyz uv mask Q1-4 pcCond finetune on 4200 ckpt ===
# Z Bsize 190:2500
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed \
    --list data_process/data_lists/stylexd_data_split_reso_256_Q1Q2Q4.pkl  --option surfz \
    --surfvae log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e4200.pt \
    --cache_dir log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e0800_pcCond_Q124/encoder_mode \
    --expr stylexdQ1Q2Q4_surfz_xyzuv_pad_zero_pcCond_finetune_vae_e4200 --train_nepoch 300000 --test_nepoch 200 --save_nepoch 5000 \
    --batch_size 2500 --chunksize -1 --padding zero --bbox_scaled 1.0 --z_scaled 1.0 \
    --block_dims 16 32 32 64 64 128 --latent_channels 1 --max_face 32 --sample_mode mode \
    --pointcloud_encoder POINT_E \
    --data_fields surf_ncs surf_uv_ncs surf_mask surf_bbox_wcs surf_uv_bbox_wcs pointcloud_feature \
    --finetune --weight log/stylexdQ1Q2Q4_surfz_xyzuv_pad_zero_pcCond/ckpts/surfz_e200000.pt



# SiggraphAsia做数据用的模型===
# zero padding xyz uv mask Q1-4 sketchCond(RADIO_V2.5-G)
# Z Bsize 188:3420 190:5000
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed \
    --list data_process/data_lists/stylexd_data_split_reso_256_Q1Q2Q4.pkl  --option surfz \
    --surfvae log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e0800.pt \
    --cache_dir log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e0800_sketchCond_radio_v2.5-g_Q124/encoder_mode \
    --expr stylexdQ1Q2Q4_surfz_xyzuv_pad_zero_sketchCond_radio_v2.5-g_Q124 --train_nepoch 100000 --test_nepoch 200 --save_nepoch 5000 \
    --batch_size 5000 --chunksize -1 --padding zero --bbox_scaled 1.0 --z_scaled 1.0 \
    --block_dims 16 32 32 64 64 128 --latent_channels 1 --max_face 32 --sample_mode mode \
    --sketch_encoder RADIO_V2.5-G --sketch_feature_dir /data/AIGP/feature_radio_v2.5-g \
    --data_fields surf_ncs surf_uv_ncs surf_mask surf_bbox_wcs surf_uv_bbox_wcs sketch_feature
# zero padding xyz uv mask Q1-4 sketchCond(laion2b) ===
# POS Bsize 187:3420
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed \
    --list data_process/data_lists/stylexd_data_split_reso_256_Q1Q2Q4.pkl --option surfpos \
    --cache_dir log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e0800_sketchCond/encoder_mode \
    --padding zero \
    --expr stylexdQ1Q2Q4_surfpos_xyzuv_pad_zero_sketchCond --train_nepoch 100000 --test_nepoch 100 --save_nepoch 1000 \
    --batch_size 2500 --max_face 32 --bbox_scaled 1.0 \
    --sketch_encoder LAION2B --sketch_feature_dir /data/AIGP/feature_laion2b \
    --data_fields surf_bbox_wcs surf_uv_bbox_wcs sketch_feature
# Z Bsize 188:3420 190:5000
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
# zero padding xyz uv mask Q1-4 pcCond ===
# POS Bsize 187:3420
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed \
    --list data_process/data_lists/stylexd_data_split_reso_256_Q1Q2Q4.pkl --option surfpos \
    --cache_dir log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e0800_pcCond/encoder_mode \
    --padding zero \
    --expr stylexdQ1Q2Q4_surfpos_xyzuv_pad_zero_pcCond --train_nepoch 100000 --test_nepoch 100 --save_nepoch 1000 \
    --batch_size 3420 --max_face 32 --bbox_scaled 1.0 \
    --pointcloud_encoder POINT_E \
    --data_fields surf_bbox_wcs surf_uv_bbox_wcs pointcloud_feature
# Z Bsize 188:3420
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



# zero padding xyz uv mask Q1-4 unCond
# POS
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed \
    --list data_process/data_lists/stylexd_data_split_reso_256_Q1Q2Q4.pkl --option surfpos \
    --cache_dir log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e0800/encoder_mode \
    --padding zero \
    --expr stylexdQ1Q2Q4_surfpos_xyzuv_pad_zero_uncond --train_nepoch 100000 --test_nepoch 100 --save_nepoch 1000 \
    --batch_size 512 --max_face 32 --bbox_scaled 1.0 \
    --data_fields surf_bbox_wcs surf_uv_bbox_wcs
# Z
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed \
    --list data_process/data_lists/stylexd_data_split_reso_256_Q1Q2Q4.pkl --option surfz \
    --surfvae log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e0800.pt \
    --cache_dir log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e0800/encoder_mode \
    --expr stylexdQ1Q2Q4_surfz_xyzuv_pad_zero_uncond --train_nepoch 100000 --test_nepoch 200 --save_nepoch 5000 \
    --batch_size 2048 --chunksize -1 --padding zero --bbox_scaled 1.0 --z_scaled 1.0 \
    --block_dims 16 32 32 64 64 128 --latent_channels 1 --max_face 32 --sample_mode mode \
    --data_fields surf_ncs surf_uv_ncs surf_mask surf_bbox_wcs surf_uv_bbox_wcs
# Z resume training
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed \
     --list data_process/data_lists/stylexd_data_split_reso_256_Q1Q2Q4.pkl --option surfz \
     --surfvae log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e0800.pt \
     --cache_dir log/stylexdQ1Q2Q4_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e0800/encoder_mode \
     --expr stylexdQ1Q2Q4_surfz_xyzuv_pad_zero_uncond --train_nepoch 100000 --test_nepoch 200 --save_nepoch 5000 \
     --batch_size 5000 --chunksize -1 --padding zero --bbox_scaled 1.0 --z_scaled 1.0 \
     --block_dims 16 32 32 64 64 128 --latent_channels 1 --max_face 32 --sample_mode mode \
     --data_fields surf_ncs surf_uv_ncs surf_mask surf_bbox_wcs surf_uv_bbox_wcs \
     --finetune --weight /data/lsr/code/style3d_gen/log/stylexdQ1Q2Q4_surfz_xyzuv_pad_zero_uncond/ckpts/surfz_e10000.pt



# zero padding xyz uv mask Q1-2 pcCond ===
# POS  Bsize 199:3420
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed \
    --list data_process/data_lists/stylexd_data_split_reso_256.pkl --option surfpos \
    --cache_dir log/stylexd_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e0800_pcCond/encoder_mode \
    --padding zero \
    --expr stylexd_surfpos_xyzuv_pad_zero_pcCond --train_nepoch 100000 --test_nepoch 100 --save_nepoch 1000 \
    --batch_size 3420 --max_face 32 --bbox_scaled 1.0 \
    --pointcloud_encoder POINT_E \
    --data_fields surf_bbox_wcs surf_uv_bbox_wcs pointcloud_feature
# Z   Bsize 199:3420
export PYTHONPATH=/data/lsr/code/style3d_gen
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption/processed \
    --list data_process/data_lists/stylexd_data_split_reso_256.pkl --option surfz \
    --surfvae log/stylexd_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e0800.pt \
    --cache_dir log/stylexd_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e0800_pcCond/encoder_mode \
    --expr stylexd_surfz_xyzuv_pad_zero_pcCond --train_nepoch 100000 --test_nepoch 200 --save_nepoch 5000 \
    --batch_size 3420 --chunksize -1 --padding zero --bbox_scaled 1.0 --z_scaled 1.0 \
    --block_dims 16 32 32 64 64 128 --latent_channels 1 --max_face 32 --sample_mode mode \
    --pointcloud_encoder POINT_E \
    --data_fields surf_ncs surf_uv_ncs surf_mask surf_bbox_wcs surf_uv_bbox_wcs pointcloud_feature



# 256 lry ===
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
    --expr stylexd_surfpos_xyzuv_pad_repeat_cond_clip_debug_1 --train_nepoch 5 --test_nepoch 2 --save_nepoch 3 \
    --batch_size 4 --max_face 32 --bbox_scaled 1.0 \
    --data_fields surf_bbox_wcs surf_uv_bbox_wcs caption


### StyleXD - SurfZ
python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption \
    --list data_process/stylexd_data_split_reso_256.pkl --option surfz \
    --surfvae log/stylexd_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e800.pt \
    --cache_dir log/stylexd_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e800/encoder_mode \
    --expr stylexd_surfz_xyzuv_mask_latent1_mode_with_caption --train_nepoch 100000 --test_nepoch 200 --save_nepoch 5000 \
    --batch_size 2048 --chunksize -1 --padding zero --bbox_scaled 1.0 --z_scaled 1.0 --text_encoder CLIP \
    --block_dims 16 32 32 64 64 128 --latent_channels 1 --max_face 32 --sample_mode mode \
    --data_fields surf_ncs surf_uv_ncs surf_mask surf_bbox_wcs surf_uv_bbox_wcs caption


python src/ldm.py --data /data/AIGP/brep_reso_256_edge_snap_with_caption \
    --list data_process/stylexd_data_split_reso_256.pkl --option surfz \
    --surfvae log/stylexd_vae_surf_256_xyz_uv_mask_unet6_latent_1/ckpts/vae_e800.pt \
    --cache_dir log/stylexd_vae_surf_256_xyz_uv_mask_unet6_latent_1/cache/vae_e800/encoder_mode \
    --expr stylexd_surfz_xyzuv_mask_latent1_mode_with_caption_test --train_nepoch 10 --test_nepoch 5 --save_nepoch 10 \
    --batch_size 1024 --chunksize -1 --padding zero --bbox_scaled 1.0 --z_scaled 1.0 --text_encoder CLIP \
    --block_dims 16 32 32 64 64 128 --latent_channels 1 --max_face 32 --sample_mode mode \
    --data_fields surf_ncs surf_uv_ncs surf_mask surf_bbox_wcs surf_uv_bbox_wcs caption


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
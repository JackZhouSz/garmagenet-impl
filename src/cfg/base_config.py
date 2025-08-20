from yacs.config import CfgNode as CN


def get_cfg_main():
    cfg = CN()
    cfg.OUTPUT_DIR = "./output"
    cfg.DEVICE = "cuda"
    return cfg


def get_cfg_dataset():
    cfg = CN()
    cfg.DATASET = CN()
    cfg.DATASET.NAME = "default"
    cfg.DATASET.ROOT = "./data"
    cfg.DATASET.BATCH_SIZE = 32
    return cfg


def get_VAE_cfg(cfg_fp=None):
    cn = CN()
    cn.num_channels = 3
    cn.block_dims = [16, 32, 32, 64, 64, 128]
    cn.layers_per_block = 2
    cn.act_fn = 'silu'
    cn.latent_channels = 1
    cn.norm_num_groups = 8
    cn.sample_size = 256
    if cfg_fp is not None:
        cn.merge_from_file(cfg_fp)

    return cn


# def load_cfg(main_yaml, dataset_yaml, model_yaml):
#     cfg_main = get_cfg_main()
#     cfg_dataset = get_cfg_dataset()
#     cfg_model = get_cfg_model()
#
#     # 先合并默认结构
#     cfg = cfg_main.clone()
#     cfg.merge_from_other_cfg(cfg_dataset)
#     cfg.merge_from_other_cfg(cfg_model)
#
#     # 再从yaml文件加载实际配置
#     cfg.merge_from_file(dataset_yaml)
#     cfg.merge_from_file(model_yaml)
#     cfg.merge_from_file(main_yaml)
#
#     return cfg


if __name__ == "__main__":
    vae_cfg = get_VAE_cfg()
    vae_cfg.merge_from_file("/home/Ex1/ProjectFiles/Pycharm_MyPaperWork/style3d_gen/src/cfg/config/models/vae/VAE_256_C1_L8x8x1.yaml")
    a=1
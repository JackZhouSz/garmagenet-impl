import os
import argparse

from datasets.sxd import *
from trainer import *


def get_args_ldm():
    
    def _str2intlist(s): return list(map(int, s.split(',')))
    
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, default='sxd', choices=['sxd', 'brep'],
    #                     help='Dataset type, choose between [sxd, brep]')
    parser.add_argument('--data', type=str, default='data_process/deepcad_parsed', 
                        help='Path to data folder')
    # parser.add_argument('--use_data_root', action="store_true",
    #                     help='If data list store absolute path, don`t use this flag.')
    parser.add_argument('--list', type=str, default='data_process/deepcad_data_split_6bit.pkl', 
                        help='Path to data list')  
    parser.add_argument('--cache_dir', type=str, default=None, help='Path to cached data (with latents).')
    parser.add_argument('--surfvae', type=str, default='log/deepcad_surfvae/epoch_400.pt', 
                        help='Path to pretrained surface vae weights')

    # following 3 currently used to surfInpainting only
    parser.add_argument('--surfvae_config', type=str, default="src/cfg/config/models/vae/VAE_256_xyz_mask_L16x16x1.yaml",
                        help='Geo+Mask vae config fp')
    parser.add_argument('--surfvae_2', type=str, default="_LSR/experiment/test_dataset/test_ckpt/stylexdQ1Q2Q4_vae_surf_256_mask_unet6_latent_16_16_1/ckpts/vae_e0050.pt",
                        help='Mask vae weights fp')
    parser.add_argument('--surfvae_2_config', type=str, default="src/cfg/config/models/vae/VAE_256_mask_L16x16x1.yaml",
                        help='Mask vae config fp')

    parser.add_argument("--option", type=str, choices=['surfpos', 'surfz','surfInpainting'], default='surfpos',
                        help="Choose between option [surfpos,edgepos,surfz,edgez] (default: surfpos)")
    parser.add_argument("--denoiser_type", type=str, choices=['default', 'hunyuan_dit'], default='default',
                        help="Choose ldm type.")
    parser.add_argument('--chunksize', type=int, default=256, help='Chunk size for data loading')
    
    # Training parameters
    parser.add_argument("--finetune",  action='store_true', help='Finetune from existing weights')
    parser.add_argument("--weight",  type=str, default=None, help='Weight path when finetuning')
    parser.add_argument("--gpu", type=int, nargs='+', default=[0, 1], help="GPU IDs to use for training (default: [0, 1])")

    parser.add_argument('--batch_size', type=int, default=512, help='input batch size')  
    parser.add_argument('--train_nepoch', type=int, default=3000, help='number of epochs to train for')
    parser.add_argument('--test_nepoch', type=int, default=2, help='number of epochs to test model')
    parser.add_argument('--save_nepoch', type=int, default=5, help='number of epochs to save model')
    
    # Dataset parameters
    parser.add_argument('--max_face', type=int, default=50, help='maximum number of faces')
    parser.add_argument('--threshold', type=float, default=0.01, help='minimum threshold between two faces')
    parser.add_argument('--bbox_scaled', type=float, default=1.0, help='scaled the bbox')
    parser.add_argument('--z_scaled', type=float, default=None, help='scaled the latent z')
    parser.add_argument("--data_aug",  action='store_true', help='Use data augmentation.')
    parser.add_argument('--data_fields', nargs='+', default=['surf_ncs'], help="Data fields to encode.")
    parser.add_argument("--padding", default="zero", type=str, choices=['repeat', 'zero', 'zerolatent'])

    # Model parameters
    parser.add_argument("--text_encoder", type=str, default=None, choices=[None, 'CLIP', 'T5', 'GME'], help="Text encoder when applying text as generation condition.")
    parser.add_argument("--pointcloud_encoder", type=str, default=None, choices=[None, 'POINT_E'], help="")
    parser.add_argument("--sketch_encoder", type=str, default=None, choices=[None, 'LAION2B', "RADIO_V2.5-G"], help="")
    parser.add_argument("--sketch_feature_dir", type=str, default="/A/B/C/D/E/F/G",  help="Text encoder when applying text as generation condition.")

    parser.add_argument('--block_dims', nargs='+', type=int, default=[32,64,64,128], help='Latent dimension of each block of the UNet model.')
    parser.add_argument('--latent_channels', type=int, default=8, help='Latent channels of the vae model.')
    parser.add_argument('--sample_mode', type=str, default='sample', choices=['mode', 'sample'], help='Encoder mode of the vae model.')
    parser.add_argument('--num_layer', type=int, default=12, help='Layer num of ldm model.')
    parser.add_argument('--embed_dim', type=int, default=768, help='Embding dim of ldm model.')

    # Save dirs and reload
    parser.add_argument('--expr', type=str, default="surface_pos", help='environment')
    parser.add_argument('--log_dir', type=str, default="log", help='name of the log folder.')
    args = parser.parse_args()
    # saved folder
    args.log_dir = f'{args.log_dir}/{args.expr}'
    return args



def run(args):
    
    # datamodule = getattr(datasets, args.datasets)
    
    # Initialize dataset and trainer
    if args.option == 'surfpos':
        train_dataset = SurfPosData(args.data, args.list, validate=False, aug=args.data_aug, args=args)
        val_dataset = SurfPosData(args.data, args.list, validate=True, aug=False, args=args)
        ldm = SurfPosTrainer(args, train_dataset, val_dataset)
    elif args.option == 'surfz':
        train_dataset = SurfZData(
            args.data, args.list, validate=False, aug=args.data_aug, args=args)
        val_dataset = SurfZData(
            args.data, args.list, validate=True, aug=False, args=args)
        ldm = SurfZTrainer(args, train_dataset, val_dataset)
    elif args.option == "surfInpainting":
            train_dataset = SurfInpaintingData(
                args.data, args.list, validate=False, aug=args.data_aug, args=args)
            val_dataset = SurfInpaintingData(
                args.data, args.list, validate=True, aug=False, args=args)
            ldm = SurfInpaintingTrainer2(args, train_dataset, val_dataset)
            # ldm = SurfInpaintingTrainer2(args, train_dataset, val_dataset)
    else:
        assert False, 'please choose between [surfpos, surfz, edgepos, edgez]'

    print('Start training...')
    
    # Main training loop
    for _ in range(args.train_nepoch):

        # Train for one epoch
        ldm.train_one_epoch()        

        # Evaluate model performance on validation set
        if ldm.epoch % args.test_nepoch == 0:
            ldm.test_val()

        # save model
        if ldm.epoch % args.save_nepoch == 0:
            ldm.save_model()

    return


if __name__ == "__main__":
    
    # Parse input augments
    args = get_args_ldm()

    # # # Set PyTorch to use only the specified GPU
    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.gpu))

    # Make project directory if not exist
    if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)
    
    run(args)
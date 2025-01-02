import os

import argparse

from trainer import SurfVAETrainer
from datasets.sxd import SurfData


def get_args_vae():    
    
    def _str2intlist(s): return list(map(int, s.split(',')))
    
    parser = argparse.ArgumentParser()

    # Dataset configuration
    parser.add_argument('--data', type=str, default='/data/AIGP/brebrep_reso_64_edge_snap', 
                        help='Path to data folder')  
    parser.add_argument('--train_list', type=str, default='data_process/stylexd_data_split_reso_64.pkl', 
                        help='Path to training list')  
    parser.add_argument('--val_list', type=str, default='data_process/stylexd_data_split_reso_64.pkl', 
                        help='Path to validation list')  
    parser.add_argument("--data_aug",  action='store_true', help='Use data augmentation')
    parser.add_argument('--data_fields', nargs='+', default=['surf_ncs'], help="Data fields to encode.")
    parser.add_argument('--chunksize', type=int, default=-1, help='Chunk size for data loading')

    # Model parameters
    parser.add_argument('--block_dims', type=_str2intlist, default="32,64,64,128", help='Latent dimension of each block of the UNet model.')
    
    # Training parameters
    parser.add_argument("--finetune",  action='store_true', help='Finetune from existing weights')
    parser.add_argument("--weight",  type=str, default=None, help='Weight path when finetuning')  
    parser.add_argument("--gpu", type=int, nargs='+', default=[0], help="GPU IDs to use for training (default: [0])")
    parser.add_argument('--batch_size', type=int, default=512, help='input batch size')
        
    # Logging configuration
    parser.add_argument('--train_nepoch', type=int, default=200, help='number of epochs to train for')    
    parser.add_argument('--save_nepoch', type=int, default=50, help='number of epochs to save model')
    parser.add_argument('--test_nepoch', type=int, default=10, help='number of epochs to test model')

    # Save dirs and reload
    parser.add_argument('--expr', type=str, default="surface_vae", help='experiment name')
    parser.add_argument('--log_dir', type=str, default="log", help='name of the log folder.')

    args = parser.parse_args()
    
    # saved folder
    args.log_dir = f'{args.log_dir}/{args.expr}'
    
    return args


def run(args):
    print('Args:', args)
    
    # Initialize dataset loader and trainer
    train_dataset = SurfData(
        args.data, args.train_list, data_fields=args.data_fields, 
        validate=False, aug=args.data_aug, chunksize=args.chunksize)
    val_dataset = SurfData(
        args.data, args.val_list, data_fields=args.data_fields, 
        validate=True, aug=False, chunksize=args.chunksize)
    vae = SurfVAETrainer(args, train_dataset, val_dataset)

    # Main training loop
    print('Start training...')
    
    for _ in range(args.train_nepoch):  

        # Train for one epoch
        vae.train_one_epoch()

        # Evaluate model performance on validation set
        if vae.epoch % args.test_nepoch == 0:
            vae.test_val()

        # save model
        if vae.epoch % args.save_nepoch == 0:
            vae.save_model()
    return
           

if __name__ == "__main__":
    
    args = get_args_vae()
    
    # Set PyTorch to use only the specified GPU
    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.gpu))
    
    # Make project directory if not exist
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    
    run(args)
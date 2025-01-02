import os
import wandb
from tqdm import tqdm

import torch
import torch.nn as nn 
from torchvision.utils import make_grid
import torchvision.transforms.functional as F
from diffusers import AutoencoderKL, DDPMScheduler
from network import *

# visualization utils
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


class SurfVAETrainer():
    """ Surface VAE Trainer """
    def __init__(self, args, train_dataset, val_dataset): 
        # Initilize model and load to gpu
        self.iters = 0
        self.epoch = 0
        self.log_dir = args.log_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = args.batch_size

        assert train_dataset.num_channels == val_dataset.num_channels, \
            'Expecting same dimensions for train and val dataset, got %d (train) and %d (val).'%(train_dataset.num_channels, val_dataset.num_channels)
        
        num_channels = train_dataset.num_channels
        sample_size = train_dataset.resolution

        model = AutoencoderKL(
            in_channels=num_channels,
            out_channels=num_channels,
            down_block_types=['DownEncoderBlock2D']*len(args.block_dims),
            up_block_types= ['UpDecoderBlock2D']*len(args.block_dims),
            block_out_channels=args.block_dims,
            layers_per_block=2,
            act_fn='silu',
            latent_channels=8,
            norm_num_groups=8,
            sample_size=sample_size,
        )

        # Load pretrained surface vae (fast encode version)
        if args.finetune:
            model.load_state_dict(torch.load(args.weight))
            
        self.model = model.to(self.device).train()

        # Initialize optimizer
        self.network_params = list(self.model.parameters())
        self.optimizer = torch.optim.AdamW(
            self.network_params,
            lr=5e-4, 
            weight_decay=1e-5
        )
        self.scaler = torch.cuda.amp.GradScaler()

        # Initialize wandb
        wandb.init(project='GarmentGen', dir=args.log_dir, name=args.expr)

        # Initilizer dataloader
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, 
                                                shuffle=True, 
                                                batch_size=self.batch_size,
                                                num_workers=8)
        self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset, 
                                             shuffle=False, 
                                             batch_size=self.batch_size,
                                             num_workers=8)
        return
    
    
    def train_one_epoch(self):
        """
        Train the model for one epoch
        """
        self.model.train()
        loss_fn = nn.MSELoss()

        progress_bar = tqdm(total=len(self.train_dataloader))
        progress_bar.set_description(f"Epoch {self.epoch}")

        # Train    
        for surf_uv in self.train_dataloader:
                        
            with torch.cuda.amp.autocast():
                surf_uv = surf_uv.to(self.device).permute(0,3,1,2)
                self.optimizer.zero_grad() # zero gradient

                # Pass through VAE 
                posterior = self.model.encode(surf_uv).latent_dist
                z = posterior.sample()      # = posterior.mean + torch.randn_like(posterior.std)*posterior.std
                dec = self.model.decode(z).sample

                # Loss functions
                kl_loss = posterior.kl().mean()
                mse_loss = loss_fn(dec, surf_uv) 
                total_loss = mse_loss + 1e-6*kl_loss

                # Update model
                self.scaler.scale(total_loss).backward()
                nn.utils.clip_grad_norm_(self.network_params, max_norm=5.0)  
                self.scaler.step(self.optimizer)
                self.scaler.update()

            # logging
            if self.iters % 10 == 0:
                _z = posterior.mode()
                wandb.log({
                    "loss-mse": mse_loss, "loss-kl": kl_loss, 
                    "z-min": z.min(), "z-max": z.max(), "z-mean": z.mean(), "z-std": z.std(), 
                    "mode-min": _z.min(), "mode-max": _z.max(), "mode-mean": _z.mean(), "mode-std": _z.std() 
                    }, step=self.iters)

            self.iters += 1
            progress_bar.update(1)

        progress_bar.close()
        
        # update train dataset
        self.train_dataset.update()
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, shuffle=True, batch_size=self.batch_size, num_workers=8)
        self.epoch += 1 
        
        return 
    

    def test_val(self):
        """
        Test the model on validation set
        """
        print('Running validation...')
        self.model.eval() # set to eval
        total_loss = 0
        total_count = 0
        mse_loss = nn.MSELoss(reduction='none')
        
        val_images = None
        
        with torch.no_grad():
            for surf_uv in self.val_dataloader:
                surf_uv = surf_uv.to(self.device).permute(0,3,1,2) # (N, H, W, C) => (N, C, H, W)
                
                posterior = self.model.encode(surf_uv).latent_dist
                z = posterior.sample()
                dec = self.model.decode(z).sample

                loss = mse_loss(dec, surf_uv).mean((1,2,3)).sum().item()
                total_loss += loss
                total_count += len(surf_uv)
                
                if val_images is None and dec.shape[0] > 16:
                    sample_idx = torch.randperm(dec.shape[0])[:16]
                    val_images = make_grid(dec[sample_idx, ...], nrow=8, normalize=True, value_range=(-1,1))
                    geo_images = wandb.Image(val_images[:3, ...], caption="Geometry output.")
                    uv_images = wandb.Image(val_images[-3:, ...], caption="UV output.")
                    mask_image = wandb.Image(val_images[-1:, ...], caption="Mask output.")
                    wandb.log({"Val-Geo": geo_images, "Val-UV": uv_images, "Val-Mask": mask_image}, step=self.iters)
                
        mse = total_loss/total_count
        self.model.train() # set to train
        wandb.log({"Val-mse": mse}, step=self.iters)
        
        self.val_dataset.update()
        self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset, 
                                             shuffle=False, 
                                             batch_size=self.batch_size,
                                             num_workers=8)    
        
        return mse
    
    
    def save_model(self):
        ckpt_log_dir = os.path.join(self.log_dir, 'ckpts')
        os.makedirs(ckpt_log_dir, exist_ok=True)
        torch.save(
            self.model.state_dict(), 
            os.path.join(ckpt_log_dir, f'vae_e{self.epoch:4d}.pt'))
        return
    
    
class SurfPosTrainer():
    """ Surface Position Trainer (3D bbox) """
    def __init__(self, args, train_dataset, val_dataset): 
        # Initilize model and load to gpu
        self.iters = 0
        self.epoch = 0
        self.log_dir = args.log_dir
        self.use_cf = args.use_cf
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize network
        model = SurfPosNet(self.use_cf)
        model = nn.DataParallel(model) # distributed training 
        self.model = model.to(self.device).train()

        self.loss_fn = nn.MSELoss()

        # Initialize diffusion scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule='linear',
            prediction_type='epsilon',
            beta_start=0.0001,
            beta_end=0.02,
            clip_sample=False,
        )

        # Initialize optimizer
        self.network_params = list(self.model.parameters())
        
        self.optimizer = torch.optim.AdamW(
            self.network_params,
            lr=5e-4,
            betas=(0.95, 0.999),
            weight_decay=1e-6,
            eps=1e-08,
        )

        self.scaler = torch.cuda.amp.GradScaler()

        # Initialize wandb
        wandb.init(project='GarmentGen', dir=self.log_dir, name=args.expr)

        # Initialize dataloader
        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=16)
        self.val_dataloader = torch.utils.data.DataLoader(
            val_dataset, shuffle=False, batch_size=args.batch_size, num_workers=16)
        
        return
    
    
    def train_one_epoch(self):
        """
        Train the model for one epoch
        """
        self.model.train()

        progress_bar = tqdm(total=len(self.train_dataloader))
        progress_bar.set_description(f"Epoch {self.epoch}")

        # Train    
        for data in self.train_dataloader:
            with torch.cuda.amp.autocast():
                
                surfPos, class_label, caption = data
                surfPos = surfPos.to(self.device)
                if class_label is not None: class_label = class_label.to(self.device)
                if caption is not None: pass
                
                
                # if self.use_cf:
                #     data_cuda = [dat.to(self.device) for dat in data] # map to gpu
                #     surfPos, class_label = data_cuda 
                # else:
                #     surfPos = data.to(self.device)
                #     class_label = None

                print('*** surfPos: ', surfPos.shape)

                bsz = len(surfPos)
                
                self.optimizer.zero_grad() # zero gradient

                # Add noise
                timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device).long()  # [batch,]
                surfPos_noise = torch.randn(surfPos.shape).to(self.device)  
                surfPos_diffused = self.noise_scheduler.add_noise(surfPos, surfPos_noise, timesteps)

                # Predict noise
                surfPos_pred = self.model(surfPos_diffused, timesteps, class_label, True)
              
                # Compute loss
                total_loss = self.loss_fn(surfPos_pred, surfPos_noise)
             
                # Update model
                self.scaler.scale(total_loss).backward()
                nn.utils.clip_grad_norm_(self.network_params, max_norm=50.0) # clip gradient
                self.scaler.step(self.optimizer)
                self.scaler.update()

            # logging
            if self.iters % 10 == 0:
                wandb.log({"Loss-noise": total_loss}, step=self.iters)

            self.iters += 1
            progress_bar.update(1)

        progress_bar.close()
        self.epoch += 1 
        return 
    

    def test_val(self):
        """
        Test the model on validation set
        """
        self.model.eval() # set to eval
        total_count = 0
        mse_loss = nn.MSELoss(reduction='none')
        total_loss = [0,0,0,0,0]

        for data in self.val_dataloader:
            if self.use_cf:
                data_cuda = [dat.to(self.device) for dat in data] # map to gpu
                surfPos, class_label = data_cuda 
            else:
                surfPos = data.to(self.device)  # (B, N, C)
                class_label = None

            bsz = len(surfPos)
        
            total_count += len(surfPos)
            

            for idx, step in enumerate([10,50,100,200,500]):
                # Evaluate at timestep 
                timesteps = torch.randint(step-1, step, (bsz,), device=self.device).long()  # [batch,]
                surfPos_noise = torch.randn(surfPos.shape).to(self.device)  
                surfPos_diffused = self.noise_scheduler.add_noise(surfPos, surfPos_noise, timesteps)
                with torch.no_grad():
                    surfPos_pred = self.model(surfPos_diffused, timesteps, class_label) 
                loss = mse_loss(surfPos_pred, surfPos_noise).mean((1,2)).sum().item()
                total_loss[idx] += loss

        mse = [loss/total_count for loss in total_loss]
        self.model.train() # set to train
        wandb.log({
            "Val-010": mse[0], "Val-050": mse[1], "Val-100": mse[2], 
            "Val-200": mse[3], "Val-500": mse[4]}, step=self.iters)
        
        return
    
    
    def save_model(self):
        ckpt_log_dir = os.path.join(self.log_dir, 'ckpts')
        os.makedirs(ckpt_log_dir, exist_ok=True)
        torch.save(
            self.model.module.state_dict(), 
            os.path.join(ckpt_log_dir, f'surfpos_e{self.epoch:4d}.pt'))
        return


class SurfZTrainer():
    """ Surface Latent Geometry Trainer. """
    def __init__(self, args, train_dataset, val_dataset): 
        
        # Initilize model and load to gpu
        self.iters = 0
        self.epoch = 0
        self.log_dir = args.log_dir
        self.z_scaled = args.z_scaled
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
                
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = args.batch_size
        
        self.pos_dim = self.train_dataset.pos_dim        
        self.num_channels = self.train_dataset.num_channels
        self.sample_size = self.train_dataset.resolution
        self.latent_channels = 8
        self.block_dims = args.block_dims
                 
        # Load pretrained surface vae (fast encode version)
        surf_vae_encoder = AutoencoderKLFastEncode(
            in_channels=self.num_channels,
            out_channels=self.num_channels,
            down_block_types=['DownEncoderBlock2D']*len(self.block_dims),
            up_block_types= ['UpDecoderBlock2D']*len(self.block_dims),
            block_out_channels=self.block_dims,
            layers_per_block=2,
            act_fn='silu',
            latent_channels=self.latent_channels,
            norm_num_groups=8,
            sample_size=self.sample_size,
        )
        
        surf_vae_encoder.load_state_dict(torch.load(args.surfvae), strict=False)
        surf_vae_encoder = nn.DataParallel(surf_vae_encoder) # distributed inference 
        self.surf_vae_encoder = surf_vae_encoder.to(self.device).eval()
        
        print('[DONE] Init parallel surface vae encoder.')
        
        train_dataset.init_encoder(self.surf_vae_encoder, self.z_scaled)
        val_dataset.init_encoder(self.surf_vae_encoder, train_dataset.z_scale)
        if self.z_scaled is None: self.z_scaled = train_dataset.z_scale
               
        # Initialize network
        model = SurfZNet(
            p_dim=self.pos_dim*2,
            z_dim=(self.sample_size//(2**(len(args.block_dims)-1)))**2 * self.latent_channels,
            num_heads=12,
            num_cf=train_dataset.num_classes
            )
        
        if args.finetune:
            state_dict = torch.load(args.weight)
            if 'model_state_dict' in state_dict: model.load_state_dict(state_dict['model_state_dict'])
            elif 'model' in state_dict: model.load_state_dict(state_dict['model'])
            else: model.load_state_dict(state_dict)
            print('Continue training from %s.'%(args.weight))
        
        model = nn.DataParallel(model) # distributed training 
        self.model = model.to(self.device).train()
        
        if args.finetune: model.load_state_dict(torch.load(args.weight))
        
        self.device = self.model.module.parameters().__next__().device

        self.loss_fn = nn.MSELoss()

        # Initialize diffusion scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule='linear',
            prediction_type='epsilon',
            beta_start=0.0001,
            beta_end=0.02,
            clip_sample=False,
        )

        # Initialize optimizer
        self.network_params = list(self.model.parameters())
        
        self.optimizer = torch.optim.AdamW(
            self.network_params,
            lr=5e-4,
            betas=(0.95, 0.999),
            weight_decay=1e-6,
            eps=1e-08,
        )

        self.scaler = torch.cuda.amp.GradScaler()

        # Initialize wandb
        wandb.init(project='GarmentGen', dir=args.log_dir, name=args.expr)

        # Initilizer dataloader
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, 
                                                shuffle=True, 
                                                batch_size=self.batch_size,
                                                num_workers=16)
        self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset, 
                                             shuffle=False, 
                                             batch_size=self.batch_size,
                                             num_workers=16)
        return
    

    def train_one_epoch(self):
        """
        Train the model for one epoch
        """
        self.model.train()

        progress_bar = tqdm(total=len(self.train_dataloader))
        progress_bar.set_description(f"Epoch {self.epoch}")

        # Train    
        for data in self.train_dataloader:
            with torch.cuda.amp.autocast():
                
                surfPos, surfZ, surf_mask, surf_cls, caption = data
                surfPos, surfZ, surf_mask, surf_cls = \
                    surfPos.to(self.device), surfZ.to(self.device), \
                    surf_mask.to(self.device), surf_cls.to(self.device)
                                        
                bsz = len(surfPos)
                
                # Augment the surface position (see https://arxiv.org/abs/2106.15282)
                # if torch.rand(1) > 0.3:
                aug_ts = torch.randint(0, 15, (bsz,), device=self.device).long()
                aug_noise = torch.randn(surfPos.shape).to(self.device)
                surfPos = self.noise_scheduler.add_noise(surfPos, aug_noise, aug_ts)
            
                surfZ = surfZ * self.z_scaled
                self.optimizer.zero_grad() # zero gradient

                # Add noise
                timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device).long()  # [batch,]
                surfZ_noise = torch.randn(surfZ.shape).to(self.device)  
                surfZ_diffused = self.noise_scheduler.add_noise(surfZ, surfZ_noise, timesteps)
                
                # Predict noise
                surfZ_pred = self.model(surfZ_diffused, timesteps, surfPos, surf_mask, surf_cls, True)

                # Loss
                total_loss = self.loss_fn(surfZ_pred[~surf_mask], surfZ_noise[~surf_mask])        
             
                # Update model
                self.scaler.scale(total_loss).backward()
                nn.utils.clip_grad_norm_(self.network_params, max_norm=50.0) # clip gradient
                self.scaler.step(self.optimizer)
                self.scaler.update()

            # logging
            if self.iters % 10 == 0:
                wandb.log({
                    "loss-noise": total_loss, "surfz-min": surfZ[~surf_mask].min(), 
                    "surfz-max": surfZ[~surf_mask].max(), "surfz-std": surfZ[~surf_mask].std(), 
                    "surf_mask-min": surf_mask.sum(dim=1).min(), "surf_mask-max": surf_mask.sum(dim=1).max(), 
                    "surfZ_pred-min": surfZ_pred[~surf_mask].min(), 'surfz_pred-max': surfZ_pred[~surf_mask].max()
                    }, step=self.iters)

            self.iters += 1
            progress_bar.update(1)

        progress_bar.close()
        # update train dataset
        if hasattr(self.train_dataset, 'data_chunks') and len(self.train_dataset.data_chunks) > 1:
            print('Updating train data chunks...')
            self.train_dataset.update()        
            self.train_dataloader = torch.utils.data.DataLoader(
                self.train_dataset, shuffle=True, 
                batch_size=self.batch_size, num_workers=16)
        
        self.epoch += 1 
        return 
    

    def test_val(self):
        """
        Test the model on validation set
        """
        self.model.eval() # set to eval
        total_count = 0
        mse_loss = nn.MSELoss(reduction='none')

        progress_bar = tqdm(total=len(self.val_dataloader))
        progress_bar.set_description(f"Testing")

        val_timesteps = [10,50,100,200,500,1000]
        total_loss = [0]*len(val_timesteps)

        vis_batch = torch.randint(0, len(self.val_dataloader), (1,)).item()        
        for batch_idx, data in enumerate(self.val_dataloader):
            surfPos, surfZ, surf_mask, surf_cls, caption = data
            surfPos, surfZ, surf_mask, surf_cls = \
                surfPos.to(self.device), surfZ.to(self.device), \
                surf_mask.to(self.device), surf_cls.to(self.device)
            
            bsz = len(surfPos)
                        
            tokens = surfZ = surfZ * self.z_scaled    

            total_count += len(surfPos)
            
            for idx, step in enumerate(val_timesteps):
                # Evaluate at timestep 
                timesteps = torch.randint(step-1, step, (bsz,), device=self.device).long()  # [batch,]
                noise = torch.randn(tokens.shape).to(self.device)  
                diffused = self.noise_scheduler.add_noise(tokens, noise, timesteps)
                
                with torch.no_grad(): pred = self.model(diffused, timesteps, surfPos, surf_mask, surf_cls)
                    
                loss = mse_loss(pred[~surf_mask], noise[~surf_mask]).mean(-1).sum().item()
                total_loss[idx] += loss

            progress_bar.update(1)
        progress_bar.close()

        mse = [loss/total_count for loss in total_loss]
        self.model.train() # set to train
        wandb.log(dict([(f"val-{step:04d}", mse[idx]) for idx, step in enumerate(val_timesteps)]), step=self.iters)
        
        if hasattr(self.val_dataset, 'data_chunks') and len(self.val_dataset.data_chunks) > 1:
            self.val_dataset.update()
            self.val_dataloader = torch.utils.data.DataLoader(
                self.val_dataset, shuffle=False, 
                batch_size=self.batch_size, num_workers=16)
        return
    

    def save_model(self):
        ckpt_log_dir = os.path.join(self.log_dir, 'ckpts')
        os.makedirs(ckpt_log_dir, exist_ok=True)        
        torch.save(
            {
                'model_state_dict': self.model.module.state_dict(),
                'z_scale': self.z_scaled
            }, 
            os.path.join(ckpt_log_dir, f'surfz_e{self.epoch:4d}.pt'))
        return
    
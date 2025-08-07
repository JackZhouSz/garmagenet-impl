import os

import numpy as np
from tqdm import tqdm

import torch
import wandb
from torchvision.utils import make_grid
from diffusers import AutoencoderKL, DDPMScheduler

from src.network import *
from src.cfg import get_VAE_cfg
from src.utils import get_wandb_logging_meta

from src.bbox_utils import bbox_deduplicate
from src.bbox_utils import get_diff_map, bbox_2d_iou, bbox_3d_iou, bbox_l2_distance
import torch.distributed as dist

class SurfVAETrainer():
    """ Surface VAE Trainer """
    def __init__(self, args, train_dataset, val_dataset):
        # Initilize model and load to gpu
        self.iters = 0
        self.epoch = 0
        self.log_dir = args.log_dir
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = args.batch_size

        self.data_fields = args.data_fields

        assert train_dataset.num_channels == val_dataset.num_channels, \
            'Expecting same dimensions for train and val dataset, got %d (train) and %d (val).'%(train_dataset.num_channels, val_dataset.num_channels)

        num_channels = train_dataset.num_channels
        sample_size = train_dataset.resolution
        latent_channels = args.latent_channels

        model = AutoencoderKL(
            in_channels=num_channels,
            out_channels=num_channels,
            down_block_types=['DownEncoderBlock2D']*len(args.block_dims),
            up_block_types= ['UpDecoderBlock2D']*len(args.block_dims),
            block_out_channels=args.block_dims,
            layers_per_block=2,
            act_fn='silu',
            latent_channels=latent_channels,
            norm_num_groups=8,
            sample_size=sample_size,
        )

        if args.finetune:
            state_dict = torch.load(args.weight)
            if 'model_state_dict' in state_dict: model.load_state_dict(state_dict['model_state_dict'])
            elif 'model' in state_dict: model.load_state_dict(state_dict['model'])
            else: model.load_state_dict(state_dict)
            print('Load SurfZNet checkpoint from %s.'%(args.weight))

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
        run_id, run_step = get_wandb_logging_meta(os.path.join(args.log_dir, 'wandb'))
        wandb.init(project='GarmentGen', dir=args.log_dir, name=args.expr, id=run_id, resume='allow')
        self.iters = run_step

        # Initilizer dataloader
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                shuffle=True,
                                                batch_size=self.batch_size,
                                                num_workers=8)
        self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset,
                                             shuffle=False,
                                             batch_size=self.batch_size,
                                             num_workers=8)

        # Get Current Epoch
        try:
            self.epoch = int(os.path.basename(args.weight).split("_e")[1].split(".")[0])
            print("Resume epoch from args.weight.\n"
                  f"Current epoch is: {self.epoch}")
        except Exception:
            self.epoch = self.iters // len(self.train_dataloader)
            print("Resume epoch from args.weight.\n"
                  f"Current epoch is: {self.epoch}")
            print("This may cause error if batch size has changed.")

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
                with torch.autograd.set_detect_anomaly(True):
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

        torch.cuda.empty_cache()
        progress_bar.close()

        # update train dataset
        self.train_dataset.update()
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, shuffle=True, batch_size=self.batch_size, num_workers=8)
        self.epoch += 1

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

                    vis_log = {}
                    if 'surf_ncs' in self.data_fields: vis_log['Val-Geo'] = wandb.Image(val_images[:3, ...], caption="Geometry output.")
                    if 'surf_wcs' in self.data_fields:
                        val_images2 = make_grid(dec[sample_idx, :3], nrow=8, normalize=False)
                        val_images2[val_images2 != 0.0] = (val_images2[val_images2 != 0.0] + 1) / 2
                        vis_log['Val-Geo-WCS'] = wandb.Image(val_images2[:3, ...], caption="Geometry WCS output.")
                        # sampled_wcs = dec[sample_idx, ...][:,:3]
                        # valid_pts = sampled_wcs[(torch.abs(sampled_wcs[:,0:1,...])>0.01).repeat(1,3,1,1)]
                        # value_range = (torch.min(sampled_wcs[sampled_wcs!=0.0]),torch.max(sampled_wcs[sampled_wcs!=0.0]))
                        # wcs_norm = make_grid(sampled_wcs, nrow=8, normalize=True, value_range=value_range)
                        # vis_log['Val-Geo'] = wandb.Image(wcs_norm[:3, ...], caption="Geometry output.")
                    if 'surf_uv_ncs' in self.data_fields: vis_log['Val-UV'] = wandb.Image(val_images[-3:, ...], caption="UV output.")
                    if 'surf_normals' in self.data_fields: vis_log['Val-Normal'] = wandb.Image(val_images[3:6, ...], caption="Normal output.")
                    if 'surf_mask' in self.data_fields: vis_log['Val-Mask'] = wandb.Image(val_images[-1:, ...], caption="Mask output.")


                    wandb.log(vis_log, step=self.iters)

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
            os.path.join(ckpt_log_dir, f'vae_e{self.epoch:04d}.pt'))
        return


def get_condition_dim(args, self):
    if args.text_encoder is not None:
        condition_dim = self.text_encoder.text_emb_dim
    elif args.pointcloud_encoder is not None:
        condition_dim = self.pointcloud_encoder.pointcloud_emb_dim
    elif args.sketch_encoder is not None:
        if args.sketch_encoder == "LAION2B":
            condition_dim = 1280
        elif args.sketch_encoder == "RADIO_V2.5-G":
            condition_dim = 1536
        else:
            raise NotImplementedError("args.sketch_encoder name wrong.")
    else:
        condition_dim = -1

    return condition_dim


class SurfPosTrainer():
    """ Surface Position Trainer (3D bbox) """
    def __init__(self, args, train_dataset, val_dataset):
        # Initilize model and load to gpu
        self.iters = 0
        self.epoch = 0
        self.log_dir = args.log_dir

        if args.device == None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = args.device

        self.bbox_scaled = args.bbox_scaled

        self.denoiser_type = args.denoiser_type

        # Initialize condition encoder
        if args.text_encoder is not None: self.text_encoder = TextEncoder(args.text_encoder, self.device)
        if args.pointcloud_encoder is not None: self.pointcloud_encoder = PointcloudEncoder(args.pointcloud_encoder, self.device)
        if args.sketch_encoder is not None: self.sketch_encoder = args.sketch_encoder

        condition_dim = get_condition_dim(args, self)

        # Initialize network
        self.pos_dim = train_dataset.pos_dim

        # Initialize network
        if self.denoiser_type == "default":
            print("Default Transformer-Encoder denoiser.")
            model = SurfPosNet(
                p_dim=self.pos_dim * 2,
                embed_dim=args.embed_dim,
                condition_dim=condition_dim,
                num_cf=train_dataset.num_classes)
        elif self.denoiser_type == "hunyuan_dit":
            print("Hunyuan2.0 Dit denoiser.")
            model = SurfPosNet_hunyuandit(
                p_dim=self.pos_dim * 2,
                embed_dim=args.embed_dim,
                condition_dim=condition_dim,
                num_cf=train_dataset.num_classes,
                num_layer=args.num_layer
            )
        else:
            raise NotImplementedError

        if args.finetune:
            state_dict = torch.load(args.weight)
            if 'bbox_scaled' in state_dict:
                self.bbox_scaled = train_dataset.bbox_scaled = val_dataset.bbox_scaled = state_dict['bbox_scaled']
            if 'model_state_dict' in state_dict: model.load_state_dict(state_dict['model_state_dict'])
            elif 'model' in state_dict: model.load_state_dict(state_dict['model'])
            else: model.load_state_dict(state_dict)
            print('Load SurfZNet checkpoint from %s.'%(args.weight))

        model = model.to(self.device)
        if args.batch_size > 1:
            model = nn.DataParallel(model)  # distributed training
        self.model = model.train()

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

        # # Initialize wandb
        run_id, run_step = get_wandb_logging_meta(os.path.join(args.log_dir, 'wandb'))
        wandb.init(project='GarmentGen', dir=args.log_dir, name=args.expr, id=run_id, resume='allow')
        self.iters = run_step

        # Initialize dataloader
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=16)
        self.val_dataloader = torch.utils.data.DataLoader(
            val_dataset, shuffle=False, batch_size=args.batch_size, num_workers=16)

        # Get Current Epoch
        try:
            self.epoch = int(os.path.basename(args.weight).split("_e")[1].split(".")[0])
            print("Resume epoch from args.weight.\n"
                  f"Current epoch is: {self.epoch}")
        except Exception:
            self.epoch = self.iters // len(self.train_dataloader)
            print("Resume epoch from args.weight.\n"
                  f"Current epoch is: {self.epoch}")
            print("This may cause error if batch size has changed.")

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
                surfPos, pad_mask, class_label, caption, pointcloud_feature, sketch_feature = data

                surfPos = surfPos.to(self.device)
                class_label = class_label.to(self.device)

                # encode condition
                if hasattr(self, 'text_encoder'):
                    condition_emb = self.text_encoder(caption)
                elif hasattr(self, 'pointcloud_encoder'):
                    condition_emb = pointcloud_feature
                elif hasattr(self, 'sketch_encoder'):
                    condition_emb = sketch_feature
                else:
                    condition_emb = None
                if condition_emb is not None:
                    condition_emb = condition_emb.to(self.device)

                bsz = len(surfPos)

                self.optimizer.zero_grad() # zero gradient

                # Add noise
                timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device).long()  # [batch,]
                surfPos_noise = torch.randn(surfPos.shape).to(self.device)
                surfPos_diffused = self.noise_scheduler.add_noise(surfPos, surfPos_noise, timesteps)

                # Predict noise
                surfPos_pred = self.model(
                    surfPos=surfPos_diffused,
                    timesteps=timesteps,
                    class_label=class_label,
                    condition=condition_emb,
                    is_train=True
                )

                # Compute loss
                total_loss = self.loss_fn(surfPos_pred, surfPos_noise)

                # Update model
                with torch.autograd.set_detect_anomaly(True):
                    self.scaler.scale(total_loss).backward()
                nn.utils.clip_grad_norm_(self.network_params, max_norm=50.0) # clip gradient
                self.scaler.step(self.optimizer)
                self.scaler.update()

            # logging
            if self.iters % 10 == 0: wandb.log({"loss-noise": total_loss}, step=self.iters)

            self.iters += 1
            progress_bar.update(1)

        torch.cuda.empty_cache()
        progress_bar.close()
        self.epoch += 1
        return

    def point_l2_error(self, pred_box, gt_box):
        assert pred_box.shape[-1] == gt_box.shape[-1]
        bbox_dim = pred_box.shape[-1]
        pred_pts = pred_box.reshape(-1, bbox_dim)
        gt_pts = gt_box.reshape(-1, bbox_dim)
        return np.linalg.norm(pred_pts - gt_pts, axis=1).mean()

    @torch.no_grad()
    def test_val(self):
        """
        Test the model on validation set
        """
        self.model.eval() # set to eval

        # evaluate stepwise ===
        total_count = 0
        mse_loss = nn.MSELoss(reduction='none')
        total_loss_mse = [0,0,0,0,0]

        # evaluate step-wise MSE ===
        print("Evaluating step-wise.")
        for data in tqdm(self.val_dataloader):
            surfPos, pad_mask, class_label, caption, pointcloud_feature, sketch_feature = data
            surfPos = surfPos.to(self.device)
            class_label = class_label.to(self.device)

            bsz = len(surfPos)
            total_count += len(surfPos)

            # encode condition
            if hasattr(self, 'text_encoder'):
                condition_emb = self.text_encoder(caption)
            elif hasattr(self, 'pointcloud_encoder'):
                condition_emb = pointcloud_feature
            elif hasattr(self, 'sketch_encoder'):
                condition_emb = sketch_feature
            else:
                condition_emb = None
            if condition_emb is not None:
                condition_emb = condition_emb.to(self.device)

            val_timesteps = [10,50,100,200,500,1000]
            total_loss_mse = [0]*len(val_timesteps)

            for idx, step in enumerate(val_timesteps):
                # Evaluate at timestep
                timesteps = torch.randint(step-1, step, (bsz,), device=self.device).long()  # [batch,]
                surfPos_noise = torch.randn(surfPos.shape).to(self.device)
                surfPos_diffused = self.noise_scheduler.add_noise(surfPos, surfPos_noise, timesteps)

                with torch.no_grad():
                    pred = self.model(
                        surfPos=surfPos_diffused,
                        timesteps=timesteps,
                        class_label=class_label,
                        condition=condition_emb,
                        is_train=True
                    )

                loss = mse_loss(pred, surfPos_noise).mean((1,2)).sum().item()
                total_loss_mse[idx] += loss

        mse = [loss/total_count for loss in total_loss_mse]

        wandb.log(
            dict([(f"val-{step:04d}", mse[idx]) for idx, step in enumerate(val_timesteps)]),
            step=self.iters
        )

        # evaluate whole denoising ===
        print("Evaluating whole denoising.")
        num_acc = []
        bbox_L2 = []
        bbox_L2_3D = []
        bbox_L2_2D = []
        bbox_IoU_3D = []
        bbox_IoU_2D = []

        for data in self.val_dataloader:
            sample_num = len(data[0])

            surfPos, pad_mask, class_label, caption, pointcloud_feature, sketch_feature = data
            surfPos = surfPos.to(self.device)
            bsz = len(surfPos)
            pad_mask = pad_mask.to(self.device)
            class_label = class_label.to(self.device)

            total_count += len(surfPos)

            # encode condition
            if hasattr(self, 'text_encoder'):
                condition_emb = self.text_encoder(caption)
            elif hasattr(self, 'pointcloud_encoder'):
                condition_emb = pointcloud_feature
            elif hasattr(self, 'sketch_encoder'):
                condition_emb = sketch_feature
            else:
                condition_emb = None
            if condition_emb is not None:
                condition_emb = condition_emb.to(self.device)

            # evaluate whole denoising ===
            # 完成整个去噪过程，评估BBox位置，以及板片数量
            surfPos_denoinsing = torch.randn(surfPos.shape).to(self.device)
            ddpm_scheduler = self.noise_scheduler
            ddpm_scheduler.set_timesteps(1000)

            for t in tqdm(ddpm_scheduler.timesteps, desc="Surf-Pos Denoising"):
                timesteps = t.reshape(-1).to(self.device).repeat(bsz)

                with torch.no_grad():
                    pred = self.model(
                        surfPos=surfPos_denoinsing,
                        timesteps=timesteps,
                        class_label=class_label,
                        condition=condition_emb,
                        is_train=True
                    )

                surfPos_denoinsing = ddpm_scheduler.step(pred, t, surfPos_denoinsing).prev_sample

            surfPos_pred = [bbox_deduplicate(b[None, ...], padding=self.train_dataset.padding)[0] for b in surfPos_denoinsing]
            n_surfs = torch.tensor([b.shape[-2] for b in surfPos_pred], device=self.device)
            n_surfs_gt = torch.sum(~pad_mask, dim=-1)
            num_acc.append((sum(n_surfs - n_surfs_gt == 0)/sample_num).item() * 100)

            # 按照3D bbox+2D bbox 对生成的板片进行匹配。
            bboxs_all_gt = surfPos
            bboxs_all_pred = surfPos_pred

            for idx in range(len(n_surfs)):
                if n_surfs[idx] != n_surfs_gt[idx]:
                    # 仅对板片数量相同的结果进行评估
                    continue
                else:
                    _bboxs_pred_ = bboxs_all_pred[idx]
                    _bboxs_gt_ = bboxs_all_gt[idx][:n_surfs_gt[idx]].detach().cpu().numpy()

                    # 根据BBox，对生成结果与GT进行一一皮胚
                    diff_map, cost, cost_total, row_ind, col_ind = get_diff_map(_bboxs_pred_, _bboxs_gt_)
                    _bboxs_gt_ = _bboxs_gt_[col_ind]

                    # bbox_2d_iou, bbox_3d_iou, bbox_l2_distance
                    bbox3diou = bbox_3d_iou(_bboxs_pred_[:, :6], _bboxs_gt_[:, :6])
                    bbox_IoU_3D.append(bbox3diou)

                    bbox2diou = bbox_2d_iou(_bboxs_pred_[:, 6:], _bboxs_gt_[:, 6:])
                    bbox_IoU_2D.append(bbox2diou)

                    bboxl2 = bbox_l2_distance(_bboxs_pred_, _bboxs_gt_)
                    bbox_L2.append(bboxl2)

                    bboxl23d = bbox_l2_distance(_bboxs_pred_[:, :6], _bboxs_gt_[:, :6])
                    bbox_L2_3D.append(bboxl23d)

                    bboxl22d = bbox_l2_distance(_bboxs_pred_[:, 6:], _bboxs_gt_[:, 6:])
                    bbox_L2_2D.append(bboxl22d)

            break   # 避免进行多轮，时间开销过大

        num_acc = sum(num_acc) / len(num_acc)

        bbox_L2 = sum(bbox_L2)/len(bbox_L2)
        bbox_L2_3D = sum(bbox_L2_3D)/len(bbox_L2_3D)
        bbox_L2_2D = sum(bbox_L2_2D)/len(bbox_L2_2D)

        bbox_IoU_3D = [sum(b) / len(b) for b in bbox_IoU_3D]
        bbox_IoU_3D = sum(bbox_IoU_3D)/len(bbox_IoU_3D)

        bbox_IoU_2D = [sum(b) / len(b) for b in bbox_IoU_2D]
        bbox_IoU_2D = sum(bbox_IoU_2D)/len(bbox_IoU_2D)

        self.model.train() # set to train
        wandb.log(
            {
                "#Panels": num_acc,
                "bbox_L2": bbox_L2,
                "bbox_L2_3D": bbox_L2_3D,
                "bbox_L2_2D": bbox_L2_2D,
                "bbox_IoU_3D" : bbox_IoU_3D,
                "bbox_IoU_2D": bbox_IoU_2D,
            },
            step=self.iters)
        return

    def save_model(self):
        ckpt_log_dir = os.path.join(self.log_dir, 'ckpts')
        os.makedirs(ckpt_log_dir, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.module.state_dict(),
            'bbox_scaled': self.bbox_scaled
        }, os.path.join(ckpt_log_dir, f'surfpos_e{self.epoch:04d}.pt'))
        return


class SurfZTrainer():
    """ Surface Latent Geometry Trainer. """
    def __init__(self, args, train_dataset, val_dataset):

        # Initilize model and load to gpu
        self.iters = 0
        self.epoch = 0
        self.log_dir = args.log_dir
        self.z_scaled = args.z_scaled
        self.bbox_scaled = args.bbox_scaled

        if args.device == None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = args.device

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = args.batch_size

        self.pos_dim = self.train_dataset.pos_dim
        self.num_channels = self.train_dataset.num_channels
        self.sample_size = self.train_dataset.resolution
        self.latent_channels = args.latent_channels
        self.block_dims = args.block_dims

        # Initialize text encoder
        if args.text_encoder is not None: self.text_encoder = TextEncoder(args.text_encoder, self.device)
        if args.pointcloud_encoder is not None: self.pointcloud_encoder = PointcloudEncoder(args.pointcloud_encoder, self.device)
        if args.sketch_encoder is not None: self.sketch_encoder = args.sketch_encoder

        condition_dim = get_condition_dim(args, self)

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
        surf_vae_encoder = surf_vae_encoder.to(self.device)
        if args.batch_size > 1:
            surf_vae_encoder = nn.DataParallel(surf_vae_encoder) # distributed inference

        # self.surf_vae_encoder = surf_vae_encoder.to(self.device).eval()
        self.surf_vae_encoder = surf_vae_encoder.eval()

        print('[DONE] Init parallel surface vae encoder.')

        train_dataset.init_encoder(self.surf_vae_encoder, self.z_scaled)
        val_dataset.init_encoder(self.surf_vae_encoder, train_dataset.z_scaled)
        if self.z_scaled is None: self.z_scaled = train_dataset.z_scaled

        # Initialize network
        if args.denoiser_type == "default":
            print("Default Transformer-Encoder denoiser.")
            model = SurfZNet(
                p_dim=self.pos_dim*2,
                z_dim=(self.sample_size//(2**(len(args.block_dims)-1)))**2 * self.latent_channels,
                num_heads=12,
                embed_dim=args.embed_dim,
                condition_dim=condition_dim,
                num_layer=args.num_layer,
                num_cf=train_dataset.num_classes
                )
        elif args.denoiser_type == "hunyuan_dit":
            print("Hunyuan2.0 Dit denoiser.")
            model = SurfZNet_hunyuandit(
                p_dim=self.pos_dim*2,
                z_dim=(self.sample_size//(2**(len(args.block_dims)-1)))**2 * self.latent_channels,
                num_heads=12,
                embed_dim=args.embed_dim,
                condition_dim=condition_dim,
                num_layer=args.num_layer,
                num_cf=train_dataset.num_classes
                )
        else:
            raise NotImplementedError

        if args.finetune:
            state_dict = torch.load(args.weight)
            if 'z_scaled' in state_dict:
                self.z_scaled = train_dataset.z_scaled = val_dataset.z_scaled = state_dict['z_scaled']
            if 'bbox_scaled' in state_dict:
                self.bbox_scaled = train_dataset.bbox_scaled = val_dataset.bbox_scaled = state_dict['bbox_scaled']
            if 'model_state_dict' in state_dict: model.load_state_dict(state_dict['model_state_dict'])
            elif 'model' in state_dict: model.load_state_dict(state_dict['model'])
            else: model.load_state_dict(state_dict)
            print('Load SurfZNet checkpoint from %s.'%(args.weight))

        model = model.to(self.device)
        if args.batch_size > 1:
            model = nn.DataParallel(model)  # distributed training
        # self.model = model.to(self.device).train()
        self.model = model.train()

        # self.device = self.model.module.parameters().__next__().device

        self.loss_fn = nn.MSELoss()

        # Initialize diffusion scheduler
        if args.scheduler == "DDPM":
            self.scheduler_type = 'DDPM'
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=1000,
                beta_schedule='linear',
                prediction_type='epsilon',
                beta_start=0.0001,
                beta_end=0.02,
                clip_sample=False,
            )
        elif args.scheduler == "HY_FMED":
            self.scheduler_type = 'HY_FMED'
            from src.models.denoisers.dit_hunyuan_2.schedulers import FlowMatchEulerDiscreteScheduler
            self.noise_scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps = 1000,
                shift=args.scheduler_shift,
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
        run_id, run_step = get_wandb_logging_meta(os.path.join(args.log_dir, 'wandb'))
        wandb.init(project='GarmentGen', dir=args.log_dir, name=args.expr, id=run_id, resume='allow')
        self.iters = run_step

        # # Initilizer dataloader
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                shuffle=True,
                                                batch_size=self.batch_size,
                                                num_workers=16)
        self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset,
                                             shuffle=False,
                                             batch_size=self.batch_size,
                                             num_workers=16)

        # Get Current Epoch
        try:
            self.epoch = int(os.path.basename(args.weight).split("_e")[1].split(".")[0])
            print("Resume epoch from args.weight.\n"
                  f"Current epoch is: {self.epoch}")
        except Exception:
            self.epoch = self.iters // len(self.train_dataloader)
            print("Resume epoch from args.weight.\n"
                  f"Current epoch is: {self.epoch}")
            print("This may cause error if batch size has changed.")


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
                surfPos, surfZ, surf_mask, surf_cls, caption, pointcloud_feature, sketch_feature  = data
                surfPos, surfZ, surf_mask, surf_cls = \
                    surfPos.to(self.device), surfZ.to(self.device), \
                    surf_mask.to(self.device), surf_cls.to(self.device)

                # encode condition
                if hasattr(self, 'text_encoder'):
                    condition_emb = self.text_encoder(caption)
                elif hasattr(self, 'pointcloud_encoder'):
                    condition_emb = pointcloud_feature
                elif hasattr(self, 'sketch_encoder'):
                    condition_emb = sketch_feature
                else:
                    condition_emb = None
                if condition_emb is not None:
                    condition_emb = condition_emb.to(self.device)

                bsz = len(surfPos)

                # Augment the surface position (see https://arxiv.org/abs/2106.15282)
                if torch.rand(1) > 0.3:
                    if self.scheduler_type == 'DDPM':
                        aug_ts = torch.randint(0, 15, (bsz,), device=self.device).long()
                        aug_noise = torch.randn(surfPos.shape).to(self.device)
                        surfPos = self.noise_scheduler.add_noise(surfPos, aug_noise, aug_ts)
                    elif self.scheduler_type == "HY_FMED":
                        aug_ts = torch.randint(0, 15, (bsz,), device=self.device).long()
                        aug_ts = self.noise_scheduler.timesteps.to(self.device)[aug_ts]
                        aug_noise = torch.randn(surfPos.shape).to(self.device)
                        surfPos = self.noise_scheduler.add_noise(surfPos, aug_noise, aug_ts)
                    else:
                        raise NotImplementedError

                surfZ = surfZ * self.z_scaled
                self.optimizer.zero_grad() # zero gradient

                # Add noise
                if self.scheduler_type == "DDPM":
                    timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device).long()  # [batch,]
                elif self.scheduler_type == "HY_FMED":
                    steps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device).long()  # [batch,]
                    timesteps =  self.noise_scheduler.timesteps.to(self.device)[steps]
                else:
                    raise NotImplementedError

                surfZ_noise = torch.randn(surfZ.shape).to(self.device)
                surfZ_diffused = self.noise_scheduler.add_noise(surfZ, surfZ_noise, timesteps)

                # Predict noise
                surfZ_pred = self.model(surfZ_diffused, timesteps, surfPos, surf_mask, surf_cls, condition_emb, is_train=True)

                # Loss
                total_loss = self.loss_fn(surfZ_pred[~surf_mask], surfZ_noise[~surf_mask])

                # Update model
                with torch.autograd.set_detect_anomaly(True):
                    self.scaler.scale(total_loss).backward()
                # self.scaler.scale(total_loss).backward()

                nn.utils.clip_grad_norm_(self.network_params, max_norm=50.0) # clip gradient
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # logging
                if self.iters % 10 == 0:
                    wandb.log({
                        "loss-noise": total_loss, "surfz-min": surfZ[~surf_mask].min(),
                        'z_scaled': self.z_scaled,
                        "surfz-max": surfZ[~surf_mask].max(), "surfz-std": surfZ[~surf_mask].std(),
                        "surf_mask-min": surf_mask.sum(dim=1).min(), "surf_mask-max": surf_mask.sum(dim=1).max(),
                        "surfZ_pred-min": surfZ_pred[~surf_mask].min(), 'surfz_pred-max': surfZ_pred[~surf_mask].max()
                        }, step=self.iters)

                self.iters += 1
                progress_bar.update(1)

        torch.cuda.empty_cache()
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
            surfPos, surfZ, surf_mask, surf_cls, caption, pointcloud_feature, sketch_feature  = data
            surfPos, surfZ, surf_mask, surf_cls = \
                surfPos.to(self.device), surfZ.to(self.device), \
                surf_mask.to(self.device), surf_cls.to(self.device)

            # encode condition
            if hasattr(self, 'text_encoder'):
                condition_emb = self.text_encoder(caption)
            elif hasattr(self, 'pointcloud_encoder'):
                condition_emb = pointcloud_feature
            elif hasattr(self, 'sketch_encoder'):
                condition_emb = sketch_feature
            else:
                condition_emb = None
            if condition_emb is not None:
                condition_emb = condition_emb.to(self.device)

            bsz = len(surfPos)

            tokens = surfZ = surfZ * self.z_scaled

            total_count += len(surfPos)

            for idx, step in enumerate(val_timesteps):
                # Evaluate at timestep
                # Add noise
                if self.scheduler_type == "DDPM":
                    timesteps = torch.randint(step-1, step, (bsz,), device=self.device).long()  # [batch,]
                elif self.scheduler_type == "HY_FMED":
                    steps = torch.randint(step-1, step, (bsz,), device=self.device).long()  # [batch,]
                    timesteps = self.noise_scheduler.timesteps.to(self.device)[steps]
                else:
                    raise NotImplementedError

                noise = torch.randn(tokens.shape).to(self.device)
                diffused = self.noise_scheduler.add_noise(tokens, noise, timesteps)

                with torch.no_grad(): pred = self.model(diffused, timesteps, surfPos, surf_mask, surf_cls, condition_emb)

                loss = mse_loss(pred[~surf_mask], noise[~surf_mask]).mean(-1).sum().item()
                total_loss[idx] += loss

            progress_bar.update(1)
        progress_bar.close()

        # todo 将 panelL2的评估代码加进来

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
                'z_scaled': self.z_scaled,
                'bbox_scaled': self.bbox_scaled
            },
            os.path.join(ckpt_log_dir, f'surfz_e{self.epoch:04d}.pt'))
        return


class SurfInpaintingTrainer():
    """
    将2Dbbox作为位置编码，将masklatent拼接到噪声中去预测噪声
    """
    def __init__(self, args, train_dataset, val_dataset):

        raise NotImplementedError
        self.args = args

        # Initilize model and load to gpu
        self.iters = 0
        self.epoch = 0
        self.log_dir = args.log_dir
        self.z_scaled = args.z_scaled
        self.bbox_scaled = args.bbox_scaled

        if args.device == None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = args.device

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = args.batch_size

        self.pos_dim = self.train_dataset.pos_dim
        self.num_channels = self.train_dataset.num_channels
        self.sample_size = self.train_dataset.resolution
        self.latent_channels = args.latent_channels
        self.block_dims = args.block_dims

        # # Initialize text encoder
        # if args.text_encoder is not None: self.text_encoder = TextEncoder(args.text_encoder, self.device)
        # if args.pointcloud_encoder is not None: self.pointcloud_encoder = PointcloudEncoder(args.pointcloud_encoder, self.device)
        # if args.sketch_encoder is not None: self.sketch_encoder = args.sketch_encoder

        # condition_dim = get_condition_dim(args, self)
        condition_dim=-1

        # Geo+Mask VAE
        self.surf_vae_encoder_1 = self.get_vae(
            args.surfvae_config,
            args.surfvae
        )
        # Mask VAE
        self.surf_vae_encoder_2 = self.get_vae(
            args.surfvae_2_config,
            args.surfvae_2
        )
        print('[DONE] Init parallel surface vae encoder.')

        train_dataset.init_encoder(
            geo_mask_encoder=self.surf_vae_encoder_1,
            mask_encoder=self.surf_vae_encoder_2,
            z_scaled = self.z_scaled
        )
        val_dataset.init_encoder(
            geo_mask_encoder=self.surf_vae_encoder_1,
            mask_encoder=self.surf_vae_encoder_2,
            z_scaled = self.z_scaled
        )
        # if self.z_scaled is None: self.z_scaled = train_dataset.z_scaled

        # GT-2Dpos GT-maskLatent 3Dpos geomaskLatent
        z_data_fields = {
            "surf_uv_bbox": 4,
            "latent_mask": 256,
            "surf_bbox": 6,
            "latent_geo_mask": 256,
        }
        z_dim = sum(list(z_data_fields.values()))

        # Initialize network
        # self.out_dim = 6+256
        self.out_dim = -1
        model = SurfZNet(
            p_dim=4,
            z_dim=z_dim,
            z_projector_dim = -1,
            out_dim=self.out_dim,
            embed_dim=args.embed_dim,
            num_heads=12,
            condition_dim=condition_dim,
            num_layer=args.num_layer,
            num_cf=train_dataset.num_classes
        )

        if args.finetune:
            state_dict = torch.load(args.weight)
            if 'z_scaled' in state_dict:
                self.z_scaled = train_dataset.z_scaled = val_dataset.z_scaled = state_dict['z_scaled']
            if 'bbox_scaled' in state_dict:
                self.bbox_scaled = train_dataset.bbox_scaled = val_dataset.bbox_scaled = state_dict['bbox_scaled']
            if 'model_state_dict' in state_dict:
                model.load_state_dict(state_dict['model_state_dict'])
            elif 'model' in state_dict:
                model.load_state_dict(state_dict['model'])
            else:
                model.load_state_dict(state_dict)
            print('Load SurfImpaintingNet checkpoint from %s.' % (args.weight))

        model = model.to(self.device)
        if args.batch_size > 1:
            model = nn.DataParallel(model)  # distributed training
        self.model = model.train()

        # self.device = self.model.module.parameters().__next__().device

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
        run_id, run_step = get_wandb_logging_meta(os.path.join(args.log_dir, 'wandb'))
        wandb.init(project='GarmentGen', dir=args.log_dir, name=args.expr, id=run_id, resume='allow')
        self.iters = run_step

        # Initilizer dataloader
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            shuffle=True,
                                                            batch_size=self.batch_size,
                                                            num_workers=16)
        self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset,
                                                          shuffle=False,
                                                          batch_size=self.batch_size,
                                                          num_workers=16)

        # Get Current Epoch
        try:
            self.epoch = int(os.path.basename(args.weight).split("_e")[1].split(".")[0])
            print("Resume epoch from args.weight.\n"
                  f"Current epoch is: {self.epoch}")
        except Exception:
            self.epoch = self.iters // len(self.train_dataloader)
            print("Resume epoch from args.weight.\n"
                  f"Current epoch is: {self.epoch}")
            print("This may cause error if batch size has changed.")

    def get_vae(self, cfg_fp, weight_fp):
        """
        load vae from config
        """

        VAE_cfg = get_VAE_cfg(cfg_fp)
        num_channels = VAE_cfg.num_channels
        block_dims = VAE_cfg.block_dims
        latent_channels = VAE_cfg.latent_channels
        sample_size = VAE_cfg.sample_size

        # Load pretrained surface vae (fast encode version)
        surf_vae_encoder = AutoencoderKLFastEncode(
            in_channels=num_channels,
            out_channels=num_channels,
            down_block_types=['DownEncoderBlock2D'] * len(block_dims),
            up_block_types=['UpDecoderBlock2D'] * len(block_dims),
            block_out_channels=block_dims,
            layers_per_block=2,
            act_fn='silu',
            latent_channels=latent_channels,
            norm_num_groups=8,
            sample_size=sample_size,
        )

        surf_vae_encoder.load_state_dict(torch.load(weight_fp, map_location=self.device), strict=False)

        surf_vae_encoder = surf_vae_encoder.to(self.device)
        if self.args.batch_size > 1:
            surf_vae_encoder = nn.DataParallel(surf_vae_encoder)  # distributed inference
        return surf_vae_encoder.eval()


    def train_one_epoch(self):
        """
        Train the model for one epoch
        """
        self.model.train()

        progress_bar = tqdm(total=len(self.train_dataloader))
        progress_bar.set_description(f"Epoch {self.epoch}")

        # Train
        for data in self.train_dataloader:
            with (torch.cuda.amp.autocast()):
                surfPos, latent_geo_mask, latent_mask, surf_mask, surf_cls, caption = data
                surfPos, latent_geo_mask, latent_mask, surf_mask, surf_cls =(
                    surfPos.to(self.device), latent_geo_mask.to(self.device), latent_mask.to(self.device), surf_mask.to(self.device), surf_cls.to(self.device))

                surf_bbox = surfPos[...,:6]
                surf_uv_bbox = surfPos[...,6:]

                # encode text
                condition_emb = None

                bsz = len(surfPos)

                # Augment the surface position (see https://arxiv.org/abs/2106.15282)
                # In inpainting work, we only add augmentation on 2D bbox
                if torch.rand(1) > 0.3:
                    aug_ts = torch.randint(0, 15, (bsz,), device=self.device).long()
                    aug_noise = torch.randn(surf_uv_bbox.shape).to(self.device)
                    surf_uv_bbox = self.noise_scheduler.add_noise(surf_uv_bbox, aug_noise, aug_ts)


                latent_geo_mask = latent_geo_mask * self.z_scaled
                latent_mask = latent_mask * self.z_scaled
                self.optimizer.zero_grad()  # zero gradient


                surfZ_gt = torch.concatenate([surf_uv_bbox, latent_mask, surf_bbox, latent_geo_mask], dim=-1)

                # 对哪些部分计算loss
                denoise_mask = torch.zeros(surfZ_gt.shape, device=surfZ_gt.device, dtype=torch.bool)
                denoise_mask[...,260:] = True

                # Add noise
                timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device).long()  # [batch,]
                surfZ_noise = torch.randn(surfZ_gt.shape).to(self.device)
                surfZ_diffused = self.noise_scheduler.add_noise(surfZ_gt, surfZ_noise, timesteps)

                # 仅部分加噪
                surfZ_diffused[~denoise_mask] = surfZ_gt[~denoise_mask]

                # Predict noise
                surfZ_pred = self.model(
                    surfZ_diffused, timesteps,
                    surf_uv_bbox, surf_mask,
                    class_label=None, condition=condition_emb, is_train=True)

                # Loss
                # w_pos = 1. * 6/256   # weight of 3D bbox, 之前单独训SurfZ和SurfPos最终收敛的mseloss的比例
                # w_lgm = 1.          # weight of geo&mask latent
                # loss_pos = self.loss_fn(surfZ_pred[~surf_mask][...,260:266], surfZ_noise[~surf_mask][...,260:266])  * w_pos
                # loss_lgm = self.loss_fn(surfZ_pred[~surf_mask][...,266:], surfZ_noise[~surf_mask][...,266:])        * w_lgm
                # total_loss = loss_pos + loss_lgm

                # w_loss = 1.
                # total_loss = self.loss_fn(surfZ_pred[~surf_mask][...,260:324], surfZ_noise[~surf_mask][...,260:324]) * w_loss

                w_loss = 1.
                if self.out_dim<0:
                    total_loss = self.loss_fn(surfZ_pred[~surf_mask][...,260:], surfZ_noise[~surf_mask][...,260:]) * w_loss
                else:
                    total_loss = self.loss_fn(surfZ_pred[~surf_mask], surfZ_noise[~surf_mask][..., 260:]) * w_loss

                # Update model
                with torch.autograd.set_detect_anomaly(True):
                    self.scaler.scale(total_loss).backward()
                nn.utils.clip_grad_norm_(self.network_params, max_norm=50.0)  # clip gradient
                self.scaler.step(self.optimizer)
                self.scaler.update()
                total_loss = total_loss/w_loss

            # logging
            if self.iters % 10 == 0:
                surfZ_geo_mask = surfZ_gt[~surf_mask][...,266:]
                wandb.log({
                    "loss-noise": total_loss,
                    # "loss-pos": loss_pos, "loss-geo_mask_latent":loss_lgm,
                    'z_scaled': self.z_scaled,
                    "surfz-max": surfZ_geo_mask.max(), "surfz-min": surfZ_geo_mask.min(), "surfz-std": surfZ_geo_mask.std(),
                    "surfZ_pred-min": surfZ_pred[~surf_mask].min(), 'surfz_pred-max': surfZ_pred[~surf_mask].max()
                }, step=self.iters)

            self.iters += 1
            progress_bar.update(1)

        torch.cuda.empty_cache()
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
        self.model.eval()  # set to eval
        total_count = 0
        mse_loss = nn.MSELoss(reduction='none')

        progress_bar = tqdm(total=len(self.val_dataloader))
        progress_bar.set_description(f"Testing")

        val_timesteps = [10, 50, 100, 200, 500, 1000]
        total_loss = [0] * len(val_timesteps)

        vis_batch = torch.randint(0, len(self.val_dataloader), (1,)).item()
        for batch_idx, data in enumerate(self.val_dataloader):
            surfPos, latent_geo_mask, latent_mask, surf_mask, surf_cls, caption = data
            surfPos, latent_geo_mask, latent_mask, surf_mask, surf_cls = (
                surfPos.to(self.device), latent_geo_mask.to(self.device), latent_mask.to(self.device), surf_mask.to(self.device), surf_cls.to(self.device))

            surf_bbox = surfPos[..., :6]
            surf_uv_bbox = surfPos[..., 6:]

            # encode text
            condition_emb = None
            bsz = len(surfPos)

            latent_geo_mask = latent_geo_mask * self.z_scaled
            latent_mask = latent_mask * self.z_scaled
            self.optimizer.zero_grad()  # zero gradient

            surfZ_gt = torch.concatenate([surf_uv_bbox, latent_mask, surf_bbox, latent_geo_mask], dim=-1)

            # 对哪些部分计算loss
            denoise_mask = torch.zeros(surfZ_gt.shape, device=surfZ_gt.device, dtype=torch.bool)
            denoise_mask[..., 260:] = True

            total_count += len(surfPos)

            for idx, step in enumerate(val_timesteps):
                # Evaluate at timestep
                timesteps = torch.randint(step - 1, step, (bsz,), device=self.device).long()  # [batch,]
                surfZ_noise = torch.randn(surfZ_gt.shape).to(self.device)
                surfZ_diffused = self.noise_scheduler.add_noise(surfZ_gt, surfZ_noise, timesteps)
                # 仅部分加噪
                # surfZ_diffused[~denoise_mask] = surfZ_gt[~denoise_mask]

                with torch.no_grad():
                    surfZ_pred = self.model(
                        surfZ_diffused, timesteps,
                        surfPos[..., 6:], surf_mask,
                        class_label=None, condition=condition_emb, is_train=True)

                # loss = mse_loss(surfZ_pred[~surf_mask][..., 260:324], surfZ_noise[~surf_mask][..., 260:324]).mean(-1).sum().item()

                if self.out_dim<0:
                    loss = mse_loss(surfZ_pred[~surf_mask][..., 260:], surfZ_noise[~surf_mask][..., 260:]).mean(-1).sum().item()
                else:
                    loss = mse_loss(surfZ_pred[~surf_mask], surfZ_noise[~surf_mask][..., 260:]).mean(-1).sum().item()
                total_loss[idx] += loss

            progress_bar.update(1)
        progress_bar.close()

        mse = [loss / total_count for loss in total_loss]
        self.model.train()  # set to train
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
                'z_scaled': self.z_scaled,
                'bbox_scaled': self.bbox_scaled
            },
            os.path.join(ckpt_log_dir, f'surfz_e{self.epoch:04d}.pt'))
        return


class SurfInpaintingTrainer2():
    """
    将Mask与2Dbbox直接作为condition
    """
    def __init__(self, args, train_dataset, val_dataset):
        self.args = args

        # Initilize model and load to gpu
        self.iters = 0
        self.epoch = 0
        self.log_dir = args.log_dir
        self.z_scaled = args.z_scaled
        self.bbox_scaled = args.bbox_scaled

        if args.device == None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = args.device

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = args.batch_size

        self.pos_dim = self.train_dataset.pos_dim
        self.num_channels = self.train_dataset.num_channels
        self.sample_size = self.train_dataset.resolution
        self.latent_channels = args.latent_channels
        self.block_dims = args.block_dims

        # # Initialize text encoder
        # if args.text_encoder is not None: self.text_encoder = TextEncoder(args.text_encoder, self.device)
        # if args.pointcloud_encoder is not None: self.pointcloud_encoder = PointcloudEncoder(args.pointcloud_encoder, self.device)
        # if args.sketch_encoder is not None: self.sketch_encoder = args.sketch_encoder

        # condition_dim = get_condition_dim(args, self)

        # Geo+Mask VAE
        self.surf_vae_encoder_1 = self.get_vae(
            args.surfvae_config,
            args.surfvae
        )
        # Mask VAE
        self.surf_vae_encoder_2 = self.get_vae(
            args.surfvae_2_config,
            args.surfvae_2
        )
        print('[DONE] Init parallel surface vae encoder.')

        train_dataset.init_encoder(
            geo_mask_encoder=self.surf_vae_encoder_1,
            mask_encoder=self.surf_vae_encoder_2,
            z_scaled = self.z_scaled
        )
        val_dataset.init_encoder(
            geo_mask_encoder=self.surf_vae_encoder_1,
            mask_encoder=self.surf_vae_encoder_2,
            z_scaled = self.z_scaled
        )
        # if self.z_scaled is None: self.z_scaled = train_dataset.z_scaled

        # GT-2Dpos GT-maskLatent 3Dpos geomaskLatent
        # z_data_fields = {
        #     "surf_uv_bbox": 4,
        #     "latent_mask": 256,
        #     "surf_bbox": 6,
        #     "latent_geo_mask": 256,
        # }
        # z_dim = sum(list(z_data_fields.values()))

        # Initialize network
        self.out_dim = -1
        self.condition_dim=256

        # Initialize network
        if args.denoiser_type == "default":
            print("Default Transformer-Encoder denoiser.")
            model = SurfZNet(
                p_dim=self.pos_dim*2,
                z_dim=(self.sample_size//(2**(len(args.block_dims)-1)))**2 * self.latent_channels,
                num_heads=12,
                condition_dim=self.condition_dim,
                num_cf=train_dataset.num_classes
                )
        elif args.denoiser_type == "hunyuan_dit":
            print("Hunyuan2.0 Dit denoiser.")
            model = SurfZNet_hunyuandit(
                p_dim=self.pos_dim*2,
                z_dim=(self.sample_size//(2**(len(args.block_dims)-1)))**2 * self.latent_channels,
                num_heads=12,
                condition_dim=self.condition_dim,
                num_cf=train_dataset.num_classes
                )
        else:
            raise NotImplementedError

        if args.finetune:
            state_dict = torch.load(args.weight)
            if 'z_scaled' in state_dict:
                self.z_scaled = train_dataset.z_scaled = val_dataset.z_scaled = state_dict['z_scaled']
            if 'bbox_scaled' in state_dict:
                self.bbox_scaled = train_dataset.bbox_scaled = val_dataset.bbox_scaled = state_dict['bbox_scaled']
            if 'model_state_dict' in state_dict:
                model.load_state_dict(state_dict['model_state_dict'])
            elif 'model' in state_dict:
                model.load_state_dict(state_dict['model'])
            else:
                model.load_state_dict(state_dict)
            print('Load SurfImpaintingNet checkpoint from %s.' % (args.weight))

        model = model.to(self.device)
        if args.batch_size > 1:
            model = nn.DataParallel(model)  # distributed training
        self.model = model.train()

        # self.device = self.model.module.parameters().__next__().device

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
        run_id, run_step = get_wandb_logging_meta(os.path.join(args.log_dir, 'wandb'))
        wandb.init(project='GarmentGen', dir=args.log_dir, name=args.expr, id=run_id, resume='allow')
        self.iters = run_step

        # Initilizer dataloader
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            shuffle=True,
                                                            batch_size=self.batch_size,
                                                            num_workers=16)
        self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset,
                                                          shuffle=False,
                                                          batch_size=self.batch_size,
                                                          num_workers=16)

        # Get Current Epoch
        try:
            self.epoch = int(os.path.basename(args.weight).split("_e")[1].split(".")[0])
            print("Resume epoch from args.weight.\n"
                  f"Current epoch is: {self.epoch}")
        except Exception:
            self.epoch = self.iters // len(self.train_dataloader)
            print("Resume epoch from args.weight.\n"
                  f"Current epoch is: {self.epoch}")
            print("This may cause error if batch size has changed.")

    def get_vae(self, cfg_fp, weight_fp):
        """
        load vae from config
        """

        VAE_cfg = get_VAE_cfg(cfg_fp)
        num_channels = VAE_cfg.num_channels
        block_dims = VAE_cfg.block_dims
        latent_channels = VAE_cfg.latent_channels
        sample_size = VAE_cfg.sample_size

        # Load pretrained surface vae (fast encode version)
        surf_vae_encoder = AutoencoderKLFastEncode(
            in_channels=num_channels,
            out_channels=num_channels,
            down_block_types=['DownEncoderBlock2D'] * len(block_dims),
            up_block_types=['UpDecoderBlock2D'] * len(block_dims),
            block_out_channels=block_dims,
            layers_per_block=2,
            act_fn='silu',
            latent_channels=latent_channels,
            norm_num_groups=8,
            sample_size=sample_size,
        )

        surf_vae_encoder.load_state_dict(torch.load(weight_fp, map_location=self.device), strict=False)

        surf_vae_encoder = surf_vae_encoder.to(self.device)
        if self.args.batch_size > 1:
            surf_vae_encoder = nn.DataParallel(surf_vae_encoder)  # distributed inference
        return surf_vae_encoder.eval()


    def train_one_epoch(self):
        """
        Train the model for one epoch
        """
        self.model.train()

        progress_bar = tqdm(total=len(self.train_dataloader))
        progress_bar.set_description(f"Epoch {self.epoch}")

        # Train
        for data in self.train_dataloader:
            with (torch.cuda.amp.autocast()):
                surfPos, latent_geo_mask, latent_mask, surf_mask, surf_cls, caption = data
                surfPos, latent_geo_mask, latent_mask, surf_mask, surf_cls =(
                    surfPos.to(self.device), latent_geo_mask.to(self.device), latent_mask.to(self.device), surf_mask.to(self.device), surf_cls.to(self.device))

                surf_bbox = surfPos[...,:6]
                surf_uv_bbox = surfPos[...,6:]

                bsz = len(surfPos)

                # Augment the surface position (see https://arxiv.org/abs/2106.15282)
                # In inpainting work, we only add augmentation on 2D bbox
                if torch.rand(1) > 0.3:
                    aug_ts = torch.randint(0, 15, (bsz,), device=self.device).long()
                    aug_noise = torch.randn(surf_uv_bbox.shape).to(self.device)
                    surf_uv_bbox = self.noise_scheduler.add_noise(surf_uv_bbox, aug_noise, aug_ts)


                latent_geo_mask = latent_geo_mask * self.z_scaled
                latent_mask = latent_mask * self.z_scaled
                self.optimizer.zero_grad()  # zero gradient

                # encode text
                # condition_emb = torch.concatenate([surf_uv_bbox, latent_mask], dim=-1)
                condition_emb = latent_mask

                surfZ_gt = torch.concatenate([surf_bbox, latent_geo_mask], dim=-1)

                # Add noise
                timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device).long()  # [batch,]
                surfZ_noise = torch.randn(surfZ_gt.shape).to(self.device)
                surfZ_diffused = self.noise_scheduler.add_noise(surfZ_gt, surfZ_noise, timesteps)

                # Predict noise
                surfZ_pred = self.model(
                    surfZ_diffused, timesteps,
                    surf_uv_bbox, surf_mask,
                    class_label=None, condition=condition_emb, is_train=True)

                # Loss
                w_loss = 1.
                total_loss = self.loss_fn(surfZ_pred[~surf_mask], surfZ_noise[~surf_mask]) * w_loss

                # Update model
                with torch.autograd.set_detect_anomaly(True):
                    self.scaler.scale(total_loss).backward()
                nn.utils.clip_grad_norm_(self.network_params, max_norm=50.0)  # clip gradient
                self.scaler.step(self.optimizer)
                self.scaler.update()
                total_loss = total_loss/w_loss

            # logging
            if self.iters % 10 == 0:
                surfZ_geo_mask = surfZ_gt[~surf_mask][..., 6:]
                wandb.log({
                    "loss-noise": total_loss,
                    # "loss-pos": loss_pos, "loss-geo_mask_latent":loss_lgm,
                    'z_scaled': self.z_scaled,
                    "surfz-max": surfZ_geo_mask.max(), "surfz-min": surfZ_geo_mask.min(), "surfz-std": surfZ_geo_mask.std(),
                    "surfZ_pred-min": surfZ_pred[~surf_mask].min(), 'surfz_pred-max': surfZ_pred[~surf_mask].max()
                }, step=self.iters)

            self.iters += 1
            progress_bar.update(1)

        torch.cuda.empty_cache()
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
        self.model.eval()  # set to eval
        total_count = 0
        mse_loss = nn.MSELoss(reduction='none')

        progress_bar = tqdm(total=len(self.val_dataloader))
        progress_bar.set_description(f"Testing")

        val_timesteps = [10, 50, 100, 200, 500, 1000]
        total_loss = [0] * len(val_timesteps)

        vis_batch = torch.randint(0, len(self.val_dataloader), (1,)).item()
        for batch_idx, data in enumerate(self.val_dataloader):
            surfPos, latent_geo_mask, latent_mask, surf_mask, surf_cls, caption = data
            surfPos, latent_geo_mask, latent_mask, surf_mask, surf_cls = (
                surfPos.to(self.device), latent_geo_mask.to(self.device), latent_mask.to(self.device), surf_mask.to(self.device), surf_cls.to(self.device))

            surf_bbox = surfPos[..., :6]
            surf_uv_bbox = surfPos[..., 6:]

            # encode text
            bsz = len(surfPos)

            latent_geo_mask = latent_geo_mask * self.z_scaled
            latent_mask = latent_mask * self.z_scaled
            self.optimizer.zero_grad()  # zero gradient

            surfZ_gt = torch.concatenate([surf_bbox, latent_geo_mask], dim=-1)

            # # 对哪些部分计算loss
            # denoise_mask = torch.zeros(surfZ_gt.shape, device=surfZ_gt.device, dtype=torch.bool)
            # denoise_mask[..., 260:] = True

            condition_emb = latent_mask

            total_count += len(surfPos)


            for idx, step in enumerate(val_timesteps):
                # Evaluate at timestep
                timesteps = torch.randint(step - 1, step, (bsz,), device=self.device).long()  # [batch,]
                surfZ_noise = torch.randn(surfZ_gt.shape).to(self.device)
                surfZ_diffused = self.noise_scheduler.add_noise(surfZ_gt, surfZ_noise, timesteps)
                # 仅部分加噪
                # surfZ_diffused[~denoise_mask] = surfZ_gt[~denoise_mask]

                with torch.no_grad():
                    surfZ_pred = self.model(
                        surfZ_diffused, timesteps,
                        surf_uv_bbox, surf_mask,
                        class_label=None, condition=condition_emb, is_train=True)
                loss = mse_loss(surfZ_pred[~surf_mask], surfZ_noise[~surf_mask]).mean(-1).sum().item()
                total_loss[idx] += loss

            progress_bar.update(1)
        progress_bar.close()

        mse = [loss / total_count for loss in total_loss]
        self.model.train()  # set to train
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
                'z_scaled': self.z_scaled,
                'bbox_scaled': self.bbox_scaled
            },
            os.path.join(ckpt_log_dir, f'surfz_e{self.epoch:04d}.pt'))
        return

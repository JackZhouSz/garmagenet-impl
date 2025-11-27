import os
from tqdm import tqdm

import torch
import wandb
import numpy as np
from torchvision.utils import make_grid
from diffusers import AutoencoderKL, DDPMScheduler

from src.network import *
from src.utils import get_wandb_logging_meta
from src.bbox_utils import bbox_deduplicate
from src.bbox_utils import get_diff_map, bbox_2d_iou, bbox_3d_iou, bbox_l2_distance


class VAETrainer():
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

        self.vae_type = args.vae_type
        if self.vae_type == "kl":
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
            # latentcode in different length should adjust KLloss param.
            self.loss_fn = nn.MSELoss()
        else:
            raise NotImplementedError

        if args.finetune:
            state_dict = torch.load(args.weight)
            if 'model_state_dict' in state_dict: model.load_state_dict(state_dict['model_state_dict'])
            elif 'model' in state_dict: model.load_state_dict(state_dict['model'])
            else: model.load_state_dict(state_dict)
            print('Load SurfZNet checkpoint from %s.'%(args.weight))

        self.model = model.to(self.device).train()

        # Initialize optimizer
        if self.vae_type == "kl":
            self.network_params = list(self.model.parameters())
            self.optimizer = torch.optim.AdamW(
                self.network_params,
                lr=args.lr,
                weight_decay=1e-5
            )
        else:
            raise NotImplementedError

        self.scaler = torch.cuda.amp.GradScaler()

        # Initialize wandb
        run_id, run_step = get_wandb_logging_meta(os.path.join(args.log_dir, 'wandb'))
        wandb.init(project='GarmentGen', dir=args.log_dir, name=args.expr, id=run_id, resume='allow')
        self.iters = run_step

        # Initilizer dataloader
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=8
        )
        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=8
        )

        # Get Current Epoch
        try:
            self.epoch = int(os.path.basename(args.weight).split("_e")[1].split(".")[0])
            print("Resume epoch from args.weight.\n"
                  f"Current epoch is: {self.epoch}")
        except Exception:
            self.epoch = self.iters // len(self.train_dataloader)
            print("Resume epoch from args.weight.\n"
                  f"Current epoch is: {self.epoch}")
            # print("This may cause error if batch size has changed.")

    def train_one_epoch(self):
        """
        Train the model for one epoch
        """
        self.model.train()

        progress_bar = tqdm(total=len(self.train_dataloader))
        progress_bar.set_description(f"Epoch {self.epoch}")

        # Train
        for surf_data in self.train_dataloader:

            with torch.cuda.amp.autocast():
                surf_data = surf_data.to(self.device).permute(0,3,1,2)
                self.optimizer.zero_grad() # zero gradient

                # Pass through VAE
                if self.vae_type == "kl":
                    posterior = self.model.encode(surf_data).latent_dist
                    z = posterior.sample()      # = posterior.mean + torch.randn_like(posterior.std)*posterior.std
                    dec = self.model.decode(z).sample

                    # Loss functions
                    kl_loss = posterior.kl().mean()
                    mse_loss = self.loss_fn(dec, surf_data)
                    total_loss = mse_loss + 1e-6 * kl_loss
                else:
                    raise NotImplementedError

                # Update model
                with torch.autograd.set_detect_anomaly(True):
                    self.scaler.scale(total_loss).backward()
                nn.utils.clip_grad_norm_(self.network_params, max_norm=5.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

            # logging
            if self.iters % 10 == 0:
                if self.vae_type == "kl":
                    _z = posterior.mode()
                    wandb.log({
                        "loss-mse": mse_loss,
                        "loss-kl": kl_loss,
                        "z-min": z.min(),
                        "z-max": z.max(),
                        "z-mean": z.mean(),
                        "z-std": z.std(),
                        "mode-min": _z.min(),
                        "mode-max": _z.max(),
                        "mode-mean": _z.mean(),
                        "mode-std": _z.std()
                    }, step=self.iters)
                else:
                    raise NotImplementedError

            self.iters += 1
            progress_bar.update(1)

        progress_bar.close()

        # update train dataset
        self.train_dataset.update()
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, shuffle=True, batch_size=self.batch_size, num_workers=8)

        if self.epoch % 100 == 0:
            torch.cuda.empty_cache()
        self.epoch += 1

    def test_val(self):
        """
        Test the model on validation set
        """
        print('Running validation...')
        self.model.eval() # set to eval
        total_loss = 0
        total_count = 0

        if self.vae_type == "kl":
            eval_loss = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError

        val_images = None

        with torch.no_grad():
            for surf_data in self.val_dataloader:
                surf_data = surf_data.to(self.device).permute(0,3,1,2) # (N, H, W, C) => (N, C, H, W)

                # Pass through VAE
                if self.vae_type == "kl":
                    posterior = self.model.encode(surf_data).latent_dist
                    z = posterior.sample()      # = posterior.mean + torch.randn_like(posterior.std)*posterior.std
                    dec = self.model.decode(z).sample
                else:
                    raise NotImplementedError

                loss = eval_loss(dec, surf_data).mean((1,2,3)).sum().item()
                total_loss += loss
                total_count += len(surf_data)

                if val_images is None and dec.shape[0] > 16:
                    sample_idx = torch.randperm(dec.shape[0])[:16]
                    val_images = make_grid(dec[sample_idx, ...], nrow=8, normalize=True, value_range=(-1,1))

                    vis_log = {}
                    if 'surf_ncs' in self.data_fields: vis_log['Val-Geo'] = wandb.Image(val_images[:3, ...], caption="Geometry output.")
                    if 'surf_wcs' in self.data_fields:
                        val_images2 = make_grid(dec[sample_idx, :3], nrow=8, normalize=False)
                        val_images2[val_images2 != 0.0] = (val_images2[val_images2 != 0.0] + 1) / 2
                        vis_log['Val-Geo-WCS'] = wandb.Image(val_images2[:3, ...], caption="Geometry WCS output.")
                    if 'surf_uv_ncs' in self.data_fields: vis_log['Val-UV'] = wandb.Image(val_images[-3:, ...], caption="UV output.")
                    if 'surf_normals' in self.data_fields: vis_log['Val-Normal'] = wandb.Image(val_images[3:6, ...], caption="Normal output.")
                    if 'surf_mask' in self.data_fields: vis_log['Val-Mask'] = wandb.Image(val_images[-1:, ...], caption="Mask output.")

                    wandb.log(vis_log, step=self.iters)

        if self.vae_type == "kl":
            mse = total_loss / total_count
            self.model.train()  # set to train
            wandb.log({"Val-mse": mse}, step=self.iters)
        else:
            raise NotImplementedError

        self.val_dataset.update()
        self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset,
                                             shuffle=False,
                                             batch_size=self.batch_size,
                                             num_workers=8)

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
        elif args.sketch_encoder == "RADIO_V2.5-H":
            condition_dim = 3840
        else:
            raise NotImplementedError("args.sketch_encoder name wrong.")
    else:
        condition_dim = -1

    return condition_dim


class TypologyGenTrainer():
    """ Garment Typology Trainer (for 3D bbox generation) """
    def __init__(self, args, train_dataset, val_dataset):
        # Initilize model and load to gpu
        self.iters = 0
        self.epoch = 0
        self.log_dir = args.log_dir

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.gpu is not None:
            self.device_ids = args.gpu
        else:
            self.device_ids = None

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.bbox_scaled = args.bbox_scaled

        # Initialize condition encoder
        if args.text_encoder is not None:
            self.text_encoder = TextEncoder(args.text_encoder, self.device)
            self.cond_encoder = self.text_encoder
        if args.pointcloud_encoder is not None:
            self.pointcloud_encoder = PointcloudEncoder(args.pointcloud_encoder, self.device)
            self.cond_encoder = self.pointcloud_encoder
        if args.sketch_encoder is not None:
            self.sketch_encoder = args.sketch_encoder
            self.cond_encoder = self.sketch_encoder

        self.condition_dim = get_condition_dim(args, self)

        self.train_dataset.init_encoder(self.cond_encoder)
        self.val_dataset.init_encoder(self.cond_encoder)

        # Initialize network
        self.pos_dim = train_dataset.pos_dim

        # Initialize network
        if args.denoiser_type == "default":
            print("Default Transformer-Encoder denoiser.")
            model = TypologyGenNet(
                p_dim=self.pos_dim * 2,
                embed_dim=args.embed_dim,
                condition_dim=self.condition_dim,
                num_cf=train_dataset.num_classes)
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

        model = nn.DataParallel(model, device_ids=self.device_ids) # distributed training
        self.model = model.to(self.device).train()

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
            lr=args.lr,
            betas=(0.95, 0.999),
            weight_decay=1e-6,
            eps=1e-08,
        )
        self.scaler = torch.cuda.amp.GradScaler()
        if args.finetune:
            if "optimizer" in state_dict:
                self.optimizer.load_state_dict(state_dict["optimizer"])
            if "scaler" in state_dict:
                self.scaler.load_state_dict(state_dict["scaler"])

        # # Initialize wandb
        run_id, run_step = get_wandb_logging_meta(os.path.join(args.log_dir, 'wandb'))
        wandb.init(project='GarmentGen', dir=args.log_dir, name=args.expr, id=run_id, resume='allow')
        self.iters = run_step

        # Initialize dataloader
        num_worker = 16
        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=num_worker
        )
        self.val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=num_worker
        )

        # Get Current Epoch
        try:
            self.epoch = int(os.path.basename(args.weight).split("_e")[1].split(".")[0])
            print("Resume epoch from args.weight.\n"
                  f"Current epoch is: {self.epoch}")
        except Exception:
            self.epoch = self.iters // len(self.train_dataloader)
            print("Resume epoch from args.weight.\n"
                  f"Current epoch is: {self.epoch}")


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
            if self.iters % 10 == 0: wandb.log({
                "loss-noise": total_loss
            }, step=self.iters)

            self.iters += 1
            progress_bar.update(1)

        progress_bar.close()
        self.epoch += 1
        if self.epoch % 1000 == 0:
            torch.cuda.empty_cache()
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
        val_timesteps = [10, 50, 100, 200, 500, 1000]
        total_loss_mse = [0] * len(val_timesteps)

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
                        is_train=False
                    )

                loss = mse_loss(pred, surfPos_noise).mean((1,2)).sum().item()
                total_loss_mse[idx] += loss

        mse = [loss/total_count for loss in total_loss_mse]

        wandb.log(
            dict([(f"val-{step:04d}", mse[idx]) for idx, step in enumerate(val_timesteps)]),
            step=self.iters
        )

        # following only evaluate when conditonal generation
        if self.condition_dim>0:
            # evaluate whole denoising ===
            try:
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
                                is_train=False
                            )

                        surfPos_denoinsing = ddpm_scheduler.step(pred, t, surfPos_denoinsing).prev_sample

                    surfPos_pred = [bbox_deduplicate(b[None, ...], padding=self.train_dataset.padding)[0] for b in surfPos_denoinsing]
                    n_surfs = torch.tensor([b.shape[-2] for b in surfPos_pred], device=self.device)
                    n_surfs_gt = torch.sum(~pad_mask, dim=-1)
                    num_acc.append((sum(n_surfs - n_surfs_gt == 0)/sample_num).item() * 100)

                    bboxs_all_gt = surfPos
                    bboxs_all_pred = surfPos_pred

                    for idx in range(len(n_surfs)):
                        if n_surfs[idx] != n_surfs_gt[idx]:
                            continue
                        else:
                            _bboxs_pred_ = bboxs_all_pred[idx]
                            _bboxs_gt_ = bboxs_all_gt[idx][:n_surfs_gt[idx]].detach().cpu().numpy()

                            # matching bbox
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
                    break # only eval once

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
            except Exception:
                pass
        return

    def save_model(self):
        ckpt_log_dir = os.path.join(self.log_dir, 'ckpts')
        os.makedirs(ckpt_log_dir, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict(),
            'bbox_scaled': self.bbox_scaled,
            'optimizer': self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict()
        }, os.path.join(ckpt_log_dir, f'typology_e{self.epoch:04d}.pt'))
        return


class GeometryGenTrainer():
    """ Garment Geometry Trainer (for Latent Geometry generation) """
    def __init__(self, args, train_dataset, val_dataset):
        self.args = args

        # Initilize model and load to gpu
        self.iters = 0
        self.epoch = 0
        self.log_dir = args.log_dir
        self.z_scaled = args.z_scaled
        self.bbox_scaled = args.bbox_scaled

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.gpu is not None:
            self.device_ids = args.gpu
        else:
            self.device_ids = None

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = args.batch_size

        if args.pos_dim is None:
            self.pos_dim = self.train_dataset.pos_dim
        else:
            # training with wcs (one stage training)
            args.pos_dim = int(args.pos_dim)
            if args.pos_dim>0:
                raise ValueError("one stage training with wcs should set pos_dim<=0")
            self.pos_dim = args.pos_dim
        self.num_channels = self.train_dataset.num_channels
        self.sample_size = self.train_dataset.resolution
        self.latent_channels = args.latent_channels
        self.block_dims = args.block_dims

        # Initialize condition encoder
        if args.text_encoder is not None:
            self.text_encoder = TextEncoder(args.text_encoder, self.device)
            self.cond_encoder = self.text_encoder
        if args.pointcloud_encoder is not None:
            self.pointcloud_encoder = PointcloudEncoder(args.pointcloud_encoder, self.device)
            self.cond_encoder = self.pointcloud_encoder
        if args.sketch_encoder is not None:
            self.sketch_encoder = args.sketch_encoder
            self.cond_encoder = self.sketch_encoder

        self.condition_dim = get_condition_dim(args, self)

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

        surf_vae_encoder.load_state_dict(torch.load(args.surfvae, map_location=self.device), strict=False)
        surf_vae_encoder = nn.DataParallel(surf_vae_encoder, device_ids=self.device_ids) # distributed inference
        self.surf_vae_encoder = surf_vae_encoder.to(self.device).eval()

        self.train_dataset.init_encoder(self.surf_vae_encoder, self.cond_encoder, self.z_scaled)
        self.val_dataset.init_encoder(self.surf_vae_encoder, self.cond_encoder, train_dataset.z_scaled)
        if self.z_scaled is None: self.z_scaled = train_dataset.z_scaled

        # Initialize network
        if args.denoiser_type == "default":
            print("Default Transformer-Encoder denoiser.")
            model = GeometryGenNet(
                p_dim=self.pos_dim*2,
                z_dim=(self.sample_size//(2**(len(args.block_dims)-1)))**2 * self.latent_channels,
                num_heads=12,
                embed_dim=args.embed_dim,
                condition_dim=self.condition_dim,
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

        model = nn.DataParallel(model, device_ids=self.device_ids) # distributed training
        self.model = model.to(self.device).train()

        self.device = self.model.module.parameters().__next__().device

        self.loss_fn = nn.MSELoss()

        # Initialize diffusion scheduler ===
        # used to add noise on gt bbox
        self.pos_noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule='linear',
            prediction_type='epsilon',
            beta_start=0.0001,
            beta_end=0.02,
            clip_sample=False,
        )
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
        else:
            raise NotImplementedError

        # Initialize optimizer
        self.network_params = list(self.model.parameters())
        self.optimizer = torch.optim.AdamW(
            self.network_params,
            lr=args.lr,
            betas=(0.95, 0.999),
            weight_decay=1e-6,
            eps=1e-08,
        )
        self.scaler = torch.cuda.amp.GradScaler()
        if args.finetune:
            if "optimizer" in state_dict:
                self.optimizer.load_state_dict(state_dict["optimizer"])
            if "scaler" in state_dict:
                self.scaler.load_state_dict(state_dict["scaler"])

        # Initialize wandb
        run_id, run_step = get_wandb_logging_meta(os.path.join(args.log_dir, 'wandb'))
        wandb.init(project='GarmentGen', dir=args.log_dir, name=args.expr, id=run_id, resume='allow')
        self.iters = run_step

        # # Initilizer dataloader
        num_worker = 16
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=num_worker
        )
        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=num_worker
        )

        # Get Current Epoch
        try:
            self.epoch = int(os.path.basename(args.weight).split("_e")[1].split(".")[0])
            print("Resume epoch from args.weight.\n"
                  f"Current epoch is: {self.epoch}")
        except Exception:
            self.epoch = self.iters // len(self.train_dataloader)
            print("Resume epoch from args.weight.\n"
                  f"Current epoch is: {self.epoch}")

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
                # train wcs without surfpos and surf_mask
                if self.pos_dim<=0 and "surf_wcs" in self.args.data_fields:
                    surf_mask[...] = False
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
                    aug_ts = torch.randint(0, 15, (bsz,), device=self.device).long()
                    aug_noise = torch.randn(surfPos.shape).to(self.device)
                    surfPos = self.pos_noise_scheduler.add_noise(surfPos, aug_noise, aug_ts)

                surfZ = surfZ * self.z_scaled
                self.optimizer.zero_grad() # zero gradient

                # forward ===
                if self.scheduler_type == "DDPM":
                    timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device).long()  # [batch,]

                    surfZ_noise = torch.randn(surfZ.shape).to(self.device)
                    surfZ_diffused = self.noise_scheduler.add_noise(surfZ, surfZ_noise, timesteps)

                    # Predict noise
                    surfZ_pred = self.model(surfZ_diffused, timesteps, surfPos, surf_mask, surf_cls, condition_emb, is_train=True)

                    # Loss
                    total_loss = self.loss_fn(surfZ_pred[~surf_mask], surfZ_noise[~surf_mask])
                else:
                    raise NotImplementedError

                # Update model ===
                with torch.autograd.set_detect_anomaly(True):
                    self.scaler.scale(total_loss).backward()
                # self.scaler.scale(total_loss).backward()

                nn.utils.clip_grad_norm_(self.network_params, max_norm=50.0)  # clip gradient
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

        progress_bar.close()
        # update train dataset
        if hasattr(self.train_dataset, 'data_chunks') and len(self.train_dataset.data_chunks) > 1:
            print('Updating train data chunks...')
            self.train_dataset.update()
            self.train_dataloader = torch.utils.data.DataLoader(
                self.train_dataset, shuffle=True,
                batch_size=self.batch_size, num_workers=16)

        self.epoch += 1
        if self.epoch % 1000 == 0:
            torch.cuda.empty_cache()
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

        for batch_idx, data in enumerate(self.val_dataloader):
            surfPos, surfZ, surf_mask, surf_cls, caption, pointcloud_feature, sketch_feature  = data
            surfPos, surfZ, surf_mask, surf_cls = \
                surfPos.to(self.device), surfZ.to(self.device), \
                surf_mask.to(self.device), surf_cls.to(self.device)
            # train wcs without surfpos and surf_mask
            if self.pos_dim <= 0 and "surf_wcs" in self.args.data_fields:
                surf_mask[...] = False
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

            surfZ = surfZ * self.z_scaled

            total_count += len(surfPos)

            with torch.no_grad():
                for idx, step in enumerate(val_timesteps):
                    # Evaluate at timestep
                    # Add noise
                    if self.scheduler_type == "DDPM":
                        timesteps = torch.randint(step-1, step, (bsz,), device=self.device).long()  # [batch,]

                        surfZ_noise = torch.randn(surfZ.shape).to(self.device)
                        surfZ_diffused = self.noise_scheduler.add_noise(surfZ, surfZ_noise, timesteps)

                        with torch.no_grad():
                            surfZ_pred = self.model(surfZ_diffused, timesteps, surfPos, surf_mask, surf_cls, condition_emb)

                        loss = mse_loss(surfZ_pred[~surf_mask], surfZ_noise[~surf_mask]).mean(-1).sum().item()
                    else:
                        raise NotImplementedError

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
                'model_state_dict': self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict(),
                'z_scaled': self.z_scaled,
                'bbox_scaled': self.bbox_scaled,
                'optimizer': self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict()
            },
            os.path.join(ckpt_log_dir, f'geometrygen_e{self.epoch:04d}.pt'))
        return


class OneStage_Gen_Trainer(GeometryGenTrainer):
    """ Surface Latent Geometry Trainer. """
    def __init__(self, args, train_dataset, val_dataset):
        self.args = args

        # Initilize model and load to gpu
        self.iters = 0
        self.epoch = 0
        self.log_dir = args.log_dir
        self.z_scaled = args.z_scaled
        self.bbox_scaled = args.bbox_scaled

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.gpu is not None:
            self.device_ids = args.gpu
        else:
            self.device_ids = None

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = args.batch_size

        if args.pos_dim is None:
            self.pos_dim = self.train_dataset.pos_dim
        else:
            # training with wcs (one stage training)
            args.pos_dim = int(args.pos_dim)
            if args.pos_dim>0:
                raise ValueError("one stage training with wcs should set pos_dim<=0")
            self.pos_dim = args.pos_dim
        self.num_channels = self.train_dataset.num_channels
        self.sample_size = self.train_dataset.resolution
        self.latent_channels = args.latent_channels
        self.block_dims = args.block_dims

        # Initialize condition encoder
        self.cond_encoder = None
        if args.text_encoder is not None:
            self.text_encoder = TextEncoder(args.text_encoder, self.device)
            self.cond_encoder = self.text_encoder
        if args.pointcloud_encoder is not None:
            self.pointcloud_encoder = PointcloudEncoder(args.pointcloud_encoder, self.device)
            self.cond_encoder = self.pointcloud_encoder
        if args.sketch_encoder is not None:
            self.sketch_encoder = args.sketch_encoder
            self.cond_encoder = self.sketch_encoder

        self.condition_dim = get_condition_dim(args, self)

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

        surf_vae_encoder.load_state_dict(torch.load(args.surfvae, map_location=self.device), strict=False)
        surf_vae_encoder = nn.DataParallel(surf_vae_encoder, device_ids=self.device_ids) # distributed inference
        self.surf_vae_encoder = surf_vae_encoder.to(self.device).eval()

        self.train_dataset.init_encoder(self.surf_vae_encoder, self.cond_encoder, self.z_scaled)
        self.val_dataset.init_encoder(self.surf_vae_encoder, self.cond_encoder, train_dataset.z_scaled)
        if self.z_scaled is None: self.z_scaled = train_dataset.z_scaled

        # Initialize network
        model_p_dim = -1
        model_z_dim = (self.sample_size//(2**(len(args.block_dims)-1)))**2 * self.latent_channels + 8  # 8 = 3Dbbox(6)+2Dscale(2)
        if args.denoiser_type == "default":
            print("Default Transformer-Encoder denoiser.")
            model = GeometryGenNet(
                p_dim=model_p_dim,
                z_dim=model_z_dim,
                num_heads=12,
                embed_dim=args.embed_dim,
                condition_dim=self.condition_dim,
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
            print('Load checkpoint from %s.'%(args.weight))

        model = nn.DataParallel(model, device_ids=self.device_ids) # distributed training
        self.model = model.to(self.device).train()

        self.device = self.model.module.parameters().__next__().device

        self.loss_fn = nn.MSELoss()

        # Initialize diffusion scheduler ===
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
        else:
            raise NotImplementedError

        # Initialize optimizer
        self.network_params = list(self.model.parameters())

        self.optimizer = torch.optim.AdamW(
            self.network_params,
            lr=args.lr,
            betas=(0.95, 0.999),
            weight_decay=1e-6,
            eps=1e-08,
        )
        self.scaler = torch.cuda.amp.GradScaler()
        if args.finetune:
            if "optimizer" in state_dict:
                self.optimizer.load_state_dict(state_dict["optimizer"])
            if "scaler" in state_dict:
                self.scaler.load_state_dict(state_dict["scaler"])


        # Initialize wandb
        run_id, run_step = get_wandb_logging_meta(os.path.join(args.log_dir, 'wandb'))
        wandb.init(project='GarmentGen', dir=args.log_dir, name=args.expr, id=run_id, resume='allow')
        self.iters = run_step

        # # Initilizer dataloader
        num_worker = 16
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=num_worker
        )
        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=num_worker
        )

        # Get Current Epoch
        try:
            self.epoch = int(os.path.basename(args.weight).split("_e")[1].split(".")[0])
            print("Resume epoch from args.weight.\n"
                  f"Current epoch is: {self.epoch}")
        except Exception:
            self.epoch = self.iters // len(self.train_dataloader)
            print("Resume epoch from args.weight.\n"
                  f"Current epoch is: {self.epoch}")
            # print("This may cause error if batch size has changed.")

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
                surfZ = surfZ * self.z_scaled

                bbox_3d_gt = surfPos[...,:6]
                scale_2d_gt = surfPos[...,8:10] - surfPos[...,6:8]
                latent_gt = torch.concatenate([surfZ, bbox_3d_gt, scale_2d_gt], dim=-1)

                surf_mask[...] = False

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
                print(condition_emb.shape)
                bsz = len(surfPos)

                self.optimizer.zero_grad() # zero gradient

                # forward ===
                if self.scheduler_type == "DDPM":
                    timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device).long()  # [batch,]

                    latent_noise = torch.randn(latent_gt.shape).to(self.device)
                    latent_diffused = self.noise_scheduler.add_noise(latent_gt, latent_noise, timesteps)

                    # Predict noise
                    latent_pred = self.model(latent_diffused, timesteps, surfPos, surf_mask, surf_cls, condition_emb, is_train=True)

                    # Loss
                    loss_latent = self.loss_fn(latent_pred[~surf_mask][:, :64], latent_noise[~surf_mask][:, :64])
                    loss_bbox = self.loss_fn(latent_pred[~surf_mask][:, 64:], latent_noise[~surf_mask][:, 64:])
                    total_loss = loss_latent * 1 + loss_bbox * 5
                else:
                    raise NotImplementedError

                # Update model ===
                with torch.autograd.set_detect_anomaly(True):
                    self.scaler.scale(total_loss).backward()
                # self.scaler.scale(total_loss).backward()

                nn.utils.clip_grad_norm_(self.network_params, max_norm=50.0)  # clip gradient
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # logging
                if self.iters % 10 == 0:
                    wandb.log({
                        "epoch": self.epoch,
                        "total_loss": total_loss,
                        'loss_latent': loss_latent.item(),
                        'loss_bbox': loss_bbox.item(),
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
        if self.epoch % 1000 == 0:
            torch.cuda.empty_cache()
        return

    def test_val(self):
        """
        Test the model on validation set
        """
        self.model.eval() # set to eval
        total_count = 0
        
        mse_loss = nn.MSELoss(reduction='none')
        l1_loss = nn.L1Loss(reduction='none')

        progress_bar = tqdm(total=len(self.val_dataloader))
        progress_bar.set_description(f"Testing")

        val_timesteps = [10,50,100,200,500,1000]
        latent_loss = [0]*len(val_timesteps)
        bbox_loss = [0]*len(val_timesteps)
        bbox_L1 = [0]*len(val_timesteps)

        for batch_idx, data in enumerate(self.val_dataloader):
            surfPos, surfZ, surf_mask, surf_cls, caption, pointcloud_feature, sketch_feature  = data
            surfPos, surfZ, surf_mask, surf_cls = \
                surfPos.to(self.device), surfZ.to(self.device), \
                surf_mask.to(self.device), surf_cls.to(self.device)
            surfZ = surfZ * self.z_scaled

            bbox_3d_gt = surfPos[..., :6]
            scale_2d_gt = surfPos[..., 8:10] - surfPos[..., 6:8]
            latent_gt = torch.concatenate([surfZ,bbox_3d_gt,scale_2d_gt], dim=-1)

            surf_mask[...] = False

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

            total_count += len(surfPos)

            with torch.no_grad():
                for idx, step in enumerate(val_timesteps):
                    # Evaluate at timestep
                    # Add noise
                    if self.scheduler_type == "DDPM":
                        # timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device).long()  # [batch,]
                        timesteps = torch.randint(step - 1, step, (bsz,), device=self.device).long()

                        latent_noise = torch.randn(latent_gt.shape).to(self.device)
                        latent_diffused = self.noise_scheduler.add_noise(latent_gt, latent_noise, timesteps)

                        # Predict noise
                        latent_pred = self.model(latent_diffused, timesteps, None, surf_mask, surf_cls, condition_emb, is_train=True)

                        # Loss
                        # If without pos embed, token generated position is random.
                        loss_latent = mse_loss(latent_pred[~surf_mask][:, :64], latent_noise[~surf_mask][:, :64]).mean(-1).sum().item()
                        loss_bbox = mse_loss(latent_pred[~surf_mask][:, 64:], latent_noise[~surf_mask][:, 64:]).mean(-1).sum().item()
                        bbox_l1 = l1_loss(latent_pred[~surf_mask][:, 64:], latent_noise[~surf_mask][:, 64:]).mean(-1).sum().item()
                    else:
                        raise NotImplementedError

                    # total_loss[idx] += loss
                    latent_loss[idx] += loss_latent
                    bbox_loss[idx] += loss_bbox
                    bbox_L1[idx] += bbox_l1

            progress_bar.update(1)
        progress_bar.close()

        # logging
        mse_latent = [loss / total_count for loss in latent_loss]
        wandb.log(dict([(f"val-latent-{step:04d}", mse_latent[idx]) for idx, step in enumerate(val_timesteps)]), step=self.iters)
        bbox_loss = [loss / total_count for loss in bbox_loss]
        wandb.log(dict([(f"val-bbox-{step:04d}", bbox_loss[idx]) for idx, step in enumerate(val_timesteps)]), step=self.iters)
        bbox_L1 = [loss / total_count for loss in bbox_L1]
        wandb.log(dict([(f"val-bbox-L1-{step:04d}", bbox_L1[idx]) for idx, step in enumerate(val_timesteps)]), step=self.iters)

        self.model.train() # set to train

        if hasattr(self.val_dataset, 'data_chunks') and len(self.val_dataset.data_chunks) > 1:
            self.val_dataset.update()
            self.val_dataloader = torch.utils.data.DataLoader(
                self.val_dataset, shuffle=False,
                batch_size=self.batch_size, num_workers=16)

        return
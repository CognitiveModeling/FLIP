import pytorch_lightning as pl
import torch as th
import torch.nn as nn
import numpy as np
from utils.optimizers import Ranger, RAdam
from utils.configuration import Configuration
from model.flip import Flip
from utils.io import Timer, AverageDict
from einops import rearrange, repeat, reduce
import time

class FlipModule(pl.LightningModule):
    def __init__(self, cfg: Configuration):
        super().__init__()
        self.cfg = cfg

        print(f"RANDOM SEED: {cfg.seed}")
        np.random.seed(cfg.seed)
        th.manual_seed(cfg.seed)

        self.net = Flip(self.cfg.model, cfg.model_path)

        print(f"Parameters:               {sum(p.numel() for p in self.parameters() if p.requires_grad)}")
        print(f"encoder Parameters:       {sum(p.numel() for p in self.net.encoder.parameters() if p.requires_grad)}")

        self.lr = self.cfg.learning_rate
        self.loss_loggers = AverageDict(1000)
        self.eval_loggers = AverageDict(50)
        self.timer = Timer()
        self.val_metrics = AverageDict()
        self.world_size_checked = False

        self.register_buffer('num_updates', th.tensor(-1))
        self.register_buffer('num_iterations', th.tensor(-1))
        self.register_buffer('time_counter', th.tensor(0.0))
        self.num_val_steps = 0
        self.last_time = time.time()

        samples_per_group = [cfg.model.num_mask_pixels // 7] * 7
        samples_per_group[0] += cfg.model.num_mask_pixels % 7
        self.register_buffer('samples_per_group', th.tensor(samples_per_group))

        print("FlipModule initialized")

    def update_dataset(self, samples_per_group):
        # Convert to list if tensor
        if isinstance(samples_per_group, th.Tensor):
            samples_per_group = samples_per_group.tolist()
        
        # Update our local copy
        self.samples_per_group = th.tensor(samples_per_group, device=self.device)
        # Update the shared memory in the datamodule
        if hasattr(self.trainer, 'datamodule'):
            if hasattr(self.trainer.datamodule, 'update_samples_per_group'):
                self.trainer.datamodule.update_samples_per_group(samples_per_group)

    def analyze_and_update_mask_sampling(self, mask_predictions, mask_targets, log_prefix=""):
        """
        Analyze mask prediction performance per group and update sampling accordingly.
        
        Args:
            mask_predictions: Binary predictions for mask pixels
            mask_targets: Ground truth mask values
        """
        with th.no_grad():
        
            group_ious = []
            running_count = 0
            for group_idx, group_size in enumerate(self.samples_per_group.tolist()):
                if group_size == 0:
                    group_ious.append(1.0) 
                    continue
                    
                # Extract this group's predictions and targets
                start_idx = running_count
                end_idx = running_count + group_size
                
                group_preds = mask_predictions[:, start_idx:end_idx]
                group_targets = mask_targets[:, start_idx:end_idx]
                
                # Calculate IoU for this group
                intersection = (group_preds * group_targets).sum(dim=1)
                union = (group_preds + group_targets).sum(dim=1) - intersection
                group_iou = (intersection / union.clamp(min=1e-6)).mean()
                
                name = f"{log_prefix}group{group_idx}_iou"
                self.log(name, group_iou)
                group_ious.append(self.loss_loggers[name])
                running_count += group_size
            
            # Adjust sampling based on performance (more samples for groups with lower IoU)
            if self.training and 'dynamic_sampling' in self.cfg and self.cfg.dynamic_sampling:
                # Convert to inverse difficulties (lower IoU = higher difficulty)
                difficulties = [1.0 - iou for iou in group_ious]
                total_difficulty = sum(difficulties)
                normalized_difficulties = [d / total_difficulty for d in difficulties]

                remaining   = self.cfg.model.num_mask_pixels - self.cfg.min_samples_per_group * 7
                new_samples = [self.cfg.min_samples_per_group + int(remaining * diff) for diff in normalized_difficulties]
                new_samples[0] += self.cfg.model.num_mask_pixels - sum(new_samples)
                
                # Update the dataset sampling
                self.update_dataset(new_samples)


    def forward(self, args):
        return self.net(**args)

    def log(self, name, value, type = 'loss'):
        super().log(name, value, on_step=True, on_epoch=True, prog_bar=False, logger=False, sync_dist=False, batch_size=self.cfg.model.batch_size)

        if name.startswith("val_"):
            self.val_metrics.update(name, value.item() if isinstance(value, th.Tensor) else value)
        elif type == 'loss':
            self.loss_loggers.update(name, value.item() if isinstance(value, th.Tensor) else value)
        else:
            self.eval_loggers.update(name, value.item() if isinstance(value, th.Tensor) else value)

    def training_step(self, batch, batch_idx):
        del batch['dataset_length']
        results = self(batch)
        self.time_counter = (self.time_counter + (time.time() - self.last_time) / 60).detach()
        self.last_time = time.time()
        self.num_iterations = (self.num_iterations + 1).detach()

        loss = results['mask_loss']
        self.log("loss",     loss)
        self.log("mask_iou", results["mask_iou"])

        self.analyze_and_update_mask_sampling(results['mask_predictions'], results['mask_targets'], log_prefix="")

        if self.num_iterations % self.cfg.model.gradient_accumulation_steps == 0:
            self.num_updates = (self.num_updates + 1).detach()
            print("Epoch {}[{}|{}|{:.2f}%]: {}, Time: {:.2f}m Loss: {:.2e}, IoU: {:.1f}%, Group-IoU: {:.1f}%|{:.1f}%|{:.1f}%|{:.1f}%|{:.1f}%|{:.1f}%|{:.1f}%, Group-Percent: {:.1f}%|{:.1f}%|{:.1f}%|{:.1f}%|{:.1f}%|{:.1f}%|{:.1f}%, Group-Count: {:d}|{:d}|{:d}|{:d}|{:d}|{:d}|{:d}".format(
                self.current_epoch,
                self.trainer.local_rank,
                self.num_updates,
                ((self.num_updates % self.cfg.eval_interval) + 1) / self.cfg.eval_interval * 100,
                str(self.timer),
                self.time_counter.item(),
                float(self.loss_loggers['loss']),
                float(self.loss_loggers['mask_iou']) * 100,
                float(self.loss_loggers['group0_iou']) * 100,
                float(self.loss_loggers['group1_iou']) * 100,
                float(self.loss_loggers['group2_iou']) * 100,
                float(self.loss_loggers['group3_iou']) * 100,
                float(self.loss_loggers['group4_iou']) * 100,
                float(self.loss_loggers['group5_iou']) * 100,
                float(self.loss_loggers['group6_iou']) * 100,
                self.samples_per_group[0] / self.cfg.model.num_mask_pixels * 100,
                self.samples_per_group[1] / self.cfg.model.num_mask_pixels * 100,
                self.samples_per_group[2] / self.cfg.model.num_mask_pixels * 100,
                self.samples_per_group[3] / self.cfg.model.num_mask_pixels * 100,
                self.samples_per_group[4] / self.cfg.model.num_mask_pixels * 100,
                self.samples_per_group[5] / self.cfg.model.num_mask_pixels * 100,
                self.samples_per_group[6] / self.cfg.model.num_mask_pixels * 100,
                int(self.samples_per_group[0]),
                int(self.samples_per_group[1]),
                int(self.samples_per_group[2]),
                int(self.samples_per_group[3]),
                int(self.samples_per_group[4]),
                int(self.samples_per_group[5]),
                int(self.samples_per_group[6]),
            ), flush=True)

        self.val_metrics = AverageDict()
        self.num_val_steps = 0

        if th.isnan(loss):
            print("Loss is NaN")
            exit()

        return loss

    def validation_step(self, batch, batch_idx):
        dataset_length = batch['dataset_length'][0].item() / (self.cfg.model.batch_size * self.cfg.world_size)
        del batch['dataset_length']
        self.num_val_steps += 1
        results = self(batch)
        self.time_counter = (self.time_counter + (time.time() - self.last_time) / 60).detach()
        self.last_time = time.time()

        loss = results['mask_loss']
        self.log("val_loss",     loss)
        self.log("val_mask_iou", results["mask_iou"])

        self.analyze_and_update_mask_sampling(results['mask_predictions'], results['mask_targets'], log_prefix="val_")

        print("Test[{}|{}|{}|{}|{:.2f}%]: {}, Time: {:.2f}m, Loss: {:.2e}, IoU: {:.1f}%, Group-IoU: {:.1f}|{:.1f}|{:.1f}|{:.1f}|{:.1f}|{:.1f}|{:.1f}, Group-Percent: {:.1f}%|{:.1f}%|{:.1f}%|{:.1f}%|{:.1f}%|{:.1f}%|{:.1f}%, Group-Count: {:d}|{:d}|{:d}|{:d}|{:d}|{:d}|{:d}".format(
            self.trainer.local_rank,
            self.trainer.current_epoch,
            self.num_val_steps,
            dataset_length,
            self.num_val_steps / dataset_length * 100,
            str(self.timer),
            self.time_counter.item(),
            float(self.val_metrics['val_loss']),
            float(self.val_metrics['val_mask_iou']) * 100,
            float(self.val_metrics['val_group0_iou']) * 100,
            float(self.val_metrics['val_group1_iou']) * 100,
            float(self.val_metrics['val_group2_iou']) * 100,
            float(self.val_metrics['val_group3_iou']) * 100,
            float(self.val_metrics['val_group4_iou']) * 100,
            float(self.val_metrics['val_group5_iou']) * 100,
            float(self.val_metrics['val_group6_iou']) * 100,
            self.samples_per_group[0] / self.cfg.model.num_mask_pixels * 100,
            self.samples_per_group[1] / self.cfg.model.num_mask_pixels * 100,
            self.samples_per_group[2] / self.cfg.model.num_mask_pixels * 100,
            self.samples_per_group[3] / self.cfg.model.num_mask_pixels * 100,
            self.samples_per_group[4] / self.cfg.model.num_mask_pixels * 100,
            self.samples_per_group[5] / self.cfg.model.num_mask_pixels * 100,
            self.samples_per_group[6] / self.cfg.model.num_mask_pixels * 100,
            int(self.samples_per_group[0]),
            int(self.samples_per_group[1]),
            int(self.samples_per_group[2]),
            int(self.samples_per_group[3]),
            int(self.samples_per_group[4]),
            int(self.samples_per_group[5]),
            int(self.samples_per_group[6]),
        ), flush=True)

        return loss

    def configure_optimizers(self):
        return Ranger(self.net.parameters(), lr=self.lr, weight_decay=self.cfg.weight_decay)


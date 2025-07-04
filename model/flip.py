import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple
from einops import rearrange, repeat, reduce
import os
from nn.encoder_sparse import FlipEncoder
import math

class Flip(nn.Module):
    def __init__(
        self,
        cfg,
        model_path,
    ):
        super(Flip, self).__init__()

        self.cfg = cfg
        bottleneck_selection = max if cfg.bottleneck_selection == 'max' else min

        self.encoder = FlipEncoder(
            layers               = cfg.encoder.layers,
            input_channels       = 3,
            channels             = cfg.encoder.channels,
            num_registers        = cfg.encoder.num_registers,
            head_size            = cfg.encoder.head_size,
            memory_efficient     = cfg.memory_efficient,
            bottleneck_selection = bottleneck_selection,
            min_patch_size       = cfg.min_patch_size,
            max_patch_size       = cfg.max_patch_size,
            alpha                = cfg.encoder.alpha,
            kq_expansion         = cfg.encoder.kq_expansion,
            v_expansion          = cfg.encoder.v_expansion,
            mlp_expansion        = cfg.encoder.mlp_expansion
        )

    def forward(
        self, 
        input_patches, 
        input_coordinates, 
        target_indices, 
        seq_lengths, 
        resolutions, 
        input_position, 
        gt_position, 
        gt_bbox,
        mask_targets,
        mask_coordinates,
        mask_seq_lengths
    ):
        B      = gt_position.shape[0]
        device = gt_position.device
        mask_targets = mask_targets.float()

        # convert input_patches from RGB to float
        input_patches = [p.float() / 255 for p in input_patches]

        mask_samples = self.encoder(
            input_patches    = input_patches,
            position         = input_position,
            coordinates      = input_coordinates,
            target_indices   = target_indices,
            seq_lengths      = seq_lengths,
            mask_coordinates = mask_coordinates,
            mask_seq_lengths = mask_seq_lengths
        )

        mask_loss = F.binary_cross_entropy_with_logits(mask_samples, mask_targets)

        with th.no_grad():
            mask_samples = rearrange((mask_samples > 0).float(), '(b n) -> b n', b=B, n=self.cfg.num_mask_pixels)
            mask_targets = rearrange(mask_targets, '(b n) -> b n', b=B, n=self.cfg.num_mask_pixels)

            intersection = (mask_samples * mask_targets).sum(dim=-1)
            union        = (mask_samples + mask_targets).sum(dim=-1) - intersection
            mask_iou     = intersection / union.clamp(min=1e-6)

        return {
            'mask_loss'           : mask_loss,
            'mask_iou'            : mask_iou.mean(),
            'mask_predictions'    : mask_samples.detach(),
            'mask_targets'        : mask_targets.detach()
        }

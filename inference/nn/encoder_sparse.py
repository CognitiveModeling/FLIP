import torch.nn as nn
import torch as th
import numpy as np
from nn.sparse_resolution_independent import FovealPatchSelection, SparseSelfAttentionLayer, SparseMaskExtraction
import torch.nn.functional as F

from typing import Tuple, Union, List
import utils
import cv2

__author__ = "Manuel Traub"

class MultiArgSequential(nn.Sequential):
    def __init__(self, *args, **kwargs):
        super(MultiArgSequential, self).__init__(*args, **kwargs)

    def forward(self, *tensor):

        for n in range(len(self)):
            if isinstance(tensor, th.Tensor) or tensor == None:
                tensor = self[n](tensor)
            else:
                tensor = self[n](*tensor)

        return tensor

class FlipEncoder(nn.Module):
    def __init__(
        self,
        layers,
        input_channels,
        channels,
        head_size,
        num_registers = 8,
        bottleneck_selection = max,
        min_patch_size = 1,
        max_patch_size = 16,
        alpha=0.1,
        kq_expansion=1,
        v_expansion=1,
        mlp_expansion=4
    ):
        super(FlipEncoder, self).__init__()
        print(f"Encoder input_channels = {input_channels}")

        self.sparsify = FovealPatchSelection(
            num_inputs       = input_channels,
            initial_channels = channels,
            num_registers    = num_registers,
            min_patch_size   = min_patch_size,
            max_patch_size   = max_patch_size,
        )

        self.mask_extractor = SparseMaskExtraction(
            channels, head_size, alpha=alpha, kq_expansion=kq_expansion, v_expansion=v_expansion, mlp_expansion=mlp_expansion
        )

        self.layers = MultiArgSequential(
            *[SparseSelfAttentionLayer(
                channels, head_size, bottleneck_selection,
                alpha=alpha, kq_expansion=kq_expansion, v_expansion=v_expansion, mlp_expansion=mlp_expansion
            ) for _ in range(layers)]
        )

    def get_input_probability(self):
        return self.sparsify.foeval_selection.patch_mask

    def prepare_kv_cache(
        self, 
        input_patches_p1,
        input_patches_p2,
        input_patches_p4,
        input_patches_p8,
        input_patches_p16,
        coordinates_p1,
        coordinates_p2,
        coordinates_p4,
        coordinates_p8,
        coordinates_p16,
        position: th.Tensor,
    ):
        latent, args = self.sparsify(
            input_patches_p1,  coordinates_p1,
            input_patches_p2,  coordinates_p2,
            input_patches_p4,  coordinates_p4,
            input_patches_p8,  coordinates_p8,
            input_patches_p16, coordinates_p16,
            position
        )
        
        embedding     = args['embedding']
        coordinates   = args['coordinates']
        
        latent = self.layers(latent, embedding)[0]

        return self.mask_extractor.prepare_kv_cache(
            latent,
            position, # need to be the input position so that the embeddings are aligned
            embedding,
        )

    def forward_with_cache(
        self,
        position: th.Tensor,
        mask_coordinates,
        k_cached: th.Tensor,
        v_cached: th.Tensor
    ):
        return self.mask_extractor.forward_with_cache(
            position, 
            mask_coordinates, 
            k_cached, 
            v_cached
        )
            
    def forward(
        self, 
        input_patches_p1,
        input_patches_p2,
        input_patches_p4,
        input_patches_p8,
        input_patches_p16,
        coordinates_p1,
        coordinates_p2,
        coordinates_p4,
        coordinates_p8,
        coordinates_p16,
        position: th.Tensor,
        mask_coordinates
    ):
        latent, args = self.sparsify(
            input_patches_p1,  coordinates_p1,
            input_patches_p2,  coordinates_p2,
            input_patches_p4,  coordinates_p4,
            input_patches_p8,  coordinates_p8,
            input_patches_p16, coordinates_p16,
            position
        )
        
        embedding     = args['embedding']
        coordinates   = args['coordinates']
        
        latent = self.layers(latent, embedding)[0]

        mask = self.mask_extractor(
            latent,
            position, # need to be the input position so that the embeddings are aligned
            embedding,
            mask_coordinates,
        )

        return mask


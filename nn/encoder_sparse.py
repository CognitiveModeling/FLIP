import torch.nn as nn
import torch as th
import numpy as np
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce
from nn.sparse_resolution_independent import FovealPatchSelection, SparseSelfAttentionLayer, SparseMaskExtraction

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


__author__ = "Manuel Traub"

class FlipEncoder(nn.Module):
    def __init__(
        self,
        layers,
        input_channels,
        channels,
        head_size,
        num_registers,
        memory_efficient,
        bottleneck_selection,
        min_patch_size,
        max_patch_size,
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
            memory_efficient = memory_efficient,
        )

        self.mask_extractor = SparseMaskExtraction(
            channels, head_size, memory_efficient, 
            alpha=alpha, kq_expansion=kq_expansion, v_expansion=v_expansion, mlp_expansion=mlp_expansion
        )

        self.layers = MultiArgSequential(
            *[SparseSelfAttentionLayer(
                channels, head_size, memory_efficient, bottleneck_selection,
                alpha=alpha, kq_expansion=kq_expansion, v_expansion=v_expansion, mlp_expansion=mlp_expansion
            ) for _ in range(layers)]
        )

    def get_input_probability(self):
        return self.sparsify.foeval_selection.patch_mask

    def forward(
        self, 
        input_patches: th.Tensor,
        position: th.Tensor,
        coordinates,
        target_indices,
        seq_lengths: th.Tensor,
        mask_coordinates = None,
        mask_seq_lengths = None,
    ):
        latent, args = self.sparsify(input_patches, position, coordinates, target_indices, seq_lengths)
        
        embedding     = args['embedding']
        coordinates   = args['coordinates']
        register_mask = args['register_mask']
        seq_lengths   = args['seq_lengths']
        
        latent = self.layers(latent, embedding, args['attn_bias'])[0]

        mask = self.mask_extractor(
            latent,
            position, # need to be the input position so that the embeddings are aligned
            embedding,
            mask_coordinates,
            seq_lengths,
            mask_seq_lengths,
        )

        return mask

    def inference(
        self, 
        input_patches: th.Tensor,
        position: th.Tensor,
        coordinates,
        target_indices,
        seq_lengths: th.Tensor,
        mask_bbox,
        mask_sizes,
        uncertain_threshold=0.01,
    ):
        latent, args = self.sparsify(input_patches, position, coordinates, target_indices, seq_lengths)
        
        embedding     = args['embedding']
        coordinates   = args['coordinates']
        register_mask = args['register_mask']
        seq_lengths   = args['seq_lengths']
        
        latent = self.layers(latent, embedding, args['attn_bias'])[0]

        mask = self.mask_extractor.inference(
            latent,
            position, # need to be the input position so that the embeddings are aligned
            embedding,
            seq_lengths,
            mask_bbox,
            mask_sizes,
            uncertain_threshold,
        )

        return mask

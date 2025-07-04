import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from xformers.ops.fmha import BlockDiagonalMask
from xformers.ops import memory_efficient_attention
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from torch.utils.checkpoint import checkpoint
from nn.complex_gaus2d_resolution_independent import ComplexGaus2D, ComplexEmbedding


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_channels, hidden_channels, out_channels, memory_efficient):
        super(PatchEmbedding, self).__init__()
        self.memory_efficient = memory_efficient

        if patch_size <= 4:
            self.layers = nn.Sequential(
                Rearrange('b c h w -> b (c h w)'),
                nn.Linear(in_channels, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, out_channels),
            )
        if patch_size == 8:
            self.layers = nn.Sequential(
                Rearrange('b c (h h1) (w w1) -> b h w (c h1 w1)', h=4, w=4, h1 = 2, w1 = 2),
                nn.Linear(in_channels // 16, hidden_channels // 16),
                Rearrange('b h w c -> b (h w c)'),
                nn.SiLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, out_channels),
            )
        elif patch_size == 16:
            self.layers = nn.Sequential(
                Rearrange('b c (h h1) (w w1) -> b h w (c h1 w1)', h=4, w=4, h1 = 4, w1 = 4),
                nn.Linear(in_channels // 16, hidden_channels // 16),
                Rearrange('b h w c -> b (h w c)'),
                nn.SiLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, out_channels),
            )
        elif patch_size == 32:
            self.layers = nn.Sequential(
                Rearrange('b c (h h1) (w w1) -> b h w (c h1 w1)', h=8, w=8, h1 = 4, w1 = 4),
                nn.Linear(in_channels // 64, hidden_channels // 16),
                Rearrange('b (h h1) (w w1) c -> b h w (h1 w1 c)', h1 = 2, w1 = 2, h = 4, w = 4),
                nn.SiLU(),
                nn.Linear(hidden_channels // 4, hidden_channels // 16),
                Rearrange('b h w c -> b (h w c)'),
                nn.SiLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, out_channels),
            )
        elif patch_size == 64:
            self.layers = nn.Sequential(
                Rearrange('b c (h h1) (w w1) -> b h w (c h1 w1)', h=16, w=16, h1 = 4, w1 = 4),
                nn.Linear(in_channels // 256, hidden_channels // 16),
                Rearrange('b (h h1) (w w1) c -> b h w (h1 w1 c)', h1 = 4, w1 = 4, h = 4, w = 4),
                nn.SiLU(),
                nn.Linear(hidden_channels, hidden_channels // 16),
                Rearrange('b h w c -> b (h w c)'),
                nn.SiLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, out_channels),
            )
        
    def forward(self, x):
        if self.memory_efficient:
            return checkpoint(self.layers, x, use_reentrant=False)
        return self.layers(x)

class FovealPatchSelection(nn.Module):
    def __init__(
        self, 
        num_inputs,
        min_patch_size, 
        max_patch_size, 
        initial_channels,
        num_registers,
        memory_efficient
    ):
        super(FovealPatchSelection, self).__init__()
        self.num_registers  = num_registers
        self.num_outputs    = initial_channels
        self.min_patch_size = min_patch_size
        min_patch_size = int(np.log2(min_patch_size))
        max_patch_size = int(np.log2(max_patch_size))
        self.register_buffer(
            'patch_sizes', 2**th.linspace(min_patch_size, max_patch_size, max_patch_size - min_patch_size + 1)
        )

        self.patch_embeddings = nn.Sequential(
            *[PatchEmbedding( 
                p, num_inputs * p**2, initial_channels*4, initial_channels, memory_efficient
            ) for p in self.patch_sizes.long().cpu().numpy()]
        )

        self.embedding  = ComplexEmbedding()

        self.registers = nn.Parameter(th.randn(num_registers, initial_channels))
        self.register_embeddings  = nn.Parameter(th.randn(num_registers, 2)*0.25-5)
        self.register_coordinates = nn.Parameter(th.randn(num_registers, 2)*0.25+5)

    def forward(self, input_patches, input_position, coordinates, target_indices, seq_lengths):
        """
        input_patches: variable list of tensorswith patches for each resolution
        input_position: tensor with position for each sample in the batch
        coordinates: list of tensors with coordinates for each patch and resolution
        target_indices: list of tensors with sample relative target indices for each patch and resolution 
        seq_lengths: tensor with the number of tokens for each resolution and each sample (B, N)
        """
        device = input_position.device
        B = input_position.shape[0]

        num_tokens       = th.stack(seq_lengths, dim=1).sum(dim=1)
        total_num_tokens = num_tokens + self.num_registers
        total_tokens     = total_num_tokens.sum()

        register_offset  = th.cat((th.zeros((1,), device=device), th.cumsum(total_num_tokens[:-1], dim=0)), dim=0).long()
        register_indices = repeat(th.arange(self.num_registers, device=device), 'n -> (b n)', b = B) + repeat(register_offset, 'b -> (b n)', n = self.num_registers)
        target_indices   = [target_indices[i] + th.repeat_interleave(register_offset + self.num_registers, seq_lengths[i], dim=0) for i, target_idx in enumerate(target_indices)]
        
        register_mask = th.zeros(total_tokens, dtype=th.bool, device=device)
        register_mask[register_indices] = True
        
        output = th.zeros((total_tokens, self.num_outputs), device=device)
        output_coordinates = th.zeros((total_tokens, 2), device=device)
        for i in range(len(self.patch_sizes)):
            if input_patches[i].shape[0] > 0:
                output[target_indices[i]] = self.patch_embeddings[i](input_patches[i])
                output_coordinates[target_indices[i]] = coordinates[i]

        output[register_indices] = repeat(self.registers, 'n c -> (b n) c', b = B)
        output_coordinates[register_mask]  = repeat(self.register_coordinates, 'n c -> (b n) c', b = B)

        output_embedding                 = th.zeros((total_tokens, 2), device=device)
        output_embedding[~register_mask] = self.embedding(input_position, output_coordinates[~register_mask], num_tokens)
        output_embedding[register_mask]  = repeat(self.register_embeddings, 'n c -> (b n) c', b = B)

        attn_bias = BlockDiagonalMask.from_seqlens(total_num_tokens.detach().cpu().numpy().tolist())

        return output, {
            'seq_lengths': total_num_tokens,
            'attn_bias': attn_bias,
            'register_mask': register_mask.view(-1, 1).float(),
            'embedding': output_embedding,
            'coordinates': output_coordinates,
        }

class SparseMaskAttention(nn.Module):
    def __init__(self, num_hidden, head_size, alpha = 0.1, kq_expansion = 4, v_expansion = 4):
        super(SparseMaskAttention, self).__init__()
        self.num_heads = max((kq_expansion * num_hidden) // head_size, 1)

        self.key = nn.Sequential(
            nn.Linear(2, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, num_hidden * v_expansion), # v_expansion since we compute actual key after combining with embedding
        )

        self.query = nn.Sequential(
            nn.Linear(2, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, num_hidden * kq_expansion),
            Rearrange('b (h d) -> 1 b h d', h = self.num_heads)
        )

        self.value = nn.Sequential(
            nn.Linear(2, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, num_hidden * v_expansion),
            Rearrange('b (h d) -> 1 b h d', h = self.num_heads)
        )

        self.k = nn.Sequential(
            nn.LayerNorm(num_hidden),
            nn.Linear(num_hidden, num_hidden * v_expansion), # v_expansion since we compute actual key after combining with embedding
        )
        self.v = nn.Sequential(
            nn.LayerNorm(num_hidden),
            nn.Linear(num_hidden, num_hidden * v_expansion),
            Rearrange('b (h d) -> 1 b h d', h = self.num_heads)
        )

        self.k_mlp = nn.Sequential(
            nn.Linear(num_hidden * 2 * v_expansion, num_hidden * v_expansion),
            nn.SiLU(),
            nn.Linear(num_hidden * v_expansion, num_hidden * kq_expansion),
            Rearrange('b (h d) -> 1 b h d', h = self.num_heads)
        )

        self.out = nn.Sequential(
            Rearrange('1 b h d -> b (h d)'),
            nn.Linear(num_hidden * v_expansion, num_hidden)
        )

    def forward(self, kv, key_embedding, query_embedding, attn_bias):
        q = self.query(query_embedding)
        k = self.k_mlp(th.cat((self.k(kv), self.key(key_embedding)), dim=1))
        v = self.v(kv) + self.value(key_embedding)
        o = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        return self.out(o)
        
    def prepare_kv_cache(self, kv, key_embedding):
        self.k_cached = self.k_mlp(th.cat((self.k(kv), self.key(key_embedding)), dim=1))
        self.v_cached = self.v(kv) + self.value(key_embedding)
    
    def forward_with_cache(self, query_embedding):
        q = self.query(query_embedding)
        o = memory_efficient_attention(q, self.k_cached, self.v_cached)
        return self.out(o)

class SparseMaskExtraction(nn.Module):
    def __init__(self, num_hidden, head_size, memory_efficient, alpha = 0.1, kq_expansion = 4, v_expansion = 4, mlp_expansion = 4):
        super(SparseMaskExtraction, self).__init__()
        self.memory_efficient = memory_efficient

        self.preprocess = FeedForwardBlock(num_hidden, num_hidden, memory_efficient, max, alpha=alpha, mlp_expansion=mlp_expansion)

        self.attention = SparseMaskAttention(num_hidden, head_size, alpha=alpha, kq_expansion=kq_expansion, v_expansion=v_expansion) 

        self.embedding  = ComplexEmbedding()
        self.postprocess = FeedForwardBlock(num_hidden, num_hidden, memory_efficient, max, alpha=alpha, mlp_expansion=mlp_expansion)
        self.to_mask     = nn.Linear(num_hidden, 1)


    def forward(self, x, input_position, input_embedding, target_coordinates, input_seq_lengths, target_seq_lengths):

        x = x + self.preprocess(x)
        target_embedding = self.embedding(input_position, target_coordinates, target_seq_lengths)

        attn_bias = BlockDiagonalMask.from_seqlens(
            target_seq_lengths.detach().cpu().numpy().tolist(),
            input_seq_lengths.detach().cpu().numpy().tolist(),
        )

        if self.memory_efficient:
            x = checkpoint(self.attention, x, input_embedding, target_embedding, attn_bias)
        else:
            x = self.attention(x, input_embedding, target_embedding, attn_bias)

        x = x + self.postprocess(x)
        return self.to_mask(x).squeeze(-1)

    def inference(self, x, input_position, input_embedding, input_seq_lengths, bbox, sizes, uncertain_threshold=0.1):

        x = x + self.preprocess(x)
        
        self.attention.prepare_kv_cache(x, input_embedding)
        
        min_x, min_y, max_x, max_y = bbox
        mask = None
        for i in range(len(sizes)):
            len_h, len_w = sizes[i]

            target_coordinates = th.stack(th.meshgrid(
                th.linspace(min_x, max_x, len_w, device=x.device),
                th.linspace(min_y, max_y, len_h, device=x.device),
            indexing='xy'), dim=-1).view(-1, 2)
            
            # select all uncertain pixels (below threshold and above 1-threshold)
            if mask is not None:
                uncertain_mask = (mask > uncertain_threshold) & (mask < 1 - uncertain_threshold)
                while uncertain_mask.sum() > 1000000:
                    uncertain_threshold = uncertain_threshold * 2
                    uncertain_mask = (mask > uncertain_threshold) & (mask < 1 - uncertain_threshold)
                target_coordinates = target_coordinates[uncertain_mask]

            if target_coordinates.shape[0] != 0:
                target_seq_lengths = th.tensor([target_coordinates.shape[0]], device=x.device)
                target_embedding = self.embedding(input_position, target_coordinates, target_seq_lengths)


                attn_out = self.attention.forward_with_cache(target_embedding)
                logits = self.to_mask(attn_out + self.postprocess(attn_out)).squeeze(-1)

                if mask is None:
                    mask = th.sigmoid(logits)
                else:
                    mask[uncertain_mask] = th.sigmoid(logits)

            if i < len(sizes) - 1:
                mask = F.interpolate(mask.view(1, 1, len_h, len_w), size=sizes[i+1], mode='bilinear', align_corners=False).view(-1)

        return mask

class SparseSelfAttention(nn.Module):
    def __init__(self, num_hidden, head_size, alpha = 0.1, kq_expansion = 4, v_expansion = 4):
        super(SparseSelfAttention, self).__init__()
        self.num_heads = max((kq_expansion * num_hidden) // head_size, 1)

        self.key = nn.Sequential(
            nn.Linear(2, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, num_hidden * kq_expansion),
            Rearrange('b (h d) -> 1 b h d', h = self.num_heads)
        )

        self.query = nn.Sequential(
            nn.Linear(2, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, num_hidden * kq_expansion),
            Rearrange('b (h d) -> 1 b h d', h = self.num_heads)
        )

        self.value = nn.Sequential(
            nn.Linear(2, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, num_hidden * v_expansion),
            Rearrange('b (h d) -> 1 b h d', h = self.num_heads)
        )

        self.q = nn.Sequential(
            nn.LayerNorm(num_hidden),
            nn.Linear(num_hidden, num_hidden * kq_expansion),
            Rearrange('b (h d) -> 1 b h d', h = self.num_heads)
        )
        self.k = nn.Sequential(
            nn.LayerNorm(num_hidden),
            nn.Linear(num_hidden, num_hidden * kq_expansion),
            Rearrange('b (h d) -> 1 b h d', h = self.num_heads)
        )
        self.v = nn.Sequential(
            nn.LayerNorm(num_hidden),
            nn.Linear(num_hidden, num_hidden * v_expansion),
            Rearrange('b (h d) -> 1 b h d', h = self.num_heads)
        )

        self.out = nn.Sequential(
            Rearrange('1 b h d -> b (h d)'),
            nn.Linear(num_hidden * v_expansion, num_hidden)
        )

        self.alpha = nn.Parameter(th.ones(1)*alpha)

    def forward(self, x, embedding, attn_bias):
        q, k, v = self.q(x), self.k(x), self.v(x)
        q = q + self.query(embedding)
        k = k + self.key(embedding)
        v = v + self.value(embedding)
        o = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        return self.out(o) * self.alpha

class FeedForwardBlock(nn.Module):
    def __init__(
            self, 
            in_features, 
            out_features, 
            memory_efficient, 
            bottleneck_selection,
            alpha = 0.1,
            mlp_expansion = 4
        ):
        super(FeedForwardBlock, self).__init__()
        self.memory_efficient = memory_efficient
        
        hidden_features = bottleneck_selection(in_features, out_features) * mlp_expansion
        self.weight1 = th.nn.Parameter(th.randn(hidden_features, in_features))
        self.bias1   = th.nn.Parameter(th.zeros(hidden_features))
        self.weight2 = th.nn.Parameter(th.randn(out_features, hidden_features))
        self.bias2   = th.nn.Parameter(th.zeros(out_features))

        th.nn.init.xavier_uniform_(self.weight1)
        th.nn.init.xavier_uniform_(self.weight2)

        self.norm = nn.LayerNorm(in_features)

        self.alpha = nn.Parameter(th.ones(1)*alpha)

    def _forward(self, x):
        x = x @ self.weight1.t() + self.bias1
        x = F.silu(x)
        x = x @ self.weight2.t() + self.bias2
        return x * self.alpha

    def forward(self, x):
        x = self.norm(x)

        if self.memory_efficient:
            return checkpoint(self._forward, x, use_reentrant=False)

        return self._forward(x)

class SparseSelfAttentionLayer(nn.Module):
    def __init__(
        self,
        num_hidden,
        head_size,
        memory_efficient,
        bottleneck_selection,
        alpha = 0.1,
        kq_expansion = 4,
        v_expansion = 4,
        mlp_expansion = 4
    ):
        super(SparseSelfAttentionLayer, self).__init__()
        self.num_heads = num_hidden // head_size
        self.memory_efficient = memory_efficient

        self.attention = SparseSelfAttention(num_hidden, head_size, alpha, kq_expansion, v_expansion)
        self.mlp = FeedForwardBlock(
            num_hidden, num_hidden, memory_efficient, bottleneck_selection, 
            alpha=alpha, mlp_expansion=mlp_expansion
        )
        self.alpha     = nn.Parameter(th.ones(1)*alpha)

    def forward(self, x, embedding, attn_bias):

        if self.memory_efficient:
            x = x + checkpoint(self.attention, x, embedding, attn_bias, use_reentrant=False)
        else:
            x = x + self.attention(x, embedding, attn_bias)

        return x + self.mlp(x) * self.alpha, embedding, attn_bias

import argparse
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import matplotlib.pyplot as plt
from PIL import Image
import math
from typing import Tuple

def sparse_embedding(input, coordinates):
    x, y, std_x, std_y, rot_a, rot_b = th.split(input, 1, dim=-1)

    # normalize rotation
    scale = th.sqrt(rot_a**2 + rot_b**2)
    scale = th.clamp(scale, min=1e-6)
    rot_a = rot_a / scale
    rot_b = rot_b / scale

    # Clip values for stability
    # Clip values for stability
    std_x = th.clip(std_x, 1e-8, None)
    std_y = th.clip(std_y, 1e-8, None)

    # compute relative coordinates
    x = coordinates[:,0:1] - x
    y = coordinates[:,1:2] - y

    # Compute rotated coordinates
    x_rot = rot_a * x - rot_b * y
    y_rot = rot_b * x + rot_a * y

    # Compute Gaussian distribution with rotated coordinates
    z_x = x_rot / std_x
    z_y = y_rot / std_y

    return th.clip(th.cat([z_x, z_y], dim=1), -5, 5)


@th.no_grad()                           # no backward pass required
def scaled_dot_product_attention(q: th.Tensor, k: th.Tensor, v: th.Tensor) -> th.Tensor:
    # --- scale ----------------------------------------------------------------
    d_k = q.shape[-1]                       # D (head dim) is static at export
    scale = 1.0 / math.sqrt(d_k)            # Python scalar → Constant node

    # --- attention weights ----------------------------------------------------
    # (1,H,N,D) @ (1,H,D,N) → (1,H,N,N)
    scores = th.matmul(q, k.transpose(-2, -1)) * scale
    attn   = th.softmax(scores, dim=-1)  # Softmax-13

    # --- weighted sum ---------------------------------------------------------
    # (1,H,N,N) @ (1,H,N,D) → (1,H,N,D)
    out = th.matmul(attn, v)
    return out

class Flatten(nn.Module):
    """th.flatten over an arbitrary span of dimensions."""
    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        super().__init__()
        self.start_dim, self.end_dim = start_dim, end_dim

    def forward(self, x):
        return th.flatten(x, self.start_dim, self.end_dim)


class Patchify(nn.Module):
    """
    Split an image into a regular (h×w) grid of (h1×w1) sub-patches
    and move the sub-patch pixels into the channel dimension:

        (B,C, h·h1, w·w1)  →  (B, h, w, C·h1·w1)
    """
    def __init__(self, h: int, h1: int, w: int, w1: int):
        super().__init__()
        self.h, self.h1, self.w, self.w1 = h, h1, w, w1

    def forward(self, x):
        b, c, _, _ = x.shape                         # (B,C,H,W)
        x = x.reshape(b, c, self.h, self.h1, self.w, self.w1)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous() # (B,h,w,C,h1,w1)
        x = x.reshape(b, self.h, self.w, c * self.h1 * self.w1)
        return x                                     # (B,h,w,⋯)


class MergeTokens(nn.Module):
    """(B, h, w, c) → (B, h·w·c) — flatten spatial + channel dims."""
    def forward(self, x):
        # th.flatten works for zero-sized batches whereas reshape(-1) does not.
        return th.flatten(x, 1)     


class Regrid(nn.Module):
    """
    (B, h·h1, w·w1, C) → (B, h, w, h1·w1·C)

    Used in the 32 & 64-pixel branches to fuse two levels of grids.
    """
    def __init__(self, h: int, h1: int, w: int, w1: int):
        super().__init__()
        self.h, self.h1, self.w, self.w1 = h, h1, w, w1

    def forward(self, x):
        b, H, W, C = x.shape
        x = x.reshape(b, self.h, self.h1, self.w, self.w1, C)      # (B,h,h1,w,w1,C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()               # (B,h,w,h1,w1,C)
        x = x.reshape(b, self.h, self.w, C * self.h1 * self.w1)
        return x


class SplitHeads(nn.Module):
    """
    (B, H·D) → (1, H, B, D)
    """
    def __init__(self, num_heads: int):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, x):
        b, hd = x.shape
        h = self.num_heads
        d = hd // h
        x = x.reshape(b, h, d)          # (B, H, D)
        x = x.unsqueeze(0)              # (1, B, H, D)
        x = x.permute(0, 2, 1, 3)       # (1, H, B, D)
        return x


class MergeHeads(nn.Module):
    """
    (1, H, B, D) → (B, H·D)

    Inverse of SplitHeads:
    """
    def forward(self, x):
        x = x.squeeze(0)                # (H, B, D)
        x = x.permute(1, 0, 2).contiguous()  # (B, H, D)
        return th.flatten(x, 1)      # (B, H·D)  – works for B = 0 too

# ---------- PatchEmbedding (no einops / Rearrange) ---------------------------

class PatchEmbedding(nn.Module):
    """
    A pure-PyTorch/ONNX rewrite of the original einops-based module.
    The algebra is identical; only the tensor plumbing changed.
    """
    def __init__(self, patch_size: int,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int) -> None:
        super().__init__()

        if patch_size <= 4:                                                # ---------------------------------
            self.layers = nn.Sequential(                                   #  b c h w  →  b (c·h·w)
                Flatten(1),
                nn.Linear(in_channels, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, out_channels),
            )

        elif patch_size == 8:                                              # ---------------------------------
            self.layers = nn.Sequential(                                   # 4×4 grid, 2×2 sub-patch
                Patchify(h=4, h1=2, w=4, w1=2),
                nn.Linear(in_channels // 16, hidden_channels // 16),
                MergeTokens(),                                             # (B, 4·4·⋯) = hidden_channels
                nn.SiLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, out_channels),
            )

        elif patch_size == 16:                                             # ---------------------------------
            self.layers = nn.Sequential(                                   # 4×4 grid, 4×4 sub-patch
                Patchify(h=4, h1=4, w=4, w1=4),
                nn.Linear(in_channels // 16, hidden_channels // 16),
                MergeTokens(),
                nn.SiLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, out_channels),
            )

        elif patch_size == 32:                                             # ---------------------------------
            self.layers = nn.Sequential(                                   # 8×8 grid → 4×4 grid collapse
                Patchify(h=8, h1=4, w=8, w1=4),
                nn.Linear(in_channels // 64, hidden_channels // 16),
                Regrid(h=4, h1=2, w=4, w1=2),
                nn.SiLU(),
                nn.Linear(hidden_channels // 4, hidden_channels // 16),
                MergeTokens(),
                nn.SiLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, out_channels),
            )

        elif patch_size == 64:                                             # ---------------------------------
            self.layers = nn.Sequential(                                   # 16×16 grid → 4×4 grid collapse
                Patchify(h=16, h1=4, w=16, w1=4),
                nn.Linear(in_channels // 256, hidden_channels // 16),
                Regrid(h=4, h1=4, w=4, w1=4),
                nn.SiLU(),
                nn.Linear(hidden_channels, hidden_channels // 16),
                MergeTokens(),
                nn.SiLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, out_channels),
            )

        else:
            raise ValueError(f"Unsupported patch_size = {patch_size}")

    def forward(self, x):
        return self.layers(x)

class FovealPatchSelection(nn.Module):
    """
    Un-rolled replacement: patch sizes are hard-wired to
        1×1, 2×2, 4×4, 8×8, 16×16.
    Batch size is assumed to be 1 during inference, which is what the
    demo requires.
    """

    def __init__(self,
                 num_inputs:       int,
                 min_patch_size:   int,        # kept only for API compatibility
                 max_patch_size:   int,        # (ignored)
                 initial_channels: int,
                 num_registers:    int):
        super().__init__()

        C = initial_channels

        # ---------- one explicit PatchEmbedding per resolution ---------------
        self.patch_embeddings = nn.Sequential(
            PatchEmbedding( 1, num_inputs *  1 *  1, C * 4, C),
            PatchEmbedding( 2, num_inputs *  2 *  2, C * 4, C),
            PatchEmbedding( 4, num_inputs *  4 *  4, C * 4, C),
            PatchEmbedding( 8, num_inputs *  8 *  8, C * 4, C),
            PatchEmbedding(16, num_inputs * 16 * 16, C * 4, C)
        )

        # ---------- learnable registers --------------------------------------
        self.num_registers        = num_registers
        self.registers            = nn.Parameter(th.randn(num_registers, C))
        self.register_embeddings  = nn.Parameter(th.randn(num_registers, 2) * 0.25 - 5.0)
        self.register_coordinates = nn.Parameter(th.randn(num_registers, 2) * 0.25 + 5.0)

    # -------------------------------------------------------------------------
    # helper: embeds exactly one resolution (no loops inside)                  
    # -------------------------------------------------------------------------
    def _embed_one(self,
                   patches: th.Tensor,
                   coords:  th.Tensor,
                   patch_embedding: nn.Module,
                   input_position: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Returns (token_vecs, pos_embs), both with first dim = Nk.
        Called five times from forward.
        """
        tokens   = patch_embedding(patches)                       # (Nk, C)
        pos_embs = sparse_embedding(input_position, coords)
        return tokens, pos_embs

    # -------------------------------------------------------------------------
    # forward: absolutely no Python loops / branches                           
    # -------------------------------------------------------------------------
    def forward(
        self,
        patches_1:  th.Tensor, coords_1:  th.Tensor,
        patches_2:  th.Tensor, coords_2:  th.Tensor,
        patches_4:  th.Tensor, coords_4:  th.Tensor,
        patches_8:  th.Tensor, coords_8:  th.Tensor,
        patches_16: th.Tensor, coords_16: th.Tensor,
        input_position: th.Tensor
    ) -> Tuple[th.Tensor, dict]:

        # ---- per-resolution embeddings (five explicit calls) ----------------
        tok_1,  emb_1  = self._embed_one(patches_1,  coords_1,  self.patch_embeddings[0], input_position)
        tok_2,  emb_2  = self._embed_one(patches_2,  coords_2,  self.patch_embeddings[1], input_position)
        tok_4,  emb_4  = self._embed_one(patches_4,  coords_4,  self.patch_embeddings[2], input_position)
        tok_8,  emb_8  = self._embed_one(patches_8,  coords_8,  self.patch_embeddings[3], input_position)
        tok_16, emb_16 = self._embed_one(patches_16, coords_16, self.patch_embeddings[4], input_position)

        # ---- concatenate variable-length parts ------------------------------
        toks_var   = th.cat((tok_1,  tok_2,  tok_4,  tok_8,  tok_16),  dim=0)
        coords_var = th.cat((coords_1, coords_2, coords_4, coords_8, coords_16), dim=0)
        emb_var    = th.cat((emb_1,  emb_2,  emb_4,  emb_8,  emb_16),  dim=0)

        # ---- append registers ------------------------------------------------
        toks_all   = th.cat((toks_var,   self.registers),             dim=0)
        coords_all = th.cat((coords_var, self.register_coordinates),  dim=0)
        emb_all    = th.cat((emb_var,    self.register_embeddings),   dim=0)

        return toks_all, {
            "embedding"     : emb_all,        # (Ntot,2)
            "coordinates"   : coords_all,     # (Ntot,2)
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
            SplitHeads(self.num_heads)
        )

        self.value = nn.Sequential(
            nn.Linear(2, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, num_hidden * v_expansion),
            SplitHeads(self.num_heads)
        )

        self.k = nn.Sequential(
            nn.LayerNorm(num_hidden),
            nn.Linear(num_hidden, num_hidden * v_expansion), # v_expansion since we compute actual key after combining with embedding
        )
        self.v = nn.Sequential(
            nn.LayerNorm(num_hidden),
            nn.Linear(num_hidden, num_hidden * v_expansion),
            SplitHeads(self.num_heads)          
        )

        self.k_mlp = nn.Sequential(
            nn.Linear(num_hidden * 2 * v_expansion, num_hidden * v_expansion),
            nn.SiLU(),
            nn.Linear(num_hidden * v_expansion, num_hidden * kq_expansion),
            SplitHeads(self.num_heads)          
        )

        self.out = nn.Sequential(
            MergeHeads(),                       
            nn.Linear(num_hidden * v_expansion, num_hidden)
        )

    def forward(self, kv, key_embedding, query_embedding):
        q = self.query(query_embedding)
        k = self.k_mlp(th.cat((self.k(kv), self.key(key_embedding)), dim=1))
        v = self.v(kv) + self.value(key_embedding)
        o = scaled_dot_product_attention(q, k, v)
        return self.out(o)
        
    def prepare_kv_cache(self, kv, key_embedding):
        k_cached = self.k_mlp(th.cat((self.k(kv), self.key(key_embedding)), dim=1))
        v_cached = self.v(kv) + self.value(key_embedding)
        return k_cached, v_cached
    
    def forward_with_cache(self, query_embedding, k_cached, v_cached) -> th.Tensor:
        q = self.query(query_embedding)
        o = scaled_dot_product_attention(q, k_cached, v_cached)
        return self.out(o)

class SparseMaskExtraction(nn.Module):
    def __init__(self, num_hidden, head_size, alpha = 0.1, kq_expansion = 4, v_expansion = 4, mlp_expansion = 4):
        super(SparseMaskExtraction, self).__init__()

        self.preprocess = FeedForwardBlock(num_hidden, num_hidden, max, alpha=alpha, mlp_expansion=mlp_expansion)

        self.attention = SparseMaskAttention(num_hidden, head_size, alpha=alpha, kq_expansion=kq_expansion, v_expansion=v_expansion) 

        self.postprocess = FeedForwardBlock(num_hidden, num_hidden, max, alpha=alpha, mlp_expansion=mlp_expansion)
        self.to_mask     = nn.Linear(num_hidden, 1)

    def prepare_kv_cache(self, x, input_position, input_embedding):
        """
        Prepare the key and value cache for the attention mechanism.
        This is used to avoid recomputing keys and values for the same inputs.
        """
        x = x + self.preprocess(x)
        return self.attention.prepare_kv_cache(x, input_embedding)

    def forward_with_cache(self, input_position, target_coordinates, k_cached, v_cached):
        """
        Forward pass using the cached keys and values.
        This is used to speed up the attention mechanism by avoiding recomputation.
        """
        target_embedding = sparse_embedding(input_position, target_coordinates)

        x = self.attention.forward_with_cache(target_embedding, k_cached, v_cached)

        x = x + self.postprocess(x)
        return self.to_mask(x).squeeze(-1)

    def forward(self, x, input_position, input_embedding, target_coordinates):

        x = x + self.preprocess(x)
        target_embedding = sparse_embedding(input_position, target_coordinates)

        x = self.attention(x, input_embedding, target_embedding)

        x = x + self.postprocess(x)
        return self.to_mask(x).squeeze(-1)

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
            SplitHeads(self.num_heads)
        )

        self.query = nn.Sequential(
            nn.Linear(2, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, num_hidden * kq_expansion),
            SplitHeads(self.num_heads)
        )

        self.value = nn.Sequential(
            nn.Linear(2, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, num_hidden * v_expansion),
            SplitHeads(self.num_heads)
        )

        self.q = nn.Sequential(
            nn.LayerNorm(num_hidden),
            nn.Linear(num_hidden, num_hidden * kq_expansion),
            SplitHeads(self.num_heads)
        )

        self.k = nn.Sequential(
            nn.LayerNorm(num_hidden),
            nn.Linear(num_hidden, num_hidden * kq_expansion),
            SplitHeads(self.num_heads)
        )

        self.v = nn.Sequential(
            nn.LayerNorm(num_hidden),
            nn.Linear(num_hidden, num_hidden * v_expansion),
            SplitHeads(self.num_heads)
        )

        self.out = nn.Sequential(
            MergeHeads(),
            nn.Linear(num_hidden * v_expansion, num_hidden)
        )

        self.alpha = nn.Parameter(th.ones(1)*alpha)

    def forward(self, x, embedding):
        q, k, v = self.q(x), self.k(x), self.v(x)
        q = q + self.query(embedding)
        k = k + self.key(embedding)
        v = v + self.value(embedding)
        o = scaled_dot_product_attention(q, k, v)
        return self.out(o) * self.alpha

class FeedForwardBlock(nn.Module):
    def __init__(
            self, 
            in_features, 
            out_features, 
            bottleneck_selection,
            alpha = 0.1,
            mlp_expansion = 4
        ):
        super(FeedForwardBlock, self).__init__()
        
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
        return self._forward(x)

class SparseSelfAttentionLayer(nn.Module):
    def __init__(
        self,
        num_hidden,
        head_size,
        bottleneck_selection,
        alpha = 0.1,
        kq_expansion = 4,
        v_expansion = 4,
        mlp_expansion = 4
    ):
        super(SparseSelfAttentionLayer, self).__init__()
        self.num_heads = num_hidden // head_size

        self.attention = SparseSelfAttention(num_hidden, head_size, alpha, kq_expansion, v_expansion)
        self.mlp = FeedForwardBlock(
            num_hidden, num_hidden, bottleneck_selection, 
            alpha=alpha, mlp_expansion=mlp_expansion
        )
        self.alpha     = nn.Parameter(th.ones(1)*alpha)

    def forward(self, x, embedding):
        x = x + self.attention(x, embedding)
        return x + self.mlp(x) * self.alpha, embedding

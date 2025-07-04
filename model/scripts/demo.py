#!/usr/bin/env python
"""
mask_extraction_single_image_interactive.py
==========================================
An interactive tool to prototype Flip‑Net masks with a *single* multi‑stage mouse
gesture instead of the coarse key‑based interface that existed previously.

––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
**Gesture grammar**

1. **Primary stroke** ­– *click‑&‑drag*  
   • **press LMB** → fixes Gaussian **centre μ** at down‑point.  
   • **drag while holding**  → live preview of **orientation θ** and
     **major standard‑deviation σ₁**.  The ellipse is clamped to an initial
     aspect‑ratio σ₁:σ₂ = 3:1, elongated along the drag vector.

2. **Minor tuning** – *mouse move* (button released)  
   • The moment the LMB is **released** the orientation θ and σ₁ are frozen.  
   • Subsequent horizontal motion (screen‑space Δx) controls the **minor
     deviation σ₂** (perpendicular to θ).  A live preview is still shown.

3. **Finalise** – *single LMB click*  
   • A second LMB click commits the Gaussian; the heavy Flip‑Net forward pass
     runs and the remaining three sub‑plots (patch layout & predicted mask)
     update.

Keys:  
  – **q** : quit

––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
Implementation notes
--------------------
*  The script now maintains a finite‑state‑machine with three states
   (IDLE→DRAW_PRIMARY→ADJUST_MINOR→IDLE).
*  Two update paths:
   – *fast preview* (heat‑map only, GPU ≪1 ms) on every mouse‑move.  
   – *full render* (runs the encoder) only when the Gaussian is finalised.
*  No more keyboard‑based σ / θ tweaks – the old handlers were removed for
   clarity.

Written for PyTorch ≥2.3 / Matplotlib 3.9.  Tested on CUDA 12.5 and MPS.
"""

import argparse
import os
import sys
import math
import enum
from dataclasses import dataclass

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import torch as th

from utils.configuration import Configuration
from model.lightning.flip import FlipModule
from nn.complex_gaus2d import ComplexGaus2D
from ext.position_wrapper import sample_continuous_patches

# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# Utility classes
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

class InteractionStage(enum.Enum):
    IDLE = 0              # waiting for first click
    DRAW_PRIMARY = 1      # LMB held – define θ and σ_major
    ADJUST_MINOR = 2      # LMB released – tweak σ_minor until next click

@dataclass
class InteractionState:
    stage: InteractionStage = InteractionStage.IDLE
    centre_px: tuple[int, int] | None = None      # (x, y) in pixel space
    σ_major_norm: float = 0.0                     # along θ (normalized)
    σ_minor_norm: float = 0.0                     # perpendicular (normalized)
    θ_rad: float = 0.0                            # orientation in radians
    release_px: tuple[int, int] | None = None     # reference for minor‑drag


class GaussianParameters:
    """Container that matches the Flip positional encoding convention."""

    def __init__(self):
        self.mu_x = 0.0
        self.mu_y = 0.0
        self.sigma_x = 0.2
        self.sigma_y = 0.2
        self.rot_a = 1.0
        self.rot_b = 0.0

    # ––– helpers –––
    def as_tensor(self) -> th.Tensor:
        return th.tensor([
            self.mu_x,
            self.mu_y,
            self.sigma_x,
            self.sigma_y,
            self.rot_a,
            self.rot_b,
        ], dtype=th.float32)

    # oriented update --------------------------------------------------------

    def update_from_interaction(self, istate: InteractionState, W: int, H: int):
        """Pull data from the interaction FSM and convert to Flip units."""
        if istate.centre_px is None:
            return  # nothing yet

        cx_pix, cy_pix = istate.centre_px

        # centre in [−1,1]
        self.mu_x = (cx_pix / W) * 2.0 - 1.0
        self.mu_y = (cy_pix / H) * 2.0 - 1.0

        # sigmas in normalised coordinates (same scaling as μ) – we treat the
        # *smaller* image dimension as the reference length such that σ≈1 spans
        # the whole image.
        scale_pix = min(W, H) / 2.0
        self.sigma_x = istate.σ_major_norm
        self.sigma_y = istate.σ_minor_norm

        # orientation unit‑vector (cosθ,sinθ)
        self.rot_a = math.cos(istate.θ_rad)
        self.rot_b = -math.sin(istate.θ_rad)

        # numerical safety
        eps = 1e-8
        nrm = math.hypot(self.rot_a, self.rot_b)
        self.rot_a /= (nrm + eps)
        self.rot_b /= (nrm + eps)

    # ––– debug –––
    def __repr__(self):
        ang = math.degrees(math.atan2(self.rot_b, self.rot_a))
        ar = self.sigma_x / (self.sigma_y + 1e-12)
        return (f"μ=({self.mu_x:+.3f},{self.mu_y:+.3f}) σ=({self.sigma_x:.3f},{self.sigma_y:.3f}) "
                f"θ={ang:+6.1f}° AR={ar:.2f}")


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# Model‑side helpers (identical to previous script except for refactor)
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

def generate_grid_coordinates(H: int, W: int) -> th.Tensor:
    return th.stack(th.meshgrid(
        th.linspace(-1, 1, W) * W / 256,
        th.linspace(-1, 1, H) * H / 256,
        indexing="xy"), dim=-1).reshape(-1, 2)


def predict_full_mask(model, rgb_image, position_tensor, patch_sizes, device):
    H, W = rgb_image.shape[:2]
    num_tokens = model.cfg.model.avg_num_tokens
    patch_sizes_float = [float(p) for p in patch_sizes]

    input_patches, input_coordinates, target_indices, seq_lengths = sample_continuous_patches(
        rgb_image, position_tensor, num_tokens, patch_sizes_float)

    position_tensor = position_tensor.unsqueeze(0).to(device)
    input_patches = [p.to(device) for p in input_patches]
    input_coordinates = [c.to(device) for c in input_coordinates]
    target_indices = [t.to(device) for t in target_indices]
    seq_lengths = [s.to(device) for s in seq_lengths]

    mask_coordinates = generate_grid_coordinates(H, W).to(device)
    mask_seq_lengths = th.tensor([H * W], dtype=th.long, device=device)
    input_patches = [p.float() / 255 for p in input_patches]

    # scale μ/σ to Flip internal representation
    position_tensor[:, 0] *= W / 256
    position_tensor[:, 1] *= H / 256
    position_tensor[:, 2] *= W / 256
    position_tensor[:, 3] *= H / 256

    with th.no_grad():
        logits = model.net.encoder(
            input_patches=input_patches,
            position=position_tensor,
            coordinates=input_coordinates,
            target_indices=target_indices,
            seq_lengths=seq_lengths,
            mask_coordinates=mask_coordinates,
            mask_seq_lengths=mask_seq_lengths)

    mask = th.sigmoid(logits).cpu().numpy().reshape(H, W)
    return mask, input_patches, input_coordinates, target_indices, seq_lengths


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# Visual helpers (overlay functions identical to original)
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

# overlay_colored_patches  … (unchanged, omitted for brevity – same as user)
# overlay_gaussian          … (unchanged)

# Copy‑pasted from original for completeness – shortened.

def overlay_colored_patches(image, input_patches, input_coordinates, patch_sizes, height, width, blend_alpha=0.5):
    gray = image[...,0]*0.299 + image[...,1]*0.587 + image[...,2]*0.114
    base = np.stack([gray, gray, gray], axis=-1).astype(np.uint8)

    #colors = {1:(255,0,0),2:(0,255,0),4:(0,0,255),8:(255,255,0),16:(255,0,255),32:(0,255,255),64:(255,128,0)}
    colors = {1:(0,0,255),2:(0,255,0),4:(255,0,0),8:(0,255,255),16:(255,0,255),32:(255,255,0),64:(0,128,255)}
    for lvl,(patches,coords,ps) in enumerate(zip(input_patches, input_coordinates, patch_sizes)):
        col = np.asarray(colors.get(int(ps),(128,128,128)),dtype=np.float32)/255.0
        for p,xy in zip(patches.cpu().numpy(),coords.cpu().numpy()):
            cx = (xy[0]*128)+width/2
            cy = (xy[1]*128)+height/2
            tlx,tly = int(cx-ps/2), int(cy-ps/2)
            if tlx<0 or tly<0 or tlx+ps>width or tly+ps>height:
                continue
            patch = p.transpose(1,2,0)/255
            blend = np.clip((1-blend_alpha)*patch + blend_alpha*col,0,1)
            base[tly:tly+ps,tlx:tlx+ps] = (blend*255).astype(np.uint8)
    return base

def overlay_gaussian(img, heat, alpha=0.5, colormap=cv2.COLORMAP_JET):
    h = np.uint8(255*(heat-heat.min())/(heat.ptp()+1e-8))
    h_col = cv2.applyColorMap(h, colormap)
    return cv2.addWeighted(img, 1-alpha, h_col, alpha, 0)


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# Main application
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

def main():
    parser = argparse.ArgumentParser("Flip‑Net mask extractor (drag interface)")
    parser.add_argument("--image", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--port", type=int, default=29500)
    parser.add_argument("--device", default="cuda" if th.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = Configuration(args.config)
    cfg.seed = args.seed
    np.random.seed(cfg.seed)
    th.manual_seed(cfg.seed)
    if hasattr(cfg, "pretraining_stage"):
        cfg.pretraining_stage = 0
    cfg.model.batch_size = 1

    device = th.device(args.device)
    print("Loading Flip checkpoint", args.checkpoint)
    model = FlipModule.load_from_checkpoint(args.checkpoint, cfg=cfg, strict=False).to(device).eval()

    rgb = cv2.imread(args.image)
    if rgb is None:
        sys.exit(f"Could not read image {args.image}")
    H, W = rgb.shape[:2]

    # Gaussian proto‑layer
    gaus2d = ComplexGaus2D(size=(H, W)).to(device)

    # patch pyramid -----------------------------------------
    mn, mx = int(math.log2(cfg.model.min_patch_size)), int(math.log2(cfg.model.max_patch_size))
    patch_sizes = [int(2**p) for p in range(mn, mx+1)]

    # shared state ------------------------------------------
    istate = InteractionState()
    gparams = GaussianParameters()

    # matplotlib canvas -------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    plt.subplots_adjust(hspace=0.25,wspace=0.25)

    im_orig = axs[0,0].imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    axs[0,0].set_title("Original")
    axs[0,0].axis("off")

    im_gauss = axs[0,1].imshow(np.zeros_like(rgb))
    axs[0,1].set_title("Gaussian overlay")
    axs[0,1].axis("off")

    im_patches = axs[1,0].imshow(np.zeros_like(rgb))
    axs[1,0].set_title("Input patches")
    axs[1,0].axis("off")

    im_mask = axs[1,1].imshow(np.zeros_like(rgb))
    axs[1,1].set_title("Predicted mask")
    axs[1,1].axis("off")

    # legend for patch sizes
    colors_dict = {1:(1,0,0),2:(0,1,0),4:(0,0,1),8:(1,1,0),16:(1,0,1),32:(0,1,1),64:(1,0.5,0)}
    legend_elems = [Patch(facecolor=colors_dict[p], label=f"{p}×{p}") for p in patch_sizes if p in colors_dict]
    if legend_elems:
        axs[1,0].legend(handles=legend_elems, loc="upper right", fontsize=9)

    fig.suptitle("Flip‑Net mask extraction – drag interface", fontsize=16)

    # ––– rendering helpers –––

    def render_gaussian_only():
        """Fast path – update only subplot [0,1]"""
        # update Gaussian tensor
        gparams.update_from_interaction(istate, W, H)
        gp_t = gparams.as_tensor().to(device).unsqueeze(0)
        print("rendering Gaussian overlay", gp_t)
        heat = gaus2d(gp_t).squeeze().cpu().numpy()
        im_gauss.set_data(cv2.cvtColor(overlay_gaussian(rgb.copy(), heat, 0.5), cv2.COLOR_BGR2RGB))
        # annotate headline with current parameters
        fig.suptitle(f"Flip‑Net – {os.path.basename(args.image)}  |  {gparams}")
        fig.canvas.draw_idle()

    def full_render():
        gparams.update_from_interaction(istate, W, H)
        gp_t = gparams.as_tensor()
        input_position = gp_t.clone()
        input_position[2] *= 0.5**0.5
        input_position[3] *= W/H * 0.5**0.5
        # heavy call
        mask, in_patches, in_coords, tgt_idx, seq_len = predict_full_mask(model, rgb, input_position, patch_sizes, device)
        patches_vis = overlay_colored_patches(rgb, in_patches, in_coords, patch_sizes, H, W)
        heat = gaus2d(gp_t.unsqueeze(0).to(device)).squeeze().cpu().numpy()

        im_gauss.set_data(cv2.cvtColor(overlay_gaussian(rgb.copy(), heat, 0.5), cv2.COLOR_BGR2RGB))
        im_patches.set_data(cv2.cvtColor(patches_vis, cv2.COLOR_BGR2RGB))
        mask_rgb = np.zeros((H,W,3),dtype=np.uint8); mask_rgb[...,1]=(mask**2*255).astype(np.uint8)
        im_mask.set_data(mask_rgb)
        fig.suptitle(f"Flip‑Net – {os.path.basename(args.image)}  |  {gparams}")
        fig.canvas.draw_idle()

    # ––– event system –––

    def on_press(ev):
        nonlocal istate
        if ev.button != 1 or ev.xdata is None or ev.ydata is None:
            return
        if istate.stage is InteractionStage.IDLE:
            # start new Gaussian
            istate.stage = InteractionStage.DRAW_PRIMARY
            istate.centre_px = (int(ev.xdata), int(ev.ydata))
        elif istate.stage is InteractionStage.ADJUST_MINOR:
            # second click → finalise
            istate.stage = InteractionStage.IDLE
            full_render()

    def on_motion(ev):
        nonlocal istate
        if ev.xdata is None or ev.ydata is None:
            return
        x, y = int(ev.xdata), int(ev.ydata)
        if istate.stage is InteractionStage.DRAW_PRIMARY and istate.centre_px is not None:
            cx, cy = istate.centre_px
            dx, dy = x - cx, y - cy
            if dx == 0 and dy == 0:
                return
            istate.θ_rad = math.atan2(dy, dx)
            dist_pix = math.hypot(dx, dy)
            istate.σ_major_norm = dist_pix / (min(W, H) / 2.0)
            istate.σ_minor_norm = istate.σ_major_norm / 3.0  # fixed AR 3:1
            render_gaussian_only()
        elif istate.stage is InteractionStage.ADJUST_MINOR and istate.release_px is not None:
            rx, _ = istate.release_px
            Δx = abs(x - rx)
            istate.σ_minor_norm = max(1e-3, Δx / (min(W, H) / 2.0))
            render_gaussian_only()

    def on_release(ev):
        nonlocal istate
        if ev.button != 1 or istate.stage is not InteractionStage.DRAW_PRIMARY:
            return
        istate.stage = InteractionStage.ADJUST_MINOR
        istate.release_px = (int(ev.xdata), int(ev.ydata))
        render_gaussian_only()

    def on_key(ev):
        if ev.key == 'q':
            plt.close(fig)
            sys.exit(0)

    cid_press = fig.canvas.mpl_connect('button_press_event', on_press)
    cid_release = fig.canvas.mpl_connect('button_release_event', on_release)
    cid_move = fig.canvas.mpl_connect('motion_notify_event', on_motion)
    cid_key = fig.canvas.mpl_connect('key_press_event', on_key)

    # instructions banner -----------------------------------
    instr = (
        "Mouse gesture:\n"
        " 1. LMB‑drag → centre + θ + σ_major (AR 3:1)\n"
        " 2. release → move horizontally to set σ_minor\n"
        " 3. second LMB click → run Flip and update.\n"
        "Key: q = quit")
    fig.text(0.5, 0.02, instr, ha='center', va='bottom', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout(rect=[0,0.05,1,0.96])
    plt.show()


if __name__ == "__main__":
    main()


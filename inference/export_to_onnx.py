#!/usr/bin/env python
"""
Export a trained FlipEncoder checkpoint to two ONNX models for KV caching:
1. flip-encoder: prepare_kv_cache -> returns k_cached, v_cached
2. flip-predictor: forward_with_cache -> returns mask_logits

Example
-------
python export_to_onnx_cached.py \
       --config cfgs/flip.json \
       --checkpoint logs/flip_model.ckpt \
       --encoder-output flip-encoder.onnx \
       --predictor-output flip-predictor.onnx \
       --opset 17
"""
import argparse
from pathlib import Path
import torch
import numpy as np
from utils.configuration import Configuration
from nn.encoder_sparse import FlipEncoder


class FlipEncoderCache(torch.nn.Module):
    """Wrapper for the encoder part that only runs prepare_kv_cache"""
    def __init__(self, flip_encoder):
        super().__init__()
        self.flip_encoder = flip_encoder
    
    def forward(self, 
                patches_p1, patches_p2, patches_p4, patches_p8, patches_p16,
                coords_p1, coords_p2, coords_p4, coords_p8, coords_p16,
                position):
        k_cached, v_cached = self.flip_encoder.prepare_kv_cache(
            patches_p1, patches_p2, patches_p4, patches_p8, patches_p16,
            coords_p1, coords_p2, coords_p4, coords_p8, coords_p16,
            position
        )
        return k_cached, v_cached


class FlipPredictorCache(torch.nn.Module):
    """Wrapper for the predictor part that only runs forward_with_cache"""
    def __init__(self, flip_encoder):
        super().__init__()
        self.flip_encoder = flip_encoder
    
    def forward(self, position, mask_coordinates, k_cached, v_cached):
        mask_logits = self.flip_encoder.forward_with_cache(
            position, mask_coordinates, k_cached, v_cached
        )
        return mask_logits


def build_model(cfg, device):
    """Instantiate FlipEncoder exactly as in evaluation."""
    model = FlipEncoder(
        layers         = cfg.model.encoder.layers,
        input_channels = 3,                             # RGB
        channels       = cfg.model.encoder.channels,
        num_registers  = cfg.model.encoder.num_registers,
        head_size      = cfg.model.encoder.head_size,
    ).to(device)
    return model


def load_checkpoint(model, ckpt_path, device):
    """Load state-dict (stripping `net.encoder.` prefix)."""
    sd = torch.load(ckpt_path, map_location=device)["state_dict"]
    sd = {k.replace("net.encoder.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[WARN] Missing keys {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys {unexpected}")
    model.eval()
    return model


def make_dummy_inputs_encoder(device, dtype=torch.float32):
    """
    Build dummy inputs for the encoder (prepare_kv_cache).
    """
    dummy = {
        # image‐patch tokens
        "patches_p1"  : torch.zeros(4,  3,  1,  1,  dtype=dtype, device=device),
        "patches_p2"  : torch.zeros(4,  3,  2,  2,  dtype=dtype, device=device),
        "patches_p4"  : torch.zeros(4,  3,  4,  4,  dtype=dtype, device=device),
        "patches_p8"  : torch.zeros(4,  3,  8,  8,  dtype=dtype, device=device),
        "patches_p16" : torch.zeros(4,  3, 16, 16, dtype=dtype, device=device),

        # patch-center coordinates
        "coords_p1"  : torch.zeros(4, 2, dtype=dtype, device=device),
        "coords_p2"  : torch.zeros(4, 2, dtype=dtype, device=device),
        "coords_p4"  : torch.zeros(4, 2, dtype=dtype, device=device),
        "coords_p8"  : torch.zeros(4, 2, dtype=dtype, device=device),
        "coords_p16" : torch.zeros(4, 2, dtype=dtype, device=device),

        # global Gaussian pose   (µx, µy, σx, σy, ρ₁, ρ₂)
        "position" : torch.zeros(1, 6, dtype=dtype, device=device),
    }
    return dummy


def make_dummy_inputs_predictor(device, dtype=torch.float32, 
                               num_tokens=20, num_heads=8, head_dim=64):
    """
    Build dummy inputs for the predictor (forward_with_cache).
    
    Note: You may need to adjust num_tokens, num_heads, head_dim based on your 
    actual model configuration and the shapes returned by prepare_kv_cache.
    """
    dummy = {
        "position": torch.zeros(1, 6, dtype=dtype, device=device),
        "mask_coordinates": torch.zeros(10, 2, dtype=dtype, device=device),
        "k_cached": torch.zeros(1, num_heads, num_tokens, head_dim, dtype=dtype, device=device),
        "v_cached": torch.zeros(1, num_heads, num_tokens, head_dim, dtype=dtype, device=device),
    }
    return dummy


def export_encoder(model, device, output_path, opset):
    """Export the encoder part (prepare_kv_cache)"""
    encoder_model = FlipEncoderCache(model)
    encoder_model.eval()
    
    dummy_inputs = make_dummy_inputs_encoder(device)
    input_list = list(dummy_inputs.values())
    input_names = list(dummy_inputs.keys())
    output_names = ["k_cached", "v_cached"]
    
    # Mark dynamic axes for encoder inputs
    dyn = {name: {0: f"n_{name}"} for name in input_names}
    dyn["k_cached"] = {2: "n_tokens"}  # num_tokens dimension
    dyn["v_cached"] = {2: "n_tokens"}  # num_tokens dimension
    
    torch.onnx.export(
        encoder_model, tuple(input_list), output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dyn,
        opset_version=opset,
        do_constant_folding=True,
        verbose=False,
    )
    
    print(f"[OK]   exported encoder ONNX model → {Path(output_path).resolve()}")
    print(f"       input names : {input_names}")
    print(f"       output names: {output_names}")


def export_predictor(model, device, output_path, opset):
    """Export the predictor part (forward_with_cache)"""
    predictor_model = FlipPredictorCache(model)
    predictor_model.eval()
    
    # First, we need to determine the actual cache dimensions by running prepare_kv_cache
    dummy_encoder_inputs = make_dummy_inputs_encoder(device)
    with torch.no_grad():
        k_cached, v_cached = model.prepare_kv_cache(
            dummy_encoder_inputs["patches_p1"], 
            dummy_encoder_inputs["patches_p2"],
            dummy_encoder_inputs["patches_p4"],
            dummy_encoder_inputs["patches_p8"],
            dummy_encoder_inputs["patches_p16"],
            dummy_encoder_inputs["coords_p1"],
            dummy_encoder_inputs["coords_p2"],
            dummy_encoder_inputs["coords_p4"],
            dummy_encoder_inputs["coords_p8"],
            dummy_encoder_inputs["coords_p16"],
            dummy_encoder_inputs["position"]
        )
    
    # Now create dummy inputs for predictor with correct cache dimensions
    dummy_predictor = {
        "position": torch.zeros(1, 6, dtype=torch.float32, device=device),
        "mask_coordinates": torch.zeros(10, 2, dtype=torch.float32, device=device),
        "k_cached": torch.zeros_like(k_cached),
        "v_cached": torch.zeros_like(v_cached),
    }
    
    input_list = list(dummy_predictor.values())
    input_names = list(dummy_predictor.keys())
    output_names = ["mask_logits"]
    
    # Mark dynamic axes for predictor
    dyn = {
        "mask_coordinates": {0: "n_mask"},
        "k_cached": {2: "n_tokens"},
        "v_cached": {2: "n_tokens"},
        "mask_logits": {0: "n_mask"}
    }
    
    torch.onnx.export(
        predictor_model, tuple(input_list), output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dyn,
        opset_version=opset,
        do_constant_folding=True,
        verbose=False,
    )
    
    print(f"[OK]   exported predictor ONNX model → {Path(output_path).resolve()}")
    print(f"       input names : {input_names}")
    print(f"       output names: {output_names}")
    print(f"       cache shapes: k_cached={k_cached.shape}, v_cached={v_cached.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("-k", "--checkpoint", required=True)
    ap.add_argument("--encoder-output", required=True,
                    help="Output path for encoder ONNX model")
    ap.add_argument("--predictor-output", required=True,
                    help="Output path for predictor ONNX model")
    ap.add_argument("--opset", type=int, default=17,
                    help="ONNX opset to export with (>=16 recommended).")
    ap.add_argument("--device", default="cpu",
                    choices=["cpu", "cuda"], help="Export device.")
    args = ap.parse_args()

    cfg = Configuration(args.config)
    device = torch.device(args.device)

    # ---------------------------------------------------------------- build --
    model = build_model(cfg, device)
    model = load_checkpoint(model, args.checkpoint, device)

    # ----------------------------------------------------------- export both --
    print("Exporting encoder model...")
    export_encoder(model, device, args.encoder_output, args.opset)
    
    print("\nExporting predictor model...")
    export_predictor(model, device, args.predictor_output, args.opset)
    
    print(f"\n[DONE] Both models exported with opset {args.opset}")


if __name__ == "__main__":
    main()

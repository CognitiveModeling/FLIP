#!/usr/bin/env python

import torch
import numpy as np
import cv2
import h5py
from PIL import Image
import pandas as pd
import argparse
import os
from einops import rearrange
import time
import math
from utils.configuration import Configuration
from model.lightning.flip import FlipModule
from ext.position_wrapper import compute_position_rot_from_rho, sample_continuous_patches
from turbojpeg import decompress as jpeg_decompress
import math
from tqdm import tqdm

def solve_n(x, y, a):
    # Ensure x and a are positive and a > 1 for this derivation
    if x <= 0 or a <= 1:
        raise ValueError("Require x > 0 and a > 1")
        
    ratio = y / x
    n_float = math.log(ratio, a)
    n = math.floor(n_float)
    return n


class Timer:
    def __init__(self):
        self.last    = time.time()
        self.passed  = 0
        self.sum     = 0
        self.elapsed = 0

    def to_str(self, remaining=None):
        time_diff     = time.time() - self.last
        self.elapsed += time_diff
        self.passed   = self.passed * 0.99 + time_diff
        self.sum      = self.sum * 0.99 + 1
        passed        = self.passed / self.sum
        self.last     = time.time()

        _str = f"{passed:.2f}s/it" if passed > 1 else f"{1.0/passed:.2f}it/s"

        if remaining == 0:
            hh, mm, ss = self._seconds_to_hms(self.elapsed)
            return f"{_str} ({hh:02d}:{mm:02d}:{ss:02d})"
        elif remaining is not None:
            est_time = self.estimate(remaining)
            hh, mm, ss = self._seconds_to_hms(est_time)
            return f"{_str} ({hh:02d}:{mm:02d}:{ss:02d})"

        return _str

    def estimate(self, remaining):
        """Estimate the remaining time."""
        passed = self.passed / self.sum
        return passed * remaining

    @staticmethod
    def _seconds_to_hms(seconds):
        """Convert seconds to hours, minutes, and seconds."""
        hh = int(seconds // 3600)
        mm = int((seconds % 3600) // 60)
        ss = int(seconds % 60)
        return hh, mm, ss

class TimeKeeper:
    def __init__(self, cuda=False):
        """
        Initialize the TimeKeeper.

        Parameters:
        - cuda (bool): If True, synchronizes CUDA before timing.
        """
        self.cuda = cuda
        self.start_times = []
        self.end_times = []
        self.elapsed_times = []

    def _synchronize_cuda(self):
        """Synchronize CUDA if cuda option is enabled."""
        if self.cuda:
            torch.cuda.synchronize()

    def start(self):
        """Start measuring time."""
        self._synchronize_cuda()
        self.start_times.append(time.perf_counter())

    def stop(self):
        """Stop measuring time and record the elapsed time."""
        self._synchronize_cuda()
        end_time = time.perf_counter()
        self.end_times.append(end_time)

        assert len(self.start_times) == len(self.end_times), f"start and stop calls do not match: {len(self.start_times)} != {len(self.end_times)}"

        elapsed = 0
        for start, end in zip(self.start_times, self.end_times):
            elapsed += end - start
        self.elapsed_times.append(elapsed)

        self.start_times.clear()
        self.end_times.clear()

    def average_time(self):
        """
        Calculate the average elapsed time of all measurements.

        Returns:
        - float: Average time in seconds.
        """
        if not self.elapsed_times:
            return 0.0
        return sum(self.elapsed_times) / len(self.elapsed_times)

    def min_time(self):
        """
        Get the minimum elapsed time.

        Returns:
        - float: Minimum time in seconds.
        """
        if not self.elapsed_times:
            return 0.0
        return min(self.elapsed_times)

    def max_time(self):
        """
        Get the maximum elapsed time.

        Returns:
        - float: Maximum time in seconds.
        """
        if not self.elapsed_times:
            return 0.0
        return max(self.elapsed_times)

    def last_time(self):
        """
        Get the last elapsed time.

        Returns:
        - float: Last time in seconds.
        """
        if not self.elapsed_times:
            return 0.0
        return self.elapsed_times[-1]

    def clear(self):
        """Clear all recorded times."""
        self.start_times.clear()
        self.end_times.clear()
        self.elapsed_times.clear()

    def __str__(self):
        return f"TimeKeeper(cuda={self.cuda}, measurements={len(self.elapsed_times)})"

# Function to calculate IoU
def calculate_iou(pred_mask, true_mask):
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    if union == 0:
        return 0.0
    return intersection / union

def predict_mask_full(model, input_rgb, position, device, preprocessing_timer, model_timer, max_overlap_threshold=None, coverage=None):
    """
    Predict a mask for the entire image using the model.
    
    Args:
        model: Trained Flip model
        input_rgb: Input RGB image
        position: Position tensor
        device: Torch device
        preprocessing_timer: Timer for preprocessing
        model_timer: Timer for model execution
        max_overlap_threshold: Maximum overlap allowed between patches
        coverage: Coverage factor for patch sampling
        
    Returns:
        Predicted mask logits as a pytorch tensor with shape (H, W)
    """
    # Get image dimensions
    H, W = input_rgb.shape[:2]
    
    preprocessing_timer.start()
    
    # Define patch sizes
    min_patch_size = int(math.log2(model.cfg.model.min_patch_size))
    max_patch_size = int(math.log2(model.cfg.model.max_patch_size))
    patch_sizes = (2**torch.linspace(min_patch_size, max_patch_size, 
                               max_patch_size - min_patch_size + 1)).long().tolist()
    
    # Sample number of tokens
    num_tokens = model.cfg.model.avg_num_tokens
    
    # Sample patches and coordinates using the position_wrapper function
    input_patches, input_coordinates, target_indices, seq_lengths = sample_continuous_patches(
        input_rgb, position, num_tokens, patch_sizes, 
        max_overlap_threshold=max_overlap_threshold, 
        coverage=coverage
    )
    
    # Generate grid coordinates for all pixels
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device) * (H/256),
        torch.linspace(-1, 1, W, device=device) * (W/256),
        indexing='ij'
    )
    mask_coordinates = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
    mask_seq_lengths = torch.tensor([H*W], dtype=torch.long, device=device)
    
    # Convert input_patches from RGB to float
    input_patches = [p.to(device).float() / 255 for p in input_patches]
    
    # move to device
    input_coordinates = [c.to(device) for c in input_coordinates]
    target_indices = [t.to(device) for t in target_indices]
    seq_lengths = [s.to(device) for s in seq_lengths]
    
    # Prepare position tensor
    position_tensor = position.unsqueeze(0).to(device)
    position_tensor[0,0] *= W/256
    position_tensor[0,1] *= H/256
    position_tensor[0,2] *= W/256
    position_tensor[0,3] *= H/256
    
    preprocessing_timer.stop()
    model_timer.start()
    
    # Predict mask
    mask_predictions = model.net.encoder(
        input_patches=input_patches,
        position=position_tensor,
        coordinates=input_coordinates,
        target_indices=target_indices,
        seq_lengths=seq_lengths,
        mask_coordinates=mask_coordinates,
        mask_seq_lengths=mask_seq_lengths
    )
    
    # Convert predictions to logits
    mask_logits = mask_predictions.reshape(H, W)
    
    model_timer.stop()
    
    return mask_logits, sum([len(p) for p in input_patches])

def predict_mask_optimized(model, input_rgb, position, device, preprocessing_timer, model_timer, max_overlap_threshold=None, coverage=None, batch_size=-1):
    """
    Predict a mask for the image using the model, but only within a 5-sigma region.
    
    Args:
        model: Trained Flip model
        input_rgb: Input RGB image 
        position: Position tensor
        device: Torch device
        preprocessing_timer: Timer for preprocessing
        model_timer: Timer for model execution
        max_overlap_threshold: Maximum overlap allowed between patches
        coverage: Coverage factor for patch sampling
        
    Returns:
        Predicted mask logits as a pytorch tensor with shape (H, W)
    """
    H, W = input_rgb.shape[:2]
    
    preprocessing_timer.start()
    
    # Define patch sizes
    min_patch_size = int(math.log2(model.cfg.model.min_patch_size))
    max_patch_size = int(math.log2(model.cfg.model.max_patch_size))
    patch_sizes = (2**torch.linspace(min_patch_size, max_patch_size, 
                               max_patch_size - min_patch_size + 1)).long().tolist()
    
    # Sample number of tokens
    num_tokens = model.cfg.model.avg_num_tokens
    
    # Sample patches and coordinates using the position_wrapper function
    input_patches, input_coordinates, target_indices, seq_lengths = sample_continuous_patches(
        input_rgb, position, num_tokens, patch_sizes,
        max_overlap_threshold=max_overlap_threshold,
        coverage=coverage
    )
    
    # Extract Gaussian parameters from position
    mu_x, mu_y, sigma_x, sigma_y = position[:4].tolist()
    
    # Create a mask of zeros for the full image
    mask_image = torch.zeros((H, W), dtype=torch.float32, device=device)
    
    # Convert normalized coordinates to image coordinates (doing this on GPU)
    mu_x_img = (mu_x + 1) * W / 2
    mu_y_img = (mu_y + 1) * H / 2
    sigma_x_img = sigma_x * W / 2
    sigma_y_img = sigma_y * H / 2
    
    # Compute bounding box using 5-sigma rule (minimal CPU operations)
    sigma_iso = max(sigma_x_img, sigma_y_img)
    x_min = max(0, int(mu_x_img - 5 * sigma_iso))
    y_min = max(0, int(mu_y_img - 5 * sigma_iso))
    x_max = min(W - 1, int(mu_x_img + 5 * sigma_iso))
    y_max = min(H - 1, int(mu_y_img + 5 * sigma_iso))
    
    # Check if bounding box is empty
    if x_min >= x_max or y_min >= y_max:
        preprocessing_timer.stop()
        return mask_image, 0
    
    # Calculate box width and height
    box_width = x_max - x_min + 1
    box_height = y_max - y_min + 1
    
    # Generate pixel coordinates within the bounding box directly on GPU
    y_range = torch.linspace(y_min, y_max, box_height, device=device)
    x_range = torch.linspace(x_min, x_max, box_width, device=device)
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')
    
    # Flatten coordinates
    y_flat = grid_y.flatten()
    x_flat = grid_x.flatten()
    
    # Convert to normalized coordinates
    norm_x = x_flat / (W/2) - 1  # Normalize to range [-1, 1]
    norm_y = y_flat / (H/2) - 1  # Normalize to range [-1, 1]
    
    # Scale by image dimensions ratio as in the original code
    norm_x = norm_x * (W/256)
    norm_y = norm_y * (H/256)
    
    # Stack coordinates
    mask_coordinates = torch.stack([norm_x, norm_y], dim=-1)
    mask_seq_lengths = torch.tensor([len(mask_coordinates)], dtype=torch.long, device=device)
    
    # Convert input_patches from RGB to float
    input_patches = [p.to(device).float() / 255 for p in input_patches]
    
    # move to device
    input_coordinates = [c.to(device) for c in input_coordinates]
    target_indices = [t.to(device) for t in target_indices]
    seq_lengths = [s.to(device) for s in seq_lengths]
    
    # Prepare position tensor
    position_tensor = position.to(device).unsqueeze(0)
    position_tensor[0,0] *= W/256
    position_tensor[0,1] *= H/256
    position_tensor[0,2] *= W/256
    position_tensor[0,3] *= H/256
    
    preprocessing_timer.stop()
    model_timer.start()
    
    # Predict mask for the bounding box coordinates
    if batch_size == -1 or len(mask_coordinates) <= batch_size:
        mask_predictions = model.net.encoder(
            input_patches=input_patches,
            position=position_tensor,
            coordinates=input_coordinates,
            target_indices=target_indices,
            seq_lengths=seq_lengths,
            mask_coordinates=mask_coordinates,
            mask_seq_lengths=mask_seq_lengths
        )
    else:
        # Split input_patches into batches
        mask_predictions = []
        for i in tqdm(range(0, len(mask_coordinates), batch_size), 
                      desc="Predicting mask in batches", unit="batch"):
            batch_coordinates = mask_coordinates[i:i + batch_size]
            batch_seq_lengths = torch.tensor([len(batch_coordinates)], dtype=torch.long, device=device)

            batch_predictions = model.net.encoder(
                input_patches=input_patches,
                position=position_tensor,
                coordinates=input_coordinates,
                target_indices=target_indices,
                seq_lengths=seq_lengths,
                mask_coordinates=batch_coordinates,
                mask_seq_lengths=batch_seq_lengths
            )

            mask_predictions.append(batch_predictions)
        mask_predictions = torch.cat(mask_predictions, dim=0)
    
    # Place the predictions in the correct locations in the full mask (using tensor indexing)
    mask_image[y_flat.long(), x_flat.long()] = mask_predictions
    
    model_timer.stop()
    
    return mask_image, sum([len(p) for p in input_patches])

def predict_mask_hirachical(model, input_rgb, position, device, num_tokens, scale_factor, preprocessing_timer, model_timer, postprocessing_timer, max_overlap_threshold=None, coverage=None):
    """
    Predict a mask for the image using the model, but only within a 5-sigma region.
    
    Args:
        model: Trained Flip model
        input_rgb: Input RGB image 
        position: Position tensor
        device: Torch device
        num_tokens: Number of mask tokens to use
        scale_factor: Scale factor for hierarchical interpolation
        preprocessing_timer: Timer for preprocessing
        model_timer: Timer for model execution
        postprocessing_timer: Timer for postprocessing
        max_overlap_threshold: Maximum overlap allowed between patches
        coverage: Coverage factor for patch sampling
        
    Returns:
        Predicted mask logits as a pytorch tensor with shape (H, W)
    """
    H, W = input_rgb.shape[:2]
    
    preprocessing_timer.start()
    
    # Define patch sizes
    min_patch_size = int(math.log2(model.cfg.model.min_patch_size))
    max_patch_size = int(math.log2(model.cfg.model.max_patch_size))
    patch_sizes = (2**torch.linspace(min_patch_size, max_patch_size, 
                               max_patch_size - min_patch_size + 1)).long().tolist()
    
    # Sample number of tokens
    num_img_tokens = model.cfg.model.avg_num_tokens
    
    # Sample patches and coordinates using the position_wrapper function
    input_patches, input_coordinates, target_indices, seq_lengths = sample_continuous_patches(
        input_rgb, position, num_img_tokens, patch_sizes,
        max_overlap_threshold=max_overlap_threshold,
        coverage=coverage
    )
    
    # Extract Gaussian parameters from position
    mu_x, mu_y, sigma_x, sigma_y = position[:4].tolist()
    
    # Convert normalized coordinates to image coordinates (doing this on GPU)
    mu_x_img = (mu_x + 1) * W / 2
    mu_y_img = (mu_y + 1) * H / 2
    sigma_x_img = sigma_x * W / 2
    sigma_y_img = sigma_y * H / 2
    
    # Compute bounding box using 5-sigma rule (minimal CPU operations)
    sigma_iso = max(sigma_x_img, sigma_y_img)
    x_min = max(0, int(mu_x_img - 5 * sigma_iso))
    y_min = max(0, int(mu_y_img - 5 * sigma_iso))
    x_max = min(W - 1, int(mu_x_img + 5 * sigma_iso))
    y_max = min(H - 1, int(mu_y_img + 5 * sigma_iso))

    bbox_height = y_max - y_min
    bbox_width = x_max - x_min
    
    # Check if bounding box is empty
    if x_min >= x_max or y_min >= y_max:
        preprocessing_timer.stop()
        return np.zeros((H, W), dtype=bool), 0

    init_width  = int(num_tokens**0.5)
    init_height = int(num_tokens**0.5)
    # preserve aspect ratio
    if bbox_width > bbox_height:
        init_width = int(init_width * bbox_width / bbox_height)
    elif bbox_height > bbox_width:
        init_height = int(init_height * bbox_height / bbox_width)

    if init_width * (scale_factor**0.5) > bbox_width or init_height * (scale_factor**0.5) > bbox_height:
        init_width = bbox_width
        init_height = bbox_height

    init_size = max(init_width, init_height)
    bbox_size = max(bbox_width, bbox_height)

    # find the number of interpolations
    n = solve_n(init_size, bbox_size, scale_factor)
    if n > 0:
        scale_factor = (bbox_size / init_size)**(1/n)

    sizes = [(init_height, init_width)]
    for i in range(1, n):
        sizes.append((int(init_height * scale_factor**i), int(init_width * scale_factor**i)))
    if sizes[-1][0] < bbox_height or sizes[-1][1] < bbox_width:
        sizes.append((bbox_height, bbox_width))

    # normalize bbox
    bbox = (x_min/128 - W/256, y_min/128 - H/256, x_max/128 - W/256, y_max/128 - H/256)
    
    # Convert input_patches from RGB to float
    input_patches = [p.to(device).float() / 255 for p in input_patches]
    
    # move to device
    input_coordinates = [c.to(device) for c in input_coordinates]
    target_indices = [t.to(device) for t in target_indices]
    seq_lengths = [s.to(device) for s in seq_lengths]
    
    # Prepare position tensor
    position_tensor = position.to(device).unsqueeze(0)
    position_tensor[0,0] *= W/256
    position_tensor[0,1] *= H/256
    position_tensor[0,2] *= W/256
    position_tensor[0,3] *= H/256
    
    preprocessing_timer.stop()
    model_timer.start()
    
    # Predict mask for the bounding box coordinates
    mask_predictions = model.net.encoder.inference(
        input_patches=input_patches,
        position=position_tensor,
        coordinates=input_coordinates,
        target_indices=target_indices,
        seq_lengths=seq_lengths,
        mask_bbox=bbox,
        mask_sizes=sizes
    )

    model_timer.stop()
    postprocessing_timer.start()
    
    # Place the predictions in the correct locations in the full mask (using tensor indexing)
    mask_image = np.zeros((H, W), dtype=bool)
    mask_image[y_min:y_max,x_min:x_max] = (mask_predictions > 0.5).view(bbox_height, bbox_width).cpu().numpy()
        
    postprocessing_timer.stop()
    
    return mask_image, sum([len(p) for p in input_patches])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", help="path to the dataset", required=True)
    parser.add_argument("--model_path", help="path to the model", required=True)
    parser.add_argument("--config", help="path to the config file", required=True)
    parser.add_argument("--seed", help="seed for the random number generator", default=42, type=int)
    parser.add_argument("--num-tokens", help="number of tokens", default=-1, type=int)
    parser.add_argument("--output_dir", help="output directory for results", default=None)
    parser.add_argument("--save_masks", help="save predicted masks", action="store_true")
    parser.add_argument("--optimized", help="use optimized mask prediction (5-sigma region)", action="store_true")
    parser.add_argument("--hirachical", help="use hirachical mask prediction", action="store_true")
    parser.add_argument("--batch-size", help="batch size for mask prediction", default=-1, type=int)
    parser.add_argument("--scale-factor", help="scale factor for hirachical mask prediction", default=3.0, type=float)
    parser.add_argument("--num-mask-tokens", help="number of mask tokens", default=2048, type=int)
    parser.add_argument("--max-overlap-threshold", help="maximum overlap threshold for patch sampling", type=float, default=2.67)
    parser.add_argument("--coverage", help="coverage factor for patch sampling", type=float, default=1.44)
    
    args = parser.parse_args()
    
    # Load configuration
    cfg = Configuration(args.config)
    cfg.seed = args.seed
    cfg.model.batch_size = 1
    
    # Set number of tokens if specified
    if args.num_tokens != -1:
        cfg.model.avg_num_tokens = args.num_tokens
        cfg.model.max_num_tokens = args.num_tokens
        cfg.model.min_num_tokens = args.num_tokens
    
    # Set random seed
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = FlipModule.load_from_checkpoint(args.model_path, cfg=cfg, strict=False, map_location=device)
    model.to(device)
    model.eval()
    
    # Load the HDF5 dataset
    hdf5_file = h5py.File(args.dataset_path, "r")

    instance_masks_images  = hdf5_file["instance_masks_images"]
    relative_mask_sizes = hdf5_file['relative_mask_sizes'] if 'relative_mask_sizes' in hdf5_file else None
    dataset_length = len(instance_masks_images)
    
    # Initialize timing and metrics
    iou_scores = []
    relative_mask_sizes = []
    mask_sizes = []
    img_widths = []
    img_heights = []
    img_sizes = []
    data_load_times = []
    preprocessing_times = []
    postprocessing_times = []
    model_times = []
    input_tokens = []
    
    # Create timers
    data_load_timer = TimeKeeper()
    preprocessing_timer = TimeKeeper(cuda=True)
    postprocessing_timer = TimeKeeper(cuda=True)
    model_timer = TimeKeeper(cuda=True)
    update_timer = Timer()
    
    # Log patch sampling parameters
    if args.max_overlap_threshold is not None:
        print(f"Using max_overlap_threshold: {args.max_overlap_threshold}")
    if args.coverage is not None:
        print(f"Using coverage: {args.coverage}")
    
    # Iterate through the dataset
    for i in range(dataset_length):
        with torch.no_grad():
            data_load_timer.start()

            image_index = instance_masks_images[i].item()
            
            # Get data sample
            input_rgb   = np.array(jpeg_decompress(hdf5_file["rgb_images"][image_index]))
            gt_mask     = hdf5_file["instance_masks"][i]
            gt_position = compute_position_rot_from_rho(torch.from_numpy(hdf5_file["positions"][i]))

            if gt_mask.dtype == np.uint8 and len(gt_mask.shape) == 1:
                gt_mask = cv2.imdecode(gt_mask, cv2.IMREAD_UNCHANGED)

            if gt_mask.max() > 1:
                gt_mask = (gt_mask > 128).squeeze().astype(np.uint8)
            else:
                gt_mask = (gt_mask > 0).squeeze().astype(np.uint8)

            data_load_timer.stop()

            # Get the coordinates of all pixels inside the true_mask
            mask_indices = np.argwhere(gt_mask > 0)
            if len(mask_indices) == 0:
                continue  # Skip if mask is empty
            
            # Get image dimensions
            height, width = input_rgb.shape[:2]
            H, W = height, width
            
            # Choose prediction function based on flag
            if args.hirachical:
                pred_binary, num_input_tokens = predict_mask_hirachical(
                    model, input_rgb, gt_position, device, 
                    args.num_mask_tokens, args.scale_factor, 
                    preprocessing_timer, model_timer, postprocessing_timer,
                    max_overlap_threshold=args.max_overlap_threshold,
                    coverage=args.coverage
                )
            else:
                if args.optimized:
                    mask_logits, num_input_tokens = predict_mask_optimized(
                        model, input_rgb, gt_position, device, 
                        preprocessing_timer, model_timer,
                        max_overlap_threshold=args.max_overlap_threshold,
                        coverage=args.coverage,
                        batch_size=args.batch_size
                    )
                else:
                    mask_logits, num_input_tokens = predict_mask_full(
                        model, input_rgb, gt_position, device, 
                        preprocessing_timer, model_timer,
                        max_overlap_threshold=args.max_overlap_threshold,
                        coverage=args.coverage
                    )

                postprocessing_timer.start()
                pred_binary = (mask_logits > 0.0).cpu().numpy().squeeze()
                postprocessing_timer.stop()
                
            
            # Record timing information
            data_load_times.append(data_load_timer.last_time())
            preprocessing_times.append(preprocessing_timer.last_time())
            postprocessing_times.append(postprocessing_timer.last_time())
            model_times.append(model_timer.last_time())
            
            # Calculate IoU
            true_mask = gt_mask.astype(bool)
            iou = calculate_iou(pred_binary, true_mask)
            
            # Save metrics
            iou_scores.append(iou)
            relative_mask_sizes.append(np.sum(true_mask) / max(height * width, 1))
            mask_sizes.append(np.sum(true_mask))
            img_widths.append(width)
            img_heights.append(height)
            img_sizes.append(height * width)
            input_tokens.append(num_input_tokens)

            # Print progress information
            print(f"Eval[{i+1}/{dataset_length}]: {update_timer.to_str(dataset_length-i-1)} "
                  f"IoU: {iou*100:.1f}, "
                  f"{data_load_timer.average_time() * 1000:.1f}ms (data load), "
                  f"{preprocessing_timer.average_time()*1000:.1f}ms (preprocessing), "
                  f"{model_timer.average_time()*1000:.1f}ms (model), "
                  f"{postprocessing_timer.average_time()*1000:.1f}ms (postprocessing), "
                  f"Size: {height}x{width} ({height*width} pixels)", flush=True)
            
            
            # Optionally save masks for visual inspection
            if args.save_masks:
                save_dir = os.path.join(args.output_dir, "masks") if args.output_dir else "masks"
                os.makedirs(save_dir, exist_ok=True)
                
                Image.fromarray((true_mask * 255).astype(np.uint8)).save(f"{save_dir}/mask_true_{i:03d}.png")
                Image.fromarray((pred_binary * 255).astype(np.uint8)).save(f"{save_dir}/mask_pred_{i:03d}.png")
                
                # Create overlay image
                img = cv2.cvtColor(input_rgb, cv2.COLOR_RGB2GRAY)
                img = np.stack([img, true_mask * 255, pred_binary * 255], axis=-1).astype(np.uint8)
                Image.fromarray(img).save(f"{save_dir}/img_overlay_{i:03d}.jpg")
    
    # Calculate average IoU
    avg_iou = np.mean(iou_scores)
    print(f"Average IoU over the dataset: {avg_iou:.4f}")
    
    # Save results to CSV
    df = pd.DataFrame({
        'iou': iou_scores,
        'relative_mask_size': relative_mask_sizes,
        'mask_size': mask_sizes,
        'img_width': img_widths,
        'img_height': img_heights,
        'img_size': img_sizes,
        'data_load_time': data_load_times,
        'preprocessing_time': preprocessing_times,
        'postprocessing_time': postprocessing_times,
        'model_time': model_times,
        'input_tokens': input_tokens,
    })
    
    # Use the name of the dataset to save the results
    csv_path = os.path.basename(args.dataset_path).replace('.h5', '.csv').replace('.hdf5', '.csv')
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        csv_path = os.path.join(args.output_dir, csv_path)
    
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

if __name__ == "__main__":
    main()

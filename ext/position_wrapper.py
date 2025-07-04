import numpy as np
import torch
import flip_position  # This is our compiled C extension
import flip_gaussian

def compute_position_rot_from_rho(position_rho):
    """
    Compute position rotation from rho representation.
    
    Args:
        position_rho (torch.Tensor): Tensor with shape (..., 5)
        
    Returns:
        torch.Tensor with shape (..., 6)
    """
    # Convert torch tensor to numpy array
    original_shape = position_rho.shape
    position_rho_np = position_rho.detach().cpu().numpy().reshape(-1, 5)
    
    # Call C extension
    result_np = flip_position.compute_position_rot_from_rho(position_rho_np)
    
    # Convert back to torch tensor
    result = torch.from_numpy(result_np).to(position_rho.device)
    return result.reshape(original_shape[:-1] + (6,))

def apply_position_augmentation(gt_position):
    """
    Apply conservative tracking to a position.
    
    Args:
        gt_position (torch.Tensor): Tensor with shape (6,)
        
    Returns:
        torch.Tensor with shape (6,)
    """
    # Convert torch tensor to numpy array
    gt_position_np = gt_position.detach().cpu().numpy()
    
    # Call C extension
    result_np = flip_position.apply_position_augmentation(gt_position_np)
    
    # Convert back to torch tensor
    return torch.from_numpy(result_np).to(gt_position.device)

def sample_num_tokens(desired_min, desired_max, desired_mean):
    """
    Sample the number of tokens from a beta distribution.
    
    Args:
        desired_min (int): Minimum number of tokens
        desired_max (int): Maximum number of tokens
        desired_mean (float): Mean number of tokens
        alpha (float): Alpha parameter for beta distribution
        
    Returns:
        int: Sampled number of tokens
    """
    return flip_position.sample_num_tokens(desired_min, desired_max, desired_mean)

def sample_continuous_patches(image, position_tensor, num_tokens, patch_sizes, max_overlap_threshold=None, coverage=None):
    """
    Sample continuous patches from an image using Gaussian distribution parameters.
    
    Args:
        image: RGB image (H, W, C) as numpy array
        position_tensor: Position tensor with Gaussian parameters
        num_tokens: Number of patches to sample
        patch_sizes: List of patch sizes to use
        
    Returns:
        Tuple of (input_patches, input_coordinates, target_indices, seq_lengths)
    """
    # Convert torch tensor to numpy array
    position_np = position_tensor.detach().cpu().numpy()
    
    # Convert patch sizes to a list of floats
    patch_sizes_list = [float(p) for p in patch_sizes]
    
    # Call C extension with optional parameters
    if max_overlap_threshold is not None and coverage is not None:
        patches, coordinates, indices, lengths = flip_position.sample_continuous_patches(
            image, position_np, num_tokens, patch_sizes_list, max_overlap_threshold, coverage
        )
    else:
        patches, coordinates, indices, lengths = flip_position.sample_continuous_patches(
            image, position_np, num_tokens, patch_sizes_list
        )
    
    # Convert to PyTorch tensors
    input_patches = [torch.from_numpy(p.transpose(0, 3, 1, 2)) for p in patches]
    input_coordinates = [torch.from_numpy(c.astype(np.float32)) for c in coordinates]
    target_indices = [torch.from_numpy(t) for t in indices]
    seq_lengths = [torch.tensor([s]) for s in lengths]
    
    return input_patches, input_coordinates, target_indices, seq_lengths

def sample_mask_with_cropping(image, position_tensor, samples_per_group):
    """
    Sample pixels from a mask using efficient cropping based on rotated Gaussian parameters.
    
    Args:
        image: A grayscale image (H x W) as numpy array
        position_tensor: Tensor with [mu_x, mu_y, sigma_x, sigma_y, rot_a, rot_b]
        samples_per_group: List of 7 integers specifying samples to take from each boundary distance group
        
    Returns:
        pixel_values: Tensor of sampled pixel values (0 or 1)
        coordinates: Tensor of normalized coordinates
        seq_lengths: Tensor of sequence length
    """
    # Convert torch tensor to numpy array
    position_np = position_tensor.detach().cpu().numpy()
    
    # Extract individual parameters
    mu_x = float(position_np[0])
    mu_y = float(position_np[1])
    sigma_x = float(position_np[2])
    sigma_y = float(position_np[3])
    rot_a = float(position_np[4])
    rot_b = float(position_np[5])
    
    # Apply the same adjustments as in the original sampling function
    H, W = image.shape[:2]
    
    # Convert from normalized coordinates (-1 to 1) to image coordinates
    mu_x = (mu_x + 1) * W / 2
    mu_y = (mu_y + 1) * H / 2
    sigma_x = sigma_x * W / 2
    sigma_y = sigma_y * H / 2
    
    # Flip rotation
    rot_a = -rot_a
    
    # Convert samples_per_group to a Python list if it's a tensor
    if isinstance(samples_per_group, torch.Tensor):
        samples_per_group = samples_per_group.tolist()
    
    # Call C extension
    results = flip_gaussian.sample_mask_with_cropping(
        image, mu_x, mu_y, sigma_x, sigma_y, rot_a, rot_b, samples_per_group
    )

    if results is None:
        return None, None, None

    pixel_values, coordinates, count = results
    
    # Convert to PyTorch tensors
    pixel_values = torch.from_numpy(pixel_values)
    coordinates = torch.from_numpy(coordinates.astype(np.float32))
    seq_lengths = torch.tensor([count])
    
    return pixel_values, coordinates, seq_lengths

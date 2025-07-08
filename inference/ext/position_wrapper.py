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


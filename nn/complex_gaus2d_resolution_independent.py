import numpy as np
import torch as th
from torch import nn
from einops import rearrange, repeat, reduce

@th.jit.script
def sparse_gaus2d(input, coordinates, seq_lengths):
    input = th.repeat_interleave(input, seq_lengths, dim=0)
    x, y, std_x, std_y, rot_a, rot_b = th.split(input, 1, dim=-1)

    # normalize rotation
    scale = th.sqrt(th.clip(rot_a**2 + rot_b**2, min=1e-16))
    rot_a = rot_a / scale
    rot_b = rot_b / scale

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
    z_x = x_rot**2 / std_x**2
    z_y = y_rot**2 / std_y**2

    return th.exp(-(z_x + z_y) / 2)

@th.jit.script
def sparse_bbox2mask(bbox, coordinates, seq_lengths):
    bbox = th.repeat_interleave(bbox, seq_lengths, dim=0)
    x_min, y_min, x_max, y_max = th.split(bbox, 1, dim=-1)
    x, y = th.split(coordinates, 1, dim=-1)

    return (x >= x_min).float() * (x <= x_max).float() * (y >= y_min).float() * (y <= y_max).float()

@th.jit.script
def sparse_embedding(input, coordinates, seq_lengths):
    input = th.repeat_interleave(input, seq_lengths, dim=0)
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
    #xy = th.cat([z_x, z_y], dim=1)
    #xy = xy / (1 + th.abs(xy / 10))
    #return xy

class ComplexGaus2D(nn.Module):
    def __init__(self):
        super(ComplexGaus2D, self).__init__()

    def forward(self, input, coordinates, seq_lengths):
        assert input.shape[1] == 6
        assert coordinates.shape[1] == 2
        return sparse_gaus2d(input, coordinates, seq_lengths)

class ComplexEmbedding(nn.Module):
    def __init__(self):
        super(ComplexEmbedding, self).__init__()

    def forward(self, input, coordinates, seq_lengths):
        assert input.shape[1] == 6
        assert coordinates.shape[1] == 2
        return sparse_embedding(input, coordinates, seq_lengths)


import torch as th
from torch import nn
from einops import rearrange, repeat, reduce

class ComplexGaus2D(nn.Module):
    def __init__(self, size=None, position_limit=1, max_std=0.5):
        super(ComplexGaus2D, self).__init__()
        self.size = size
        self.position_limit = position_limit
        self.min_std = 0.1
        self.max_std = max_std

        self.register_buffer("grid_x", th.zeros(1, 1, 1, 1), persistent=False)
        self.register_buffer("grid_y", th.zeros(1, 1, 1, 1), persistent=False)

        if size is not None:
            self.min_std = 1.0 / min(size)
            self.update_grid(size)

    def update_grid(self, size):
        if size != self.grid_x.shape[2:]:
            self.size = size
            self.min_std = 1.0 / min(size)
            H, W = size

            self.grid_x = th.arange(W, device=self.grid_x.device).float()
            self.grid_y = th.arange(H, device=self.grid_x.device).float()

            self.grid_x = (self.grid_x / (W - 1)) * 2 - 1
            self.grid_y = (self.grid_y / (H - 1)) * 2 - 1

            self.grid_x = self.grid_x.view(1, 1, 1, -1).expand(1, 1, H, W).clone()
            self.grid_y = self.grid_y.view(1, 1, -1, 1).expand(1, 1, H, W).clone()

    def from_rot(self, input: th.Tensor):
        assert input.shape[1] == 6
        H, W = self.size

        x = rearrange(input[:, 0:1], 'b c -> b c 1 1')
        y = rearrange(input[:, 1:2], 'b c -> b c 1 1')
        std_x = rearrange(input[:, 2:3], 'b c -> b c 1 1')
        std_y = rearrange(input[:, 3:4], 'b c -> b c 1 1')
        rot_a = rearrange(input[:, 4:5], 'b c -> b c 1 1')
        rot_b = rearrange(input[:, 5:6], 'b c -> b c 1 1')

        # normalize rotation
        scale = th.sqrt(th.clip(rot_a**2 + rot_b**2, min=1e-16))
        rot_a = rot_a / scale
        rot_b = rot_b / scale

        # Clip values for stability
        x = th.clip(x, -self.position_limit, self.position_limit)
        y = th.clip(y, -self.position_limit, self.position_limit)
        std_x = th.clip(std_x, self.min_std, self.max_std)
        std_y = th.clip(std_y, self.min_std, self.max_std)

        # compute relative coordinates
        x = self.grid_x - x
        y = self.grid_y - y

        # Compute rotated coordinates
        x_rot = rot_a * x - rot_b * y
        y_rot = rot_b * x + rot_a * y

        # Compute Gaussian distribution with rotated coordinates
        z_x = x_rot**2 / std_x**2
        z_y = y_rot**2 / std_y**2

        return th.exp(-(z_x + z_y) / 2)

    def integral(self, input: th.Tensor):
        assert input.shape[1] >= 5

        var_x    = input[:, 2:3]
        var_y    = input[:, 3:4]
        covar_xy = input[:, 4:5]

        # Clip values for stability
        var_x    = th.clip(var_x, self.min_std**2, self.max_std**2)
        var_y    = th.clip(var_y, self.min_std**2, self.max_std**2)
        covar_xy = th.clip(covar_xy, -self.max_std**2, self.max_std**2)

        return 2 * th.pi * th.sqrt(th.clip(var_x * var_y - covar_xy**2, min=1e-16))

    def from_rho(self, input: th.Tensor):
        assert input.shape[1] >= 5
        H, W = self.size

        x        = rearrange(input[:, 0:1], 'b c -> b c 1 1')
        y        = rearrange(input[:, 1:2], 'b c -> b c 1 1')
        var_x    = rearrange(input[:, 2:3], 'b c -> b c 1 1')
        var_y    = rearrange(input[:, 3:4], 'b c -> b c 1 1')
        covar_xy = rearrange(input[:, 4:5], 'b c -> b c 1 1')

        # Clip values for stability
        x        = th.clip(x, -self.position_limit, self.position_limit)
        y        = th.clip(y, -self.position_limit, self.position_limit)
        var_x    = th.clip(var_x, self.min_std**2, self.max_std**2)
        var_y    = th.clip(var_y, self.min_std**2, self.max_std**2)
        covar_xy = th.clip(covar_xy, -self.max_std**2, self.max_std**2)

        z_x = (self.grid_x - x) ** 2 / var_x
        z_y = (self.grid_y - y) ** 2 / var_y
        z_xy = 2 * covar_xy * (self.grid_x - x) * (self.grid_y - y) / (var_x * var_y)
        
        Z = th.clip(z_x + z_y - z_xy, min=1e-8)
        denom = th.clip(2 * (1 - (covar_xy**2) / (var_x * var_y)), min=1e-8)

        return th.exp(-Z / denom)

    def forward(self, input: th.Tensor):
        assert input.shape[1] >= 5 and input.shape[1] <= 6
        if input.shape[1] == 5:
            return self.from_rho(input)
        else:
            return self.from_rot(input)

def compute_gaussian_params_from_mask(mask):
    B, C, height, width = mask.shape
    mask = (mask > 0.5).float()
    
    # Create a meshgrid
    x_range = th.linspace(-1, 1, width).to(mask.device)
    y_range = th.linspace(-1, 1, height).to(mask.device)
    x_grid, y_grid = th.meshgrid(x_range, y_range, indexing='xy')
    y_grid = y_grid.expand(B, 1, height, width)
    x_grid = x_grid.expand(B, 1, height, width)

    # Filter the coordinates using element-wise multiplication with the binary mask
    x = x_grid * mask
    y = y_grid * mask

    # Compute mean
    sum_mask = th.clip(th.sum(mask, dim=[2, 3], keepdim=True), min=1e-16)
    mu_x = th.sum(x, dim=[2, 3], keepdim=True) / sum_mask
    mu_y = th.sum(y, dim=[2, 3], keepdim=True) / sum_mask

    mu_x2 = th.sum(x**2, dim=[2, 3], keepdim=True) / sum_mask
    mu_y2 = th.sum(y**2, dim=[2, 3], keepdim=True) / sum_mask

    mu_xy = th.sum(x*y, dim=[2, 3], keepdim=True) / sum_mask
    
    mu_x = mu_x[:, 0, 0, 0]
    mu_y = mu_y[:, 0, 0, 0]
    mu_x2 = mu_x2[:, 0, 0, 0]
    mu_y2 = mu_y2[:, 0, 0, 0]
    mu_xy = mu_xy[:, 0, 0, 0]

    sigma_x2 = mu_x2 - mu_x**2
    sigma_y2 = mu_y2 - mu_y**2
    sigma_xy = mu_xy - mu_x * mu_y

    position_rho = th.stack((mu_x, mu_y, sigma_x2, sigma_y2, sigma_xy), dim=1)
    
    # Eigen decomposition
    term    = th.sqrt(th.clip((sigma_x2 - sigma_y2)**2 + 4*sigma_xy**2, min=1e-16))
    sigma_x = th.sqrt(th.clip((sigma_x2 + sigma_y2) / 2 - term / 2, min=1e-16))
    sigma_y = th.sqrt(th.clip((sigma_x2 + sigma_y2) / 2 + term / 2, min=1e-16))
    
    # Compute eigenvectors
    rot_b = th.ones_like(sigma_x)
    rot_a = 2 * sigma_xy * rot_b / th.clip(sigma_x2 - sigma_y2 + term, min=1e-16)

    # Normalize rotation components
    scale = th.sqrt(th.clip(rot_a ** 2 + rot_b ** 2, min=1e-16))
    rot_a = rot_a / scale
    rot_b = rot_b / scale
    
    position_rot = th.stack((mu_x, mu_y, sigma_x, sigma_y, rot_a, rot_b), dim=1)
    return position_rho, position_rot



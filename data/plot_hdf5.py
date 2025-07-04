import h5py
import argparse
import matplotlib.pyplot as plt
import numpy as np
import random
from turbojpeg import decompress
from PIL import Image
import io
import torch as th
import torch.nn as nn
from einops import rearrange
import cv2

class ComplexGaus2D(nn.Module):
    def __init__(self, size=None, position_limit=1):
        super(ComplexGaus2D, self).__init__()
        self.size = size
        self.position_limit = position_limit
        self.min_std = 0.1
        self.max_std = 0.5

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

def decode_jpeg(image): 
    if len(image.shape) > 1:
        return image.astype(np.float32).transpose((1, 2, 0)) / 255.0
    return decompress(image)

def decode_png(image):
    return cv2.imdecode(image, cv2.IMREAD_UNCHANGED)

def plot_dataset(h5_file, sequence_index, image_index = None):
    plt.clf()

    if '/sequence_indices' not in h5_file:
        image_index = sequence_index

    if image_index is None:
        sequence_start, sequence_lenght = h5_file['/sequence_indices'][sequence_index]
        sequence_end = sequence_start + sequence_lenght 
        image_index = random.randint(sequence_start, sequence_end)
    print(f'Plotting sequence {sequence_index} image {image_index}')
        
    mask_index_start = 0
    mask_index_end = 0
    if 'image_instance_indices' in h5_file and len(h5_file['/image_instance_indices']) > 0:
        mask_index_start, mask_index_length = h5_file['/image_instance_indices'][image_index]
        mask_index_end = mask_index_start + mask_index_length
    
    mask = None
    if '/instance_masks' in h5_file and len(h5_file['/instance_masks']) > 0:
        #mask = (h5_file['/instance_masks'][mask_index_start:mask_index_end] * (0.5+np.random.rand(mask_index_end-mask_index_start,1,1,1)*0.5)).sum(axis=0)[0]
        mask = h5_file['/instance_masks'][mask_index_start:mask_index_end]

        if len(mask.shape) == 1:
            mask = [decode_png(m) for m in mask]
            mask = th.stack([th.from_numpy(m) for m in mask]).unsqueeze(1).numpy()

        if '/positions' in h5_file and len(h5_file['/positions']) > 0:
            positions = th.from_numpy(h5_file['/positions'][mask_index_start:mask_index_end])
            gaus2d = ComplexGaus2D(mask.shape[2:])
            mask = gaus2d(positions).numpy() * mask * 0.9 + mask * 0.1

        mask = (mask * np.linspace(1, 0.2, mask_index_end-mask_index_start)[:, None, None, None]).sum(axis=0)[0]
    
    foreground_mask = None
    if '/foreground_mask' in h5_file and len(h5_file['/foreground_mask']) > 0:
        foreground_mask = h5_file['/foreground_mask'][image_index][0]
    
    plt.subplot(2, 3, 1)
    if h5_file['/rgb_images'][image_index].dtype == np.uint8:
        print('Decoding JPEG')
        plt.imshow(decode_jpeg(h5_file['/rgb_images'][image_index]))
    else:
        plt.imshow(h5_file['/rgb_images'][image_index].transpose((1, 2, 0))[:,:,::-1])
    if '/instance_mask_bboxes' in h5_file and len(h5_file['/instance_mask_bboxes']) > 0:
        for bbox in h5_file['/instance_mask_bboxes'][mask_index_start:mask_index_end]:
            plt.gca().add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], edgecolor='r', facecolor='none'))
    plt.title('RGB Image')
    
    if '/depth_images' in h5_file and len(h5_file['/depth_images']) > 0:
        plt.subplot(2, 3, 2)
        if h5_file['/depth_images'][image_index].dtype == np.uint8:
            plt.imshow(decode_png(h5_file['/depth_images'][image_index]))
        else:
            plt.imshow(h5_file['/depth_images'][image_index][0], cmap='gray')

        print(h5_file['/depth_images'][image_index].dtype)
        plt.title('Depth Image')

    if '/raw_depth' in h5_file and len(h5_file['/raw_depth']) > 0:
        plt.subplot(2, 3, 3)
        plt.imshow(h5_file['/raw_depth'][image_index][0], cmap='gray')
        plt.title('Raw Depth')

    if '/instance_depth_unoccluded' in h5_file and len(h5_file['/instance_depth_unoccluded']) > 0:
        plt.subplot(2, 3, 3)
        if h5_file['/instance_depth_unoccluded'][image_index].dtype == np.uint8:
            plt.imshow(decode_png(h5_file['/instance_depth_unoccluded'][image_index]))
        else:
            plt.imshow(h5_file['/instance_depth_unoccluded'][image_index][0], cmap='gray')
        plt.title('Raw Depth')
    
    if '/forward_flow' in h5_file and len(h5_file['/forward_flow']) > 0:
        plt.subplot(2, 3, 4)
        plt.imshow(np.linalg.norm(h5_file['/forward_flow'][image_index], axis=0), cmap='gray')
        plt.title('Forward Flow')

    if '/instance_rgb_unoccluded' in h5_file and len(h5_file['/instance_rgb_unoccluded']) > 0:
        plt.subplot(2, 3, 4)
        if h5_file['/instance_rgb_unoccluded'][image_index].dtype == np.uint8:
            plt.imshow(decode_png(h5_file['/instance_rgb_unoccluded'][image_index]))
        else:
            plt.imshow(h5_file['/instance_rgb_unoccluded'][image_index][0], cmap='gray')
        plt.title('Raw RGB')

    if '/backward_flow' in h5_file and len(h5_file['/backward_flow']) > 0:
        plt.subplot(2, 3, 5)
        plt.imshow(np.linalg.norm(h5_file['/backward_flow'][image_index], axis=0), cmap='gray')
        plt.title('Backward Flow')

    if '/instance_masks_unoccluded' in h5_file and len(h5_file['/instance_masks_unoccluded']) > 0:
        plt.subplot(2, 3, 5)
        plt.imshow(h5_file['/instance_masks_unoccluded'][image_index][0] / 255, cmap='gray')
        plt.title('Raw Mask')

    plt.subplot(2, 3, 6)
    if random.random() < 0.5 and foreground_mask is not None:
        if foreground_mask is not None:
            plt.imshow(foreground_mask, cmap='gray')
        if mask is not None:
            for bbox in h5_file['/instance_mask_bboxes'][mask_index_start:mask_index_end]:
                plt.gca().add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], edgecolor='r', facecolor='none'))
        plt.title('Foreground Mask')
    else:
        if mask is not None:
            plt.imshow(mask, cmap='gray')
            if '/instance_mask_bboxes' in h5_file and len(h5_file['/instance_mask_bboxes']) > 0:
                for bbox in h5_file['/instance_mask_bboxes'][mask_index_start:mask_index_end]:
                    plt.gca().add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], edgecolor='r', facecolor='none'))
        plt.title('Instance Mask and BBox')

    plt.tight_layout()
    plt.gcf().canvas.draw()

sequence_index = 0

def on_key(event, h5_file, num_sequences, num_images = None):
    global sequence_index
    img_index = None
    if event.key == 'n':
        #sequence_index = random.randint(0, num_sequences - 1)
        #img_index = random.randint(0, num_images - 1) if num_images is not None else None

        sequence_index = sequence_index + 1
        plot_dataset(h5_file, sequence_index, sequence_index)

    if event.key == 'm':
        sequence_index = random.randint(0, num_sequences - 1)
        plot_dataset(h5_file, sequence_index, None)

    if event.key == 'r':
        plot_dataset(h5_file, sequence_index, sequence_index)

def main():
    parser = argparse.ArgumentParser(description="Plot random image data for a given HDF5 file.")
    parser.add_argument("filename", help="Path to the HDF5 file.")
    args = parser.parse_args()

    with h5py.File(args.filename, 'r') as h5_file:
        num_sequences = len(h5_file['/sequence_indices']) if '/sequence_indices' in h5_file else len(h5_file['/rgb_images'])
        image_index = num_images = None
        if num_sequences == 0:
            num_sequences = len(h5_file['/rgb_images'])
            num_images = len(h5_file['/rgb_images'])
            image_index = random.randint(0, num_sequences - 1)

        fig = plt.figure(figsize=(15, 10))  # Create the figure upfront
        fig.canvas.mpl_connect('key_press_event', lambda event: on_key(event, h5_file, num_sequences, num_images))  # Connect the event handler
        plot_dataset(h5_file, 0, 0)
        plt.show()  # Keep show() at the end to keep the figure open

if __name__ == "__main__":
    main()

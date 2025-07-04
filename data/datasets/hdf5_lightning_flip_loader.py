import torch as th
import torch.nn as nn
from torch.utils.data import Dataset
import h5py
import numpy as np
import cv2
import math
from turbojpeg import decompress as jpeg_decompress
# Import optimized C extension functions
from ext.position_wrapper import (
    compute_position_rot_from_rho,
    apply_position_augmentation,
    sample_num_tokens,
    sample_continuous_patches,
    sample_mask_with_cropping,  
)

def decompress(image):
    return np.array(jpeg_decompress(image)).astype(np.uint8)


class HDF5_Dataset(Dataset):
    def __init__(
        self, 
        hdf5_file_path,
        num_masks_per_image=2,
        cfg=None,
        seed=1234, 
        shared_samples_per_group=None,
        sampling_lock=None
    ):
        print("NUM_FRAMES", num_masks_per_image)
        self.filename = hdf5_file_path

        self.num_masks_per_image = num_masks_per_image

        self.hdf5_file_path = hdf5_file_path
        self.hdf5_file = h5py.File(hdf5_file_path, "r")

        if "sequence_indices" in self.hdf5_file and len(self.hdf5_file["sequence_indices"]) > 0:
            self.sequence_indices = self.hdf5_file["sequence_indices"][:] 
        else:
            self.sequence_indices = np.stack((
                np.arange(len(self.hdf5_file["rgb_images"])).astype(np.int32), 
                np.ones(len(self.hdf5_file["rgb_images"])).astype(np.int32)
            ), axis=1)

        self.image_instance_indices = self.hdf5_file["image_instance_indices"][:]
        self.dataset_length   = len(self.sequence_indices)

        np.random.seed(seed)
        self.indices = np.arange(self.dataset_length)
        np.random.shuffle(self.indices)

        self.hdf5_file.close()
        self.hdf5_file = None

        self.max_patch_size = cfg.model.max_patch_size
        self.min_patch_size = cfg.model.min_patch_size

        self.min_num_patches = cfg.model.min_num_tokens
        self.max_num_patches = cfg.model.max_num_tokens
        self.avg_num_patches = cfg.model.avg_num_tokens

        min_patch_size = int(math.log2(self.min_patch_size))
        max_patch_size = int(math.log2(self.max_patch_size))
        self.patch_sizes = (2**th.linspace(min_patch_size, max_patch_size, max_patch_size - min_patch_size + 1)).long().tolist()

        # Setup for shared memory
        self.shared_samples_per_group = shared_samples_per_group
        self.sampling_lock = sampling_lock
        
        self.default_samples_per_group = [cfg.model.num_mask_pixels // 7] * 7
        self.default_samples_per_group[0] += cfg.model.num_mask_pixels % 7

        print(f"Loaded HDF5 dataset {hdf5_file_path} with size {self.dataset_length}", flush=True)

    def get_current_samples_per_group(self):
        """
        Get the current samples_per_group values, either from shared memory or local copy.
        """
        if self.shared_samples_per_group is not None:
            with self.sampling_lock:
                return self.shared_samples_per_group.tolist()
        else:
            return self.default_samples_per_group


    def __len__(self):
        return self.dataset_length

    def load_hdf5_file(self):
        if self.hdf5_file is None:
            self.hdf5_file = h5py.File(self.hdf5_file_path, "r")

    def get_data(self, seq_index, frame_index, mask_offset, input_position = None, num_tokens=None):
        self.load_hdf5_file()
        sequence_start, sequence_len = self.sequence_indices[seq_index]
        sequence_end = sequence_start + sequence_len
        frame_index = sequence_start + (frame_index % sequence_len)

        num_masks = self.image_instance_indices[sequence_start][1]
        mask_offset = mask_offset % num_masks
        mask_index = self.image_instance_indices[frame_index][0] + mask_offset

        rgb_image = self.hdf5_file["rgb_images"][frame_index]
        if rgb_image.dtype == np.uint8 and len(rgb_image.shape) == 1:
            rgb_image = decompress(rgb_image)

        instance_mask = self.hdf5_file["instance_masks"][mask_index]

        if instance_mask.dtype == np.uint8 and len(instance_mask.shape) == 1:
            instance_mask = cv2.imdecode(instance_mask, cv2.IMREAD_UNCHANGED)
            instance_mask = np.expand_dims(instance_mask, axis=2)
            instance_mask[...] = instance_mask > 128 
        else:
            instance_mask[...] = instance_mask > 128
            instance_mask = instance_mask.transpose(1, 2, 0)

        data = self._load(frame_index, mask_index, num_tokens, input_position)
        if data is None:
            return self.get_data(np.random.randint(0, len(self)), frame_index, mask_offset)

        input_patches = data['input_patches']
        input_coordinates = data['input_coordinates']
        target_indices = data['target_indices']
        seq_lengths = data['seq_lengths']
        resolution = data['resolution']
        input_position = data['input_position']
        gt_position = data['gt_position']
        gt_bbox = data['gt_bbox']
        mask_coordinates = data['mask_coordinates']
        mask_targets = data['mask_targets']
        mask_seq_lengths = data['mask_seq_lengths']

                
        resolution = resolution.unsqueeze(0)
        input_position = input_position.unsqueeze(0)
        gt_position = gt_position.unsqueeze(0)
        gt_bbox = gt_bbox.unsqueeze(0)

        gt_bbox_radii = th.stack(((gt_bbox[0,2] - gt_bbox[0,0]) / 2, (gt_bbox[0,3] - gt_bbox[0,1]) / 2), dim=0)
        loss_valid = (gt_position[0,2] > 0.005).float() * (gt_position[0,3] > 0.005).float()
        loss_valid = loss_valid * (gt_bbox_radii[0] > 0.005).float() * (gt_bbox_radii[1] > 0.005).float()
        loss_valid = loss_valid * (gt_bbox_radii[0] < 25).float()    * (gt_bbox_radii[1] < 25).float()
        loss_valid = loss_valid * float(th.stack(seq_lengths).sum().item() >= self.min_num_patches)
        loss_valid = loss_valid * (th.stack(seq_lengths).sum() > 0.5).float()

        if loss_valid == 0:
            return self.get_data(np.random.randint(0, len(self)), frame_index, mask_offset)

        return rgb_image, instance_mask, input_patches, input_coordinates, target_indices, seq_lengths, resolution, input_position, gt_position, gt_bbox, mask_coordinates, mask_targets, mask_seq_lengths

    def load_rgb(self, frame_index):

        rgb_image = self.hdf5_file["rgb_images"][frame_index]

        # Handle compressed datasets
        if rgb_image.dtype == np.uint8 and len(rgb_image.shape) == 1:
            rgb_image = decompress(rgb_image)

        return rgb_image

    def load_mask(self, mask_index):

        instance_mask = self.hdf5_file["instance_masks"][mask_index]

        if instance_mask.dtype == np.uint8 and len(instance_mask.shape) == 1:
            instance_mask = cv2.imdecode(instance_mask, cv2.IMREAD_UNCHANGED)

        instance_mask = (instance_mask > 128).squeeze().astype(np.uint8)

        return instance_mask


    def _load(self, frame_index, mask_index, num_tokens=None, input_position_override=None, img_override=None):
        image = self.load_rgb(frame_index) if img_override is None else img_override
        instance_mask = self.load_mask(mask_index)

        gt_position = compute_position_rot_from_rho(th.from_numpy(self.hdf5_file["positions"][mask_index:mask_index+1]))[0]
        gt_bbox     = th.from_numpy(self.hdf5_file["instance_mask_bboxes"][mask_index]).float()

        input_position = gt_position.clone()
        H, W = image.shape[:2]

        positive_sample = True
        simple_sample = True

        randv = np.random.rand()

        # Use the get_current_samples_per_group method to get the latest values
        samples_per_group = self.get_current_samples_per_group()
        
        # shift the position
        input_position = apply_position_augmentation(gt_position)

        if input_position_override is not None:
            input_position = input_position_override.clone()

            # normalize the position
            input_position[0] /= W/256
            input_position[1] /= H/256
            input_position[2] /= W/256
            input_position[3] /= H/256

            positive_sample = True
            simple_sample = True

        # sample patches
        num_tokens = sample_num_tokens(self.min_num_patches, self.max_num_patches, self.avg_num_patches) if num_tokens is None else num_tokens
        input_patches, input_coordinates, target_indices, seq_lengths = sample_continuous_patches(image, input_position.clone(), num_tokens, self.patch_sizes)
        mask_targets, mask_coordinates, mask_seq_lengths = sample_mask_with_cropping(instance_mask, input_position.clone(), samples_per_group)

        if mask_targets is None:
            return None

        H, W = image.shape[:2]

        # denormalize the position
        gt_position[0] *= W/256
        gt_position[1] *= H/256
        gt_position[2] *= W/256
        gt_position[3] *= H/256
        input_position[0] *= W/256
        input_position[1] *= H/256
        input_position[2] *= W/256
        input_position[3] *= H/256

        gt_bbox[0] = gt_bbox[0] / 128 - W / 256
        gt_bbox[1] = gt_bbox[1] / 128 - H / 256
        gt_bbox[2] = gt_bbox[2] / 128 - W / 256
        gt_bbox[3] = gt_bbox[3] / 128 - H / 256

        resolution = th.tensor([H, W])
        simple_sample   = th.tensor([simple_sample], dtype=th.bool)
        positive_sample = th.tensor([positive_sample], dtype=th.bool)

        gt_bbox_radii = th.stack(((gt_bbox[2] - gt_bbox[0]) / 2, (gt_bbox[3] - gt_bbox[1]) / 2), dim=0)
        loss_valid = (gt_position[2] > 0.005).float() * (gt_position[3] > 0.005).float()
        loss_valid = loss_valid * (gt_bbox_radii[0] > 0.005).float() * (gt_bbox_radii[1] > 0.005).float()
        loss_valid = loss_valid * (gt_bbox_radii[0] < 25).float()    * (gt_bbox_radii[1] < 25).float()
        loss_valid = loss_valid * float(th.stack(seq_lengths).sum().item() >= self.min_num_patches)
        loss_valid = loss_valid * (th.stack(seq_lengths).sum() > 0.5).float()

        if loss_valid == 0:
            return None

        return {
            'input_patches': input_patches,
            'input_coordinates': input_coordinates,
            'target_indices': target_indices,
            'seq_lengths': seq_lengths,
            'resolution': resolution,
            'input_position': input_position,
            'gt_position': gt_position,
            'gt_bbox': gt_bbox,
            'simple_sample': simple_sample,
            'positive_sample': positive_sample,
            'mask_targets': mask_targets,
            'mask_coordinates': mask_coordinates,
            'mask_seq_lengths': mask_seq_lengths,
        }

    def __getitem__(self, index):
        with th.no_grad():

            # Open the HDF5 file if it is not already open
            self.load_hdf5_file()

            # Get the start and end indices of the sequence
            sequence_start, sequence_len = self.sequence_indices[index]
            sequence_end = sequence_start + sequence_len

            # get the number of masks per image
            num_masks = self.image_instance_indices[sequence_start][1]

            # Initialize the output dictionary
            data = []

            # Choose a single frame from the sequence
            if sequence_len > 1:
                frame_index = np.random.choice(range(sequence_start, sequence_end))
            else:
                frame_index = sequence_start
                
            # Load the image once
            image = self.load_rgb(frame_index)
            
            # Choose num_masks_per_image different masks randomly
            mask_range   = range(num_masks)
            mask_offsets = np.random.choice(mask_range, self.num_masks_per_image, replace=(len(mask_range) < self.num_masks_per_image))
            mask_indices = [self.image_instance_indices[frame_index][0] + mask_offset for mask_offset in mask_offsets]
            
            # Process each mask with the same image
            for t, mask_index in enumerate(mask_indices):
                # Use the _load method with the same image but different masks
                frame_data = self._load(frame_index, mask_index, img_override=image)
                
                if frame_data is None:
                    return self.__getitem__(np.random.randint(0, len(self)))

                data.append({})
                data[t]['input_patches']     = frame_data['input_patches']
                data[t]['input_coordinates'] = frame_data['input_coordinates']
                data[t]['target_indices']    = frame_data['target_indices']
                data[t]['seq_lengths']       = frame_data['seq_lengths']
                data[t]['resolution']        = frame_data['resolution']
                data[t]['input_position']    = frame_data['input_position']
                data[t]['gt_position']       = frame_data['gt_position']
                data[t]['gt_bbox']           = frame_data['gt_bbox']
                data[t]['simple_sample']     = frame_data['simple_sample']
                data[t]['positive_sample']   = frame_data['positive_sample']
                data[t]['mask_targets']      = frame_data['mask_targets']
                data[t]['mask_coordinates']  = frame_data['mask_coordinates']
                data[t]['mask_seq_lengths']  = frame_data['mask_seq_lengths']

            return {'data': data}


class HDF5_DatasetGroup(Dataset):
    def __init__(self, datasets, group_weight, name = "unnamed"):
        self.datasets = datasets
        self.lenght   = sum([len(d) for d in self.datasets])
        self.group_weight = group_weight
        self.name = name

        print(f"{name} dataset size: {self.lenght}")

    def update_samples_per_group(self, samples_per_group):
        for d in self.datasets:
            d.update_samples_per_group(samples_per_group)

    def __len__(self):
        return self.lenght

    def get_weight(self):
        return self.group_weight

    def get_data(self, seq_index, frame_index, mask_offset, input_position = None, num_tokens=None):
        for d in self.datasets:
            if seq_index < len(d):
                return d.get_data(seq_index, frame_index, mask_offset, input_position, num_tokens)
            else:
                seq_index -= len(d)

        assert False, f"index out of bounds {index} {self.lenght} {self.name} {len(self.datasets)}"

    def __getitem__(self, index):
        for d in self.datasets:
            if index < len(d):
                return d[index]
            else:
                index -= len(d)

        assert False, f"index out of bounds {index} {self.lenght} {self.name} {len(self.datasets)}"

class ChainedHDF5_Dataset(Dataset):
    def __init__(self, hdf5_datasets, length_multiplier = 1):
        self.datasets = hdf5_datasets
        self.dataset_offset = 1000**3
        self.length_multiplier = length_multiplier

        weights = np.array([d.get_weight() for d in self.datasets])

        total_length = sum([len(d) for d in self.datasets])
        for i, d in enumerate([len(d) for d in self.datasets]):
            if weights[i] < 0:
                weights[i] = d / total_length

        self.weights  = weights / np.sum(weights)

        self.lenght  = sum([int(total_length * w) for w in self.weights])

        print(f"dataset size: {self.lenght}")
        for d, w in zip(self.datasets, self.weights):
            print(f"resampling dataset {len(d):10d}|{100*len(d)/total_length:7.3f}% -> {int(total_length * w):12d}|{100*w:7.3f}% ({d.name})")

        self.cumulative_lengths = np.cumsum([int(total_length * w) for w in self.weights])

    def update_samples_per_group(self, samples_per_group):
        for d in self.datasets:
            d.update_samples_per_group(samples_per_group)

    def __len__(self):
        return self.lenght

    def get_data(self, seq_index, frame_index, mask_offset, input_position = None, num_tokens=None):
        for d in self.datasets:
            if seq_index < len(d):
                return d.get_data(seq_index, frame_index, mask_offset, input_position, num_tokens)
            else:
                seq_index -= len(d)

        assert False, f"index out of bounds {index} {self.lenght}"

    def __getitem__(self, combined_index):

        dataset_index = combined_index // self.dataset_offset
        index         = combined_index % self.dataset_offset

        batch = self.datasets[dataset_index][index]
        # Add dataset length to the batch
        batch['dataset_length'] = th.tensor([self.lenght * self.length_multiplier])
        
        return batch

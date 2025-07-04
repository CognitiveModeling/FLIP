from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from utils.configuration import Configuration
import torch as th
from data.datasets.hdf5_lightning_flip_loader import HDF5_Dataset, HDF5_DatasetGroup, ChainedHDF5_Dataset
from data.sampler.object_sampler import DistributedObjectSampler
import torch.multiprocessing as mp
from torch.multiprocessing import Lock

def custom_collate_fn(batch):
    """
    Custom collate function that processes batches from the HDF5_Dataset.
    Organizes data by time steps and patch sizes, combining across batch dimension.
    
    Args:
        batch: List of dictionaries, each containing a 'data' key with time step data
        
    Returns:
        Tuple of processed tensors
    """
    num_timesteps = len(batch[0]['data'])

    # Initialize lists for each component
    input_patches = []
    input_coordinates = []
    target_indices = [] 
    seq_lengths = []
    resolutions = []
    input_positions = []
    gt_positions = []
    gt_bboxes = []
    mask_targets = []
    mask_coordinates = []
    mask_seq_lengths = []
    dataset_length = []

    
    # Process each batch item
    for item in batch:
        # Collect data from all time steps
        for t, time_data in enumerate(item['data']):
            input_patches.append(time_data['input_patches'])
            input_coordinates.append(time_data['input_coordinates'])
            target_indices.append(time_data['target_indices'])
            seq_lengths.append(time_data['seq_lengths'])
            resolutions.append(time_data['resolution'])
            input_positions.append(time_data['input_position'])
            gt_positions.append(time_data['gt_position'])
            gt_bboxes.append(time_data['gt_bbox'])
            mask_targets.append(time_data['mask_targets'])
            mask_coordinates.append(time_data['mask_coordinates'])
            mask_seq_lengths.append(time_data['mask_seq_lengths'])
        
        dataset_length.append(item['dataset_length'])


    # Process input_patches (concatenate across batch for each patch size)
    input_patches = [
        th.cat([sample[patch_idx] for sample in input_patches], dim=0)
        for patch_idx in range(len(input_patches[0]))
    ]

    input_coordinates = [
        th.cat([sample[patch_idx] for sample in input_coordinates], dim=0)
        for patch_idx in range(len(input_coordinates[0]))
    ]

    target_indices = [
        th.cat([sample[patch_idx] for sample in target_indices], dim=0)
        for patch_idx in range(len(target_indices[0]))
    ]

    seq_lengths = [
        th.cat([sample[patch_idx] for sample in seq_lengths], dim=0)
        for patch_idx in range(len(seq_lengths[0]))
    ]

    # Process other tensors
    resolutions = th.stack(resolutions)
    input_positions = th.stack(input_positions)
    gt_positions = th.stack(gt_positions)
    gt_bboxes = th.stack(gt_bboxes)
    mask_targets = th.cat(mask_targets)
    mask_coordinates = th.cat(mask_coordinates)
    mask_seq_lengths = th.cat(mask_seq_lengths)

    # Process dataset_length if present
    dataset_length = th.cat(dataset_length)
    
    return {
        'input_patches':     input_patches,
        'input_coordinates': input_coordinates,
        'target_indices':    target_indices,
        'seq_lengths':       seq_lengths,
        'resolutions':       resolutions,
        'input_position':    input_positions,
        'gt_position':       gt_positions,
        'gt_bbox':           gt_bboxes,
        'mask_targets':      mask_targets,
        'mask_coordinates':  mask_coordinates,
        'mask_seq_lengths':  mask_seq_lengths,
        'dataset_length':    dataset_length
    }

class FlipDataModule(LightningDataModule):
    def __init__(self, cfg: Configuration):
        super().__init__()

        # Set up shared memory for samples_per_group
        samples_per_group = [cfg.model.num_mask_pixels // 7] * 7
        samples_per_group[0] += cfg.model.num_mask_pixels % 7
        
        # Create shared tensor for samples_per_group
        self.shared_samples_per_group = th.tensor(samples_per_group).share_memory_()
        
        # Create a shared lock
        self.sampling_lock = mp.Lock()
        
        trainset = []
        valset = []
        testset = []

        DatasetClass = HDF5_Dataset

        for group in cfg.data.train:
            data_group = []
            
            for path in group.paths:
                data_group.append(DatasetClass(
                    hdf5_file_path           = path, 
                    num_masks_per_image      = cfg.num_masks_per_image,
                    cfg                      = cfg,
                    seed                     = cfg.seed,
                    shared_samples_per_group = self.shared_samples_per_group,
                    sampling_lock            = self.sampling_lock
                ))

            trainset.append(HDF5_DatasetGroup(data_group, group.weight, group.name))

        if 'val' in cfg.data:
            for group in cfg.data.val:
                data_group = []
                for path in group.paths:
                    data_group.append(DatasetClass(
                        hdf5_file_path           = path,
                        num_masks_per_image      = cfg.num_masks_per_image,
                        cfg                      = cfg,
                        shared_samples_per_group = self.shared_samples_per_group,
                        sampling_lock            = self.sampling_lock
                    ))

                valset.append(HDF5_DatasetGroup(data_group, group.weight, group.name))

        self.cfg = cfg
        self.trainset = ChainedHDF5_Dataset(trainset, cfg.trainset_length_multiplier)
        self.valset = ChainedHDF5_Dataset(valset) 

        self.batch_size = self.cfg.model.batch_size
        self.samplers = []
        self.trainset_length_multiplier = cfg.trainset_length_multiplier
        self.testset_length_multiplier = cfg.testset_length_multiplier
    
    def update_samples_per_group(self, new_samples_per_group):
        """
        Update the shared samples_per_group tensor.
        Args:
            new_samples_per_group: New values for samples_per_group
        """
        with self.sampling_lock:
            # Convert to tensor if needed
            if not isinstance(new_samples_per_group, th.Tensor):
                new_samples_per_group = th.tensor(new_samples_per_group, 
                                                 dtype=self.shared_samples_per_group.dtype,
                                                 device=self.shared_samples_per_group.device)
            
            # Update the shared tensor
            self.shared_samples_per_group.copy_(new_samples_per_group)

    def train_dataloader(self):
        self.samplers.append(DistributedObjectSampler(self.trainset, shuffle=True, seed=self.cfg.seed, length_multiplier=self.trainset_length_multiplier))

        return DataLoader(
            self.trainset,
            collate_fn = custom_collate_fn,
            pin_memory=True,
            num_workers=self.cfg.num_workers,
            batch_size=self.batch_size,
            sampler=self.samplers[-1],
            drop_last=True,
            prefetch_factor=self.cfg.prefetch_factor,
            persistent_workers=True,
            shuffle=False
        )

    def val_dataloader(self):
        sampler = DistributedObjectSampler(self.valset, shuffle=True, length_multiplier=self.testset_length_multiplier)

        return DataLoader(
            self.valset, 
            collate_fn = custom_collate_fn,
            pin_memory=True, 
            num_workers=self.cfg.num_workers, 
            batch_size=self.batch_size,
            sampler=sampler,
            drop_last=True, 
            prefetch_factor=self.cfg.prefetch_factor, 
            persistent_workers=True,
            shuffle=False
        )

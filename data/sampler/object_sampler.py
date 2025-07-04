import math
import numpy as np

import torch
from torch.utils.data import Dataset, Sampler
import torch.distributed as dist
from tqdm import tqdm
from collections import deque

def is_dist_initialized():
    """Check if distributed training is initialized."""
    if not dist.is_available():
        return False
    try:
        if not dist.is_initialized():
            return False
    except:
        return False
    return True

class DistributedHDF5DatasetSampler():
    def __init__(self, hdf5_dataset, shuffle=True, seed=1234):
        self.dataset = hdf5_dataset
        self.shuffle = shuffle
        self.seed    = seed
        self.epoch   = 0
        
        # Check if distributed is initialized
        self.distributed = is_dist_initialized()
        
        if self.distributed:
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
            
            if self.rank >= self.num_replicas or self.rank < 0:
                raise ValueError(f"Invalid rank {self.rank}, rank should be in the interval [0, {self.num_replicas - 1}]")
        else:
            # Non-distributed mode
            self.num_replicas = 1
            self.rank = 0
            
        if len(self.dataset) % self.num_replicas != 0: 
            self.num_samples = math.ceil((len(self.dataset) - self.num_replicas) / self.num_replicas)
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)

        self.total_size = self.num_samples * self.num_replicas
        self.distribut_indices()

    def distribut_indices(self):
        self.epoch += 1
        np.random.seed(self.seed + self.epoch)
        indices = np.random.permutation(self.total_size) if self.shuffle else np.arange(self.total_size)
        
        # Both distributed and non-distributed modes use the same logic,
        # but non-distributed mode has rank=0 and num_replicas=1
        self.indices = deque(indices[self.rank*self.num_samples:(self.rank + 1)*self.num_samples])

    def __len__(self):
        return self.num_samples

    def pop(self):
        if len(self.indices) == 0:
            self.distribut_indices()
            rank_str = f"rank {self.rank} " if self.distributed else ""
            print(f"{rank_str}is out of indices, re-distributing indices (dataset size: {len(self.dataset)}) ({self.dataset.name})")

        return self.indices.popleft()

class DistributedObjectSampler(Sampler):
    def __init__(self, chained_hdf5_dataset: Dataset, shuffle=True, seed=1234, length_multiplier=1):
        # Check if distributed is initialized
        self.distributed = is_dist_initialized()
        
        # Adjust seed based on rank in distributed mode
        if self.distributed:
            seed += dist.get_rank()

        self.cumulative_lengths = chained_hdf5_dataset.cumulative_lengths
        self.dataset_offset    = chained_hdf5_dataset.dataset_offset
        self.length            = chained_hdf5_dataset.lenght
        self.samplers          = [DistributedHDF5DatasetSampler(dataset, shuffle, seed) for dataset in chained_hdf5_dataset.datasets]
        self.shuffle           = shuffle
        self.epoch             = 0
        self.seed              = seed + 12121212
        self.length_multiplier = length_multiplier

    def __iter__(self):
        indices = []
        for _ in tqdm(range(self.length_multiplier)):
            self.epoch += 1
            np.random.seed(self.seed + self.epoch)
            pseudo_indices = list(np.random.permutation(self.length) if self.shuffle else np.arange(self.length))

            for i in pseudo_indices:
                dataset_index = np.argmax(self.cumulative_lengths > i)
                sample_index  = self.samplers[dataset_index].pop()
                indices.append(sample_index + self.dataset_offset * dataset_index)

                if self.dataset_offset <= sample_index:
                    raise ValueError(f"dataset_offset ({self.dataset_offset}) <= sample_index ({sample_index})")

        return iter(indices)

    def __len__(self) -> int:
        return self.length * self.length_multiplier

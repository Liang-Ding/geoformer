import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import os
import h5py

database_name = "patches"

def create_dataloader(data_path, batch_size, num_workers, distributed=False, rank=0, world_size=1):
    r"""Create data loader"""
    hdf5_file_paths = []
    for dirpath, _, filenames in os.walk(data_path):
        for filename in filenames:
            if filename.lower().endswith(('.hdf5', '.h5')):
                hdf5_file_paths.append(os.path.join(dirpath, filename))

    dataset = Scraper_h5(hdf5_file_paths)
    # Create samplers for multiple GPUs (distributed=True) or single GPU (distributed=False)
    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        shuffle = False  # Sampler handles shuffling
    else:
        sampler = None
        shuffle = False  # or True if you prefer shuffling on single-GPU

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        sampler=sampler,
        pin_memory=True,
    )
    return dataloader


class Scraper_h5(Dataset):
    def __init__(self, hdf5_file_paths):
        self.hdf5_file_paths = hdf5_file_paths
        self.file_handles = [None] * len(hdf5_file_paths)  # Lazy load files
        self.lengths = []
        self.total_length = 0

        for path in self.hdf5_file_paths:
            with h5py.File(path, 'r') as f:
                self.lengths.append(f[database_name].shape[0])
                self.total_length += f[database_name].shape[0]

    def __len__(self):
        return self.total_length

    def get_file_and_index(self, idx):
        # Identify which file the index belongs to and the local index within that file
        for i, length in enumerate(self.lengths):
            if idx < length:
                return i, idx
            idx -= length
        raise IndexError('Index out of range')

    def __getitem__(self, idx):
        file_idx, local_idx = self.get_file_and_index(idx)
        # Open the HDF5 file if it hasn't been opened
        if self.file_handles[file_idx] is None:
            self.file_handles[file_idx] = h5py.File(self.hdf5_file_paths[file_idx], 'r')

        one_sample = self.file_handles[file_idx][database_name][local_idx]
        _label = one_sample[0].astype(np.int32)

        '''Lithology Groups'''
        # create labels from CGMC (Integers (1-35) represent lithologies)
        # Group 1: 1
        # Group 2: 2
        # Group 3: 3, background)
        num_category = 3
        mask_group1 = np.isin(_label, [1, 2, 3, 4, 5])
        mask_group2 = np.isin(_label, [17, 19, 33, 34, 35])
        _label = np.where(mask_group1, 1, np.where(mask_group2, 2, 0))

        _label = torch.tensor(_label, dtype=torch.int).long()
        label = F.one_hot(_label, num_classes=num_category)
        label = label.permute(2, 0, 1)  # (C, H, W)
        label = label.clone().detach().to(torch.float32).contiguous()

        features = torch.tensor(one_sample[1:, :, :], dtype=torch.float32).contiguous()

        return features, label

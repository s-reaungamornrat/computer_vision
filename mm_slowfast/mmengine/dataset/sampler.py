from __future__ import annotations
from typing import Optional, Iterator, Sized

import math
import itertools

import torch
from torch.utils.data import Sampler

import numpy as np

class DefaultSampler(Sampler):
    """The default data sampler for both distributed and non-distributed environment

    The differences from Pytoch DistributedSampler are as below:
    1. This sampler supports non-distributed environment
    2. The round up behaviors are a little different. 
        - If `round_up=True`, this sampler will add extra samples to make the number of samples evenly divisible by the world size. And this behavior
            is the same as the `DistributedSampler` with `drop_last=False`
        - IF `round_up=False`, this sampler won't remove or add any samples while the `DistributedSampler` with `drop_last=True` will remove tail samples
    Args:
        dataset (Sized): The dataset
        shuffle (bool): Whether to shuffle the data. Default to True
        seed (int, optional): Random seed used to shuffle the sampler if `shuffle=True`. This number should be identical across all processes in the 
            distributed group. Default to None
        round_up (bool): Whether to add extra samples to make the number of samples evenly divisible by the world size. Default to True.
    """
    def __init__(self, dataset:Sized, shuffle:bool=True, seed:Optional[int]=None, round_up:bool=True)->None:
        self.rank=0
        self.world_size=1
        self.dataset=dataset
        self.shuffle=shuffle
        if seed is None: seed= np.random.randint(2**31)
        self.seed=seed
        self.epoch=0
        self.round_up=round_up
        if self.round_up: # every comnpute gets the same number of samples
            # num_samples: number of data samples assigned to the current compute (rank)
            self.num_samples=math.ceil(len(self.dataset)/self.world_size) # ensure every compute has the same number of iterations
            # total_size: the perceived total size of the data across all computes
            self.total_size=self.num_samples*self.world_size
        else:
            self.num_samples=math.ceil((len(self.dataset)-rank)/self.world_size)
            self.total_size=len(self.dataset)

    def __iter__(self)->Iterator[int]:
        """Iterate the indices"""
        # deterministicallt shuffle based on epoch and seed
        if self.shuffle:
            g=torch.Generator()
            g.manual_seed(self.seed+self.epoch)
            indices=torch.randperm(len(self.dataset), generator=g).tolist()
        else: indices=torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        if self.round_up:
            # int(self.total_size/len(indices) +1) the amount of repeat to perform, with minimum of 1
            # indices* int(self.total_size/len(indices) +1) repeat indices for `int(self.total_size/len(indices) +1)` times
            # ...[:self.total_size] reduce the repeated list down to the exact padded size
            indices=(indices* int(self.total_size/len(indices) +1))[:self.total_size]
        # subsample
        indices=indices[self.rank:self.total_size:self.world_size]

        return iter(indices)

    def __len__(self)->int:
        """The number of samples in this rank"""
        return self.num_samples

    def set_epoch(self, epoch:int)->None:
        """Set the epoch for this sampler

        When `shuffle=True`, this ensures all replicas use a different random ordering for each epoch. Otherwise, the next iteration of this sampler will 
        yeild the same ordering
        
        Args:
            epoch (int): Epoch number
        """
        self.epoch=epoch
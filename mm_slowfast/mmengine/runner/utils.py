from __future__ import annotations
from typing import Optional

import random

import torch
import numpy as np

from computer_vision.slowfast.mmengine.utils.misc import is_list_of

def calc_dynamic_intervals(start_interval:int, dynamic_interval_list:Optional[list[tuple[int,int]]]=None)->tuple[list[int], list[int]]:
    """Calculate dynamic intervals
    Args:
        start_interval (int): The interval used in the beginning. 
        dynamic_interval_list (list[tuple[int,int]], optional): The first element in the tuple is a milestone and the second element is an interval.
            The interval is used after the corresponding milestone. Default to None.
    Returns:
        (tuple[list[int], list[int]]): List of milestone and its corresponding intervals
    """
    if dynamic_interval_list is None: return [0], [start_interval]
    assert is_list_of(dynamic_interval_list, tuple)

    dynamic_milestones=[0]
    dynamic_milestones.extend([dynamic_interval[0] for dynamic_interval in dynamic_interval_list])

    dynamic_intervals=[start_interval]
    dynamic_intervals.extend([dynamic_interval[1] for dynamic_interval in dynamic_interval_list])

    return dynamic_milestones, dynamic_intervals

def set_random_seed(seed:Optional[int]=1, deterministic:bool=False)->int:
    """Set random seed
    Args:
        seed (int,optional): Seed to be used
        deterministic (bool): Whether to set the deterministic option for cudnn backend, i.e., 
            set `torch.backends.cudnn.deterministic` to True and `torch.backends.cudnn.benchmark` to False.
            Defaults to False. See https://pytorch.org/docs/stable/notes/randomness.html for more detail.\
        diff_rank_seed (bool): Whether to add rank number to the random seed to have different random seed in different threads. Default to False\
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if deterministic:
        if torch.backends.cudnn.benchmark:
            print('torch.backends.cudnn.benchmark is going to be set as `False` to cause cuDNN to deterministically choose an algorithm')
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False
        torch.use_deterministic_algorithms(True)
    return seed
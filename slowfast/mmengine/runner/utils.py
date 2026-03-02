from __future__ import annotations
from typing import Optional

import random

import torch
import numpy as np

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
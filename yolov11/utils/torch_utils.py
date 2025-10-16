import os
import random
import warnings

import torch
import numpy as np


def init_seeds(seed=0, deterministic=False):
    """
    Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html.
    Args:
        seed (int, optinal): Random seed.
        deterministic (bool, optional): Whether to set deterministic algorithms
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        if int(torch.__version__[0])>=2:
            torch.use_deterministic_algorithms(True, warn_only=True) # warn if deterministic is not possible
            torch.backends.cudnn.deterministic=True
            os.environ['CUBLAS_WORKSPACE_CONFIG']=":4096:8"
            os.environ["PYTHONHASHSEED"]=str(seed)
        else: warnings.warn(f'Upgrade to torch>=2.0.0 for deterministic training')
    else:
        # Unset all the configurations applied for deterministic training
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic=False
        os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
        os.environ.pop('PYTHONHASHSEED', None)
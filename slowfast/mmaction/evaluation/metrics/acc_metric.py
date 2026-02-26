from __future__ import annotations
from typing import Optional, Union, Sequence

import torch
import numpy as np

def to_tensor(value):
    """Convert value to torch.Tensor"""
    if isinstance(value, np.ndarray): value=torch.from_numpy(value)
    elif isinstance(value, Sequence) and not isinstance(value, str): value=torch.tensor(value)
    assert isinstance(value, torch.Tensor), f"{type(value)} is not an available argument"
    return value
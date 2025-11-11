from __future__ import annotations

import math
import torch

def make_divisible(x:int, divisor:int|torch.Tensor):
    """
    Return the nearest number that is divisible by the given divisor
    Args:
        x (int): The number to make divisible
        divisor (int|torch.Tensor): The divisor
    Returns:
        (int): The nearest number divisible by the divisor
    """
    if isinstance(divisor, torch.Tensor): divisor=int(divisor.max()) # to int
    return math.ceil(x/divisor)*divisor


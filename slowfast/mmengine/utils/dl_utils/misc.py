from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm

def has_batch_norm(model:nn.Module)->bool:
    """Detect whether model has a BatchNormalization layer
    Args:
        model (nn.Module): Learning model
    Returns:
        (bool): Whether model has a Batchnorm layer
    """
    if isinstance(model, _BatchNorm): return True

    for m in model.children():
        if has_batch_norm(m): return True
    return False
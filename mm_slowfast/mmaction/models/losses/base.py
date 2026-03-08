from __future__ import annotations
from abc import ABCMeta, abstractmethod

import torch.nn as nn

class BaseWeightedLoss(nn.Module, metaclass=ABCMeta):
    """Base class for loss

    All subclass should overwrite ``forward()`` method which returns the normal loss without loss weights

    Args:
        loss_weight (float): Factor scalar multiplied on the loss. Default to 1.
    Reference: https://github.com/open-mmlab/mmaction2/blob/main/mmaction/models/losses/base.py
    """
    def __init__(self, loss_weight=1.):
        super().__init__()
        self.loss_weight=loss_weight
        
    @abstractmethod
    def _forward(self, *args, **kwargs):
        """Forward function"""
        pass

    def forward(self, *args, **kwargs):
        """
        Returns:
            (torch.Tensor): Calculated loss
        """
        ret=self._forward(*args, **kwargs)

        if isinstance(ret, dict):
            for k in ret:
                if 'loss' in k: ret[k]*=self.loss_weight
        else: ret*=self.loss_weight
            
        return ret
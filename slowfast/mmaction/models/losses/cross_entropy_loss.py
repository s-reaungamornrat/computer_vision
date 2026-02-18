from __future__ import annotations

from typing import Optional

import numpy as np

import torch
import torch.nn.functional as F

from .base import BaseWeightedLoss

class CrossEntropyLoss(BaseWeightedLoss):
    """Cross entropy loss

    Support two kinds of labels and their corresponding loss type. It's worth mentioning that loss type will be detected by the shape of `cls_score`
    and `label`
    1) Hard label: This label is an integer array and all of the elements are in the range [0, num_classes-1]. This label's shape should be `cls_score`'s
       shape with the `num_classes` dimension removed
    2) Soft label (probability distribution over classes): This label is a probability dostribution and all of the elements are in the range [0,1]. The 
       label's shape must be the sa,e as `cls_score`. For now, only 2-dim soft label is supported.

    Args:
        loss_weight (float): Factor scalar multipled on the loss. Default to 1.
        class_weight (list[float]|None): Loss weight for each class. If set as None, use the same weight for all classes. Only applies to CrossEntropyLoss.
            Default to None.
    Reference: https://github.com/open-mmlab/mmaction2/blob/main/mmaction/models/losses/cross_entropy_loss.py
    """
    def __init__(self, loss_weight:float=1.0, class_weight:Optional[list[float]]=None)->None:
        
        super().__init__(loss_weight=loss_weight)
        self.class_weight=None
        if class_weight is not None: self.class_weight=torch.Tensor(class_weight)

    def _forward(self,cls_score:torch.Tensor, label:torch.Tensor, **kwargs)->torch.Tensor:
        """Forward function
        Args:
            cls_score (torch.Tensor): The class score
            label (torch.Tensor): The ground truth label
            kwargs (keyword arguments): Any keyword arguments to be used in calculating CrossEntropyLoss
        Returns:
            (torch.Tensor): Cross-entropy loss
        """
        if cls_score.size()==label.size():
            #calculate loss for soft label
            assert cls_score.dim()==2, 'Only support 2-dim soft label'
            assert len(kwargs)==0, (f'For now, no extra args are supoorted for soft label, but got {kwargs}')

            lsm=F.log_softmax(cls_score,1)
            if self.class_weight is not None:
                self.class_weight=self.class_weight.to(cls_score.device)
                lsm=lsm*self.class_weight.unsqueeze(0)
            loss_cls=-(label*lsm).sum(1)

            # default reduction 'mean'
            if self.class_weight is not None:
                # Use weighted average as pytorch CrossEntropyLoss does
                # For more information, please visit https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html # noqa
                loss_cls=loss_cls.sum()/torch.sum(self.class_weight.unsqueeze(0) * label)
            else: loss_cls=loss_cls.mean()
                
        else: # calculate loss for hard label
            if self.class_weight is not None:
                assert 'weight' not in kwargs, 'The key "weight" already exists'
                kwargs['weight']=self.class_weight.to(cls_score.device)
            loss_cls=F.cross_entropy(cls_score, label, **kwargs)
        return loss_cls


if __name__ == __main__

    loss=CrossEntropyLoss(loss_weight=1.)
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.empty(3, dtype=torch.long).random_(5)
    output = loss(input, target)
output.backward()
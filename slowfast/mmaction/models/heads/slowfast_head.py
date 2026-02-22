from __future__ import annotations

from typing import Optional, Callable

import torch
import torch.nn as nn

from .base import BaseHead
from computer_vision.slowfast.mmaction.models.losses.cross_entropy_loss import CrossEntropyLoss
from computer_vision.slowfast.mmengine.model.weight_init import normal_init

class SlowFastHead(BaseHead):
    """The classification head for SlowFast
    Args:
        num_classes (int): Number of classes to classified
        in_channels (int): Number of channels of input features
        loss_cls (Callable): Loss
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'
        dropout_ratio (float): Probability of dropout layer. Default: 0.8
        init_std (float): Std value for Initiation. Default: 0.01
        kwargs (dict): Any keyword argument to be used to initialize the head
    """
    def __init__(self, num_classes:int, in_channels:int, loss_cls:Callable=CrossEntropyLoss(loss_weight=1.), spatial_type:str='avg',
                 dropout_ratio:float=0.8, init_std:float=0.01, **kwargs)->None:
        
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        
        self.spatial_type=spatial_type
        self.dropout_ratio=dropout_ratio
        self.init_std=init_std

        if self.dropout_ratio>1e-9: self.dropout=nn.Dropout(p=self.dropout_ratio)
        else: self.dropout=None
        self.fc_cls=nn.Linear(in_channels, num_classes)

        if self.spatial_type=='avg': self.avg_pool=nn.AdaptiveAvgPool3d((1,1,1))
        else: self.avg_pool=None
            
        assert self.avg_pool is not None, 'self.avg_pool must not be None'

    def init_weight(self)->None:
        """Initialize the parameters from scratch"""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x:tuple[torch.Tensor],**kwargs)->torch.Tensor:
        """
        Args:
            x (tuple[torch.Tensor]): Tuple of slow and fast features with shape (N, channel_slow, T1, H, W) and (N, channel_fast, T2, H, W),
                respectively, where N is the batch size, channel_slow is the number of channels from slow path, channel_fast is the 
                number of channels from fast path (typically channel_fast < channel_slow). T1 is the temporal dimension of slow path
                and T2 is the temporal dimension of fast path (typically T1<T2). H and W are height and width which are equal from both paths
        Returns:
            (torch.Tensor): Classification scores with shape (N,num_classes) where N is the batch size
        """
        # (N, channel_slow, T1, H, W), (N, channel_fast, T2, H, W)
        x_slow, x_fast=x
        # (N, channel_slow, 1, 1, 1), (N, channel_fast, 1, 1, 1)
        x_slow=self.avg_pool(x_slow)
        x_fast=self.avg_pool(x_fast)
        # (N, channel_fast+channel_slow, 1, 1, 1)
        x=torch.cat([x_fast, x_slow], dim=1)

        if self.dropout is not None: x=self.dropout(x)

        # (N,C)
        x=x.view(x.size(0),-1)
        # (N, num_classes)
        cls_score=self.fc_cls(x)
        return cls_score


if __name__ == "__main__":
    
    channel_slow=128
    channel_fast=64
    slow_fast_head=SlowFastHead(num_classes=101, in_channels=channel_slow+channel_fast, loss_cls=CrossEntropyLoss(loss_weight=1.), 
                                spatial_type='avg', dropout_ratio=0.8, init_std=0.01, multi_class=True)
    x=[torch.rand(3,c,4,6,6) if i==0 else  torch.rand(3,c,32,6,6) for i, c in enumerate([channel_slow,channel_fast])]
    print('x ', [i.shape for i in x])
    out=slow_fast_head(x)
    print('out ', out.shape)
    nn.MSELoss()(out, torch.rand_like(out)).backward()
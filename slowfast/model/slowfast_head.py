from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Optional, Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

class SlowFastHead(nn.Module):
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

    def __init__(self, num_classes:int, in_channels:int, spatial_type:str='avg', dropout_ratio:float=0.8, init_std:float=0.01, multi_class:bool=False,
                label_smooth_eps:float=0., topk:Union[int, tuple[int]]=(1,5), average_clips:Optional[dict]=None, init_cfg:Optional[dict]=None)->None:
        
        super(SlowFastHead, self).__init__()
        
        self.num_classes=num_classes
        self.in_channels=in_channels
        self.multi_class=multi_class
        self.label_smooth_eps=label_smooth_eps
        self.average_clips=average_clips
        assert isinstance(topk, (int, tuple)), f"topk must be int or tuple[int] but got {type(topk)}"
        if isinstance(topk, int): topk=(topk,)
        assert all(k>0 for k in topk), f"Top-k must be > 0, but got {topk}"
        self.topk=topk

        
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

    def average_clip(self, cls_scores:torch.Tensor, num_segs:int=1)->torch.Tensor:
        """Averaging class scores over multiple clips

        Using different averaging types ('scores' or 'prob' or None, which defined in test_cfg) to computed the final averaged class score. Only 
        called in test mode

        Args:
            cls_scores (torch.Tensor): Class scores to be averaged, with shape (B*num_segs, num_classes)
            num_segs (int): Number of clips for each input sample
        Returns:
            (torch.Tensor): Averaged class scores
        """
        assert self.average_clips in ['score', 'prob', None], (f"{self.average_clips} is not supoorted. "
                                                               f"Currently supported ones are ['score', 'prob', None]")
        batch_size=cls_scores.shape[0]
        cls_scores=cls_scores.view((batch_size//num_segs, num_segs)+cls_scores.shape[1:])
        if self.average_clips is None: return cls_scores
        elif self.average_clips=='prob': cls_scores=F.softmax(cls_scores, dim=-1).mean(dim=1) # average along num_clips dimension
        elif self.average_clips=='score': cls_scores=cls_scores.mean(dim=1) # average along num_clips dimension
        return cls_scores


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
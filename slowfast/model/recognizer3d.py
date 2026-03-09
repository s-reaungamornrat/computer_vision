from __future__ import annotations
from typing import Iterable, Optional, Union
from collections import OrderedDict, defaultdict

import copy
import inspect
import warnings

import torch.nn as nn

from .slowfast_head import SlowFastHead
from .resnet3d_slowfast import ResNet3dSlowFast


class Recognizer3D(nn.Module):
    """Recognizers
    Args:
        backbone (dict): Backbone modules to extract feature
        cls_head (dict): Classification head to process feature. 
        train_cfg (dict, optional): Config for training. Default to None
        test_cfg (dict, optional): Config for testing. Default to None
        data_preprocessor (dict, optional): The pre-process config of class `ActionDataPreprocessor`, usually
            including ``mean``, ``std``, and ``format_shape``. Default to None.
        init_cfg (dict | list[dict], optional): Initialization config dict
    References:
        https://github.com/open-mmlab/mmaction2/blob/main/mmaction/models/recognizers/recognizer3d.py
        https://github.com/open-mmlab/mmaction2/blob/main/mmaction/models/recognizers/base.py
        https://github.com/open-mmlab/mmengine/blob/main/mmengine/model/base_model/base_model.py
    """
    def __init__(self, backbone:dict, cls_head:dict|None=None)->None:

        super().__init__()
        
        self._is_init = False

        # backbone
        backbone_=backbone
        if 'type' in backbone:
            backbone_=copy.deepcopy(backbone)
            backbone_.pop('type')
        self.backbone=ResNet3dSlowFast(**backbone_)
        
        # head
        cls_head_=cls_head
        if 'type' in cls_head:
            cls_head_=copy.deepcopy(cls_head)
            cls_head_.pop('type')
        self.cls_head=SlowFastHead(**cls_head_)
        
        
    def forward(self, inputs:torch.Tensor)->torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): The input tensor with shape (N, num_crops, C, T, H, W) or (N, C, T, H, W) 
            mode (str): Return what kind of value. Default to 'tensor'
        Returns:
            (torch.Tensor): Classification scores with shape (N,num_classes) where N is the batch size
        """
        num_crops=1 # num_crops
        if inputs.ndim==6: num_crops=inputs.shape[1] # num_crops
        # (N,C,T,H,W)  or (N, num_crops, C, T, H, W) -> (N*num_crops, C, T, H, W)
        # num_crops is calculated by 
        # 1) `twice_sample` in `SampleFrames`
        # 2) `num_sample_positions` in `DenseSameFrames`
        # 3) `ThreeCrop/TenCrop` in `test_pipeline`
        # 4) `num_clips` in `SampleFrames` or its subclass if `clip_len!=1`
        inputs=inputs.view((-1,)+inputs.shape[-4:])
        
        # Return features extracted through backbone
        x=self.backbone(inputs)
        return self.cls_head(x)
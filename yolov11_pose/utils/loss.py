from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from computer_vision.yolov11_pose.utils.tal import TaskAlignedAssigner

class DFLoss(nn.Module):
    """Criterion class for computing Distribution Focal Loss (DFL)"""

    def __init__(self, reg_max:int=16)->None:
        """Initialize the DFL module with regularization maximum, i.e., number of bin to estimate to
        Args:
            reg_max (int): Number of bins governing the estimated value ranges
        """
        super().__init__()
        self.reg_max=reg_max

class BboxLoss(nn.Module):
    """Criterion class for computing training losses for bounding boxes"""
    
    def __init__(self, reg_max:int=16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings.
        Args:
            reg_max (int): Number of bins governing the estimated value ranges, i.e., total number of bins for each parameter
        """
        super().__init__()
        self.dfl_loss=DFLoss(reg_max) if reg_max>1 else None

class v8DetectionLoss:
    """Criterion class for computing training losses for YOLOv8 object detection"""

    def __init__(self, model, tal_topk:int=10): # model must be de-paralleled
        """Initialize v8DetectionLoss with model parameters and task-aligned assignment settings"""
        device=next(model.parameters()).device # get model device
        h=model.args
        m=model.model[-1] # Detect() module
        self.bce=nn.BCEWithLogitsLoss(reduction='none')
        self.hyp=h
        self.stride=m.stride # 1D tensor
        self.nc=m.nc
        self.no=m.nc+m.reg_max*4 # number of output dimension is number of classes + number of bins for each parameters (4 parameters of bboxes)
        self.reg_max=m.reg_max
        self.device=device

        self.use_dfl=m.reg_max>1
        self.assigner=TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.)
        self.bbox_loss=BboxLoss(m.reg_max).to(device)
        self.proj=torch.arange(m.reg_max, dtype=torch.float, device=device) # bin values from 0 to reg_max
        


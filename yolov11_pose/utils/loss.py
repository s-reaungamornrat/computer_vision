from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from computer_vision.yolov11_pose.utils.metrics import OKS_SIGMA, bbox_iou
from computer_vision.yolov11_pose.utils.tal import TaskAlignedAssigner, dist2bbox, bbox2dist
from computer_vision.yolov11_pose.utils.ops import xywh2xyxy

class DFLoss(nn.Module):
    """Criterion class for computing Distribution Focal Loss (DFL)"""

    def __init__(self, reg_max:int=16)->None:
        """Initialize the DFL module with regularization maximum, i.e., number of bin to estimate to
        Args:
            reg_max (int): Number of bins governing the estimated value ranges
        """
        super().__init__()
        self.reg_max=reg_max
        
    def __call__(self, pred_dist:torch.Tensor, target:torch.Tensor)->torch.Tensor:
        """Return sum of left and right DFL losses from https://ieeexplore.ieee.org/document/9792391
        Args:
            pred_dist (torch.Tensor): Predicted distance of left-top and right-bottom from the anchor centers with shape (4*M,reg_max) where
                4 is for left,top,right,bottom, M is the number of boxes, and reg_max is the number of distance bins
            target (torch.Tensor): Target distance left-top and right-bottom from the anchor centers with shape (M, 4)
        Returns:
            (torch.Tensor): DFL loss with shape (M,1)
        """
        target=target.clamp_(0, self.reg_max-1-0.01) # clamp between 0 and maximum distance
        tl=target.long() # (M, 4) target left, i.e., round down to int
        tr=tl+1 # (M, 4) target right
        wl=tr-target# (M, 4) weight left: distance between target and target-right
        wr=1-wl # (M, 4) weight right
        # tl.view(-1) convert (M, 4) to (M*4,)
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction='none').view(tl.shape)*wl+ # (M*4)->(M,4)
            F.cross_entropy(pred_dist, tr.view(-1), reduction='none').view(tl.shape)*wr # (M*4)->(M,4)
        ).mean(-1, keepdim=True)  # (M,4)->(M,1)

class BboxLoss(nn.Module):
    """Criterion class for computing training losses for bounding boxes"""
    
    def __init__(self, reg_max:int=16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings.
        Args:
            reg_max (int): Number of bins governing the estimated value ranges, i.e., total number of bins for each parameter
        """
        super().__init__()
        self.dfl_loss=DFLoss(reg_max) if reg_max>1 else None

class KeypointLoss(nn.Module):
    """Criterion class for computing keypoint losses"""
    def __init__(self, sigmas:torch.Tensor)->None:
        """Initialize the KeypointLoss class with keypoint sigmas"""
        super().__init__()
        self.sigmas=sigmas
        

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

    def preprocess(self, targets:torch.Tensor, batch_size:int, scale_tensor:torch.Tensor)->torch.Tensor:
        """Preprocess targets by converting to tensor format and scaling coordinates
        Args:
            targets (torch.Tensor): LxE target tensor where L is the number of labels and E is the target dimension, typically
                6 comprising Lx1 image_index, Lx1 class index, and Lx4 bounding box coordinates
            batch_size (int): Batch size
            scale_tensor (torch.Tensor): Image size tensor comprising (4,) elements of width, height, width, height
        Returns:
            (torch.Tensor): BxMx(E-1) tensor where B is batch size, M is the maximum number of labels each image in this batch has, E-1 is the
                target dimensions after removing image-index dimension so containing cls, xyxy bounding box coordinates in pixel units
        """
        nl, ne=targets.shape # number of labels, number of elements
        if nl==0: out=torch.zeros(batch_size, 0, ne-1, device=self.device)
        else: 
            i=targets[:,0] # image index
            _, counts=i.unique(return_counts=True) # number of labels in each image (number of each unique image index)
            counts=counts.to(dtype=torch.int32)
            out=torch.zeros(batch_size, counts.max(), ne-1, device=self.device)
            for j in range(batch_size):
                matches=i==j
                if n:=matches.sum(): out[j,:n]=targets[matches, 1:]
            out[..., 1:5]=xywh2xyxy(out[...,1:5]).mul_(scale_tensor)
        return out

    def bbox_decode(self, anchor_points:torch.Tensor, pred_dist:torch.Tensor)->torch.Tensor:
        """Decode predicted object bounding box coordinates from anchor points and distribution.
        Args:
            anchor_points (torch.Tensor): Nx2 anchor positions where N is the sum of H*W from all levels or the number of anchors 
                and 2 for x and y in feature units
            pred_dist (torch.Tensor): BxNx(4*reg_max) predicted DFL distribution for each bounding coordinates in feature units
        Returns:
            (torch.Tensor): BxNx4 bounding box coordinates in feature units
        """
        if self.use_dfl:
            b, a, c =pred_dist.shape # batch, anchors=N, channels=4*reg_max
            # BxNx4xreg_max reg_max = BxNx4 distribution distance from each anchor points
            pred_dist=pred_dist.view(b, a, 4, c//4).softmax(dim=3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)
    
class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 pose estimation"""
    def __init__(self, model): # model must be de-paralleled
        """Initialize v8PoseLoss with model parameters and key-point-specific loss functions"""
        super().__init__(model)
        self.kpt_shape=model.model[-1].kpt_shape
        self.bce_pose=nn.BCEWithLogitsLoss()
        is_pose=self.kpt_shape==[17, 3]
        nkpt=self.kpt_shape[0] # number of keypoints
        sigmas=torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device)/nkpt
        self.keypoint_loss=KeypointLoss(sigmas=sigmas)

    @staticmethod
    def kpts_decode(anchor_points:torch.Tensor, pred_kpts:torch.Tensor)->torch.Tensor:
        """Decode predicted keypoints to feature map grid coordinates
        Args:
            anchor_points (torch.Tensor): Nx2 anchor positions where N is the number of anchors or the sum of all H*W from all levels
            pred_kpts (torch.Tensor): BxNxQxd predicted keypoint offsets where Q is the number of keypoints per instance and d is keypoint dimension
                e.g., BxNx17x3
        Returns:
            (torch.Tensor): BxNxQxd predicted keypoint locations in feature-grid coordinates
        """
        y=pred_kpts.clone()
        # Expand normalized offsets to double allowing a cell center/anchor point to reach out and predict a keypoint that might be in a neighboring 
        # cell
        y[...,:2]*=2. # BxNxQx2 
        # Turn the predicted normalized offsets to feature-grid coordinates and convert cell-center to cell top-left reference by subtracting 0.5
        y[...,0]+=anchor_points[:,[0]]-0.5 # y[...,0] is of size BxNxQ and anchor_points[:,[0]] is of size Nx1
        y[...,1]+=anchor_points[:,[1]]-0.5
        return y
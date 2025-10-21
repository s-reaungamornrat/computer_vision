from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .tal import make_anchors, dist2bbox, TaskAlignedAssigner
from .ops import xywh2xyxy

class DFLoss(nn.Module):
    """
    Criterion class for computing Distributed Focal Loss (DFL)
    """
    def __init__(self, reg_max:int=16)->None:
        """
        Initialize the DFL module with regularization maximum
        """
        super().__init__()
        self.reg_max=reg_max
        
class BboxLoss(nn.Module):
    """
    Criterion class for computing training losses for bounding boxes
    """
    def __init__(self, reg_max:int=16):
        """
        Initialize the BboxLoss module with regularization maximum and DFL settings
        """
        super().__init__()
        self.dfl_loss=DFLoss(reg_max) if reg_max>1 else None

class v8DetectionLoss:
    """
    Criterion class for computing taining losses for YOLOv8 object detection

    DFL convert regression into classification of offset bins, i.e., offsets are divided into bins. 
    DFL estimates probabality distribution of these bins and the final offset is the weighted sum of the bin offsets by the probabilty 
    distribution, i.e., expectation/expected offset. For example, YOLOv11 devides offsets into 16 bins and the model estimates logits 
    which then are turned in probability (through softmax) of each offset from [0,1,2,...15]. 
    """
    def __init__(self, args, model, tal_topk:int=10): # model must be de-paralleled
        """
        Initialize v8DetectionLoss with model parameters and task-aligned assignment settings
        """
        device=next(model.parameters()).device # get model device
        m=model.model[-1] # Detect() module
        self.hyp=args
        self.bce=nn.BCEWithLogitsLoss(reduction='none')
        self.stride=m.stride # model strides
        self.nc=m.nc # number of classes
        self.no = m.nc+m.reg_max*4 
        self.reg_max=m.reg_max
        self.device=device

        self.use_dfl=m.reg_max>1

        self.assigner=TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss=BboxLoss(m.reg_max).to(device)
        self.proj=torch.arange(m.reg_max, dtype=torch.float, device=device)
        
    def preprocess(self, targets:torch.Tensor, batch_size:int, scale_tensor:torch.Tensor)->torch.Tensor:
        """
        Preprocess targets by converting to tensor format and scaling coordinates. Given targets of size (N, 6) where 6 include
        batch_idx (image index), cls, bounding-boxes, convert it to BxMx5 where B is the batch size, M is the maximum number of boxes/labels 
        per image in the batch, and 5 is for cls and bounding-boxes. For the batch idex that the number of boxes < the maximum 
        number of boxes/labels, the output targets are padded with 0. 
        Args:
            targets (torch.Tensor): (N,6) labels where 6 is for batch_idx (image index), cls, and bounding boxes in the normalized xywh format
            batch_size (int): The number of items in the batch
            scale_tensor (torch.Tensor): The tensor of width, height, width, height of input images to denormalize bounding box coordinates
                of size (4,)
        Returns:
            (torch.Tensor): BxMx5 labels where B is the batch_size, M is the maximum number of labels in each image, 5 is for cls and 
                bounding boxes in the xyxy format in pixel units of input image size
        """
        nl,ne=targets.shape
        if nl==0: return torch.zeros((batch_size, 0, ne-1), device=self.device)
        i=targets[:,0] # image index
        _, counts=i.unique(return_counts=True)
        counts=counts.to(dtype=torch.int32)
        out=torch.zeros(batch_size, counts.max(), ne-1, device=self.device)
        for j in range(batch_size):
            matches=i==j
            n=matches.sum()
            if n: out[j, :n]=targets[matches, 1:]
        out[...,1:5]=xywh2xyxy(out[...,1:5].mul_(scale_tensor))
        return out
        
    def bbox_decode(self,anchor_points:torch.Tensor, pred_dist:torch.Tensor)->torch.Tensor:
        """
        Decode predicted object bounding box coordinates from anchor points and distribution. This function
        decodes discritized bounding box predictions (DFL) into continuous coordinates

        DFL convert regression into classification of offset bins, i.e., offsets are divided into bins. DFL estimates probabality 
        distribution of these bins and the final offset is the weighted sum of the bin offsets by the probabilty distribution, 
        i.e., expectation/expected offset. For example, YOLOv11 devides offsets into 16 bins and the model estimates logits which 
        then are turned in probability (through softmax) of each offset from [0,1,2,...15]. 
        Args:
            anchor_points (torch.Tensor): mHWx2 where mHW is the number of feature grid/pixels across all scale, representing 
                the number of anchor points, and 2 is for x, y representing centers of boxes in feature-grid units
            pred_dist (torch.Tensor): BxmHWx4 where B is the batch size, mHW is the number of feature grid/pixels across all scale and
                4 is for predicted (left,top) and (right, bottom) coordinates of boxes 
        Returns:
            (torch.Tensor): BxmHWx4 bounding box coordinates in xywh or xyxy format in feature-grid units
        """
        if self.use_dfl:
            # c=4xbins where 4 is for x, y, w, h
            b, a, c=pred_dist.shape # batch, anchors, channels, where anchors=mHW for m scales, and channels is 4reg_max
            # pred_dist.view(b, a, 4, c//4) gets the logits for each dimension in (left, top, right, bottom)--the logits size = number of bins = c//4
            # .softmax(3) converts logits into probability distribution; each of the 4 coordinates has a categorical distribution across possible bins
            # probabilities @ self.proj (projection-vector): computes the expected offset values based on the distribution and the discrete offset
            #    bin location in self.proj. Thus, converting the categorical distribution to a differential continuous value
            #  matmul( B x mHw x 4 x reg_max,  (reg_max) ) -> B x mHw x 4
            pred_dist=pred_dist.view(b, a, 4, c//4).softmax(dim=3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(distance=pred_dist, anchor_points=anchor_points, xywh=False)

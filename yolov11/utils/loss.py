from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import bbox_iou
from .tal import make_anchors, dist2bbox, bbox2dist, TaskAlignedAssigner
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

    def __call__(self, pred_dist:torch.Tensor, target:torch.Tensor)->torch.Tensor:
        """
        Compute the sum of left and right DFL losses from https://ieeexplore.ieee.org/document/9792391.
        Args:
            pred_dist (torch.Tensor): 4Nxbins where 4 is for left, top, right, bottom, and bins is the
                number of offset bins of DFL (e.g., bins=16)
            target (torch.Tensor): Nx4 where 4 is for left, top, right, bottom in feature-grid units
        Returns:
            (torch.Tensor): The sum of left and right cross entropy losses, with shape Nx1
        """    
        # Target values must be between 0 to n_bins-1
        target=target.clamp_(0, self.reg_max-1 - 0.01)
        tl=target.long() # target left
        tr=tl+1 # target right
        # cross weighting: left-loss is weighed by the distance between target-right and target
        # and right loss is weighed by the distance between target and target-left 
        # This is so that the side that closer to the target gets higher weight
        wl=tr-target # weight left
        wr=1-wl # weight right
        #  The same shape modification for both line: (4Nxbins, 4N) -view->(Nx4)
        # (Nx4 + Nx4) -mean->Nx1
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction='none').view(tl.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction='none').view(tl.shape) * wr 
                ).mean(-1, keepdim=True)
        
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
    
    def forward(self, pred_dist:torch.Tensor, pred_bboxes:torch.Tensor, anchor_points:torch.Tensor,target_bboxes:torch.Tensor,
                target_scores:torch.Tensor, target_scores_sum:torch.Tensor, fg_mask:torch.Tensor)->tuple[torch.Tensor, torch.Tensor]:
        """
        Compute IoU and DFL losses for bounding boxes
        Args:
            pred_dist (torch.Tensor): (b, n_anchors, 4*reg_max) where reg_max is the number of discrete offset bins of DFL
                and 4 represents left, top, right, bottom coordinates of boxes 
            pred_bboxes (torch.Tensor): (b, n_anchors, 4) predicted bounding boxes in the xyxy format in feature-grid units
            anchor_points (torch.Tensor): (n_anchors, 2) anchor point location in x and y in feature-grid units
            target_bboxes (torch.Tensor): (b, n_anchors, 4) Ground truth bounding boxes in the xyxy format in feature-grid units
            target_scores (torch.Tensor): (b, n_anchors, nc) Weighted target scores where nc is the number of classes
            target_scores_sum (torch.Tensor): Scalar sum of target_scores or 1 if the sum < 1
            fg_mask (torch.Tensor): (b, n_anchors): Mask of positive anchors
        Returns:
            (torch.Tensor): Scalar IoU loss tensor
            (torch.Tensor): Scalar DFL loss tensor
        """
        # For each positive anchor, find the sum of classification scores that will be used to weight IoU
        #      (b, n_anchors, nc) -sum-> (b, n_anchors) -fg_mask->N -unsqueeze->Nx1
        weight=target_scores.sum(dim=-1)[fg_mask].unsqueeze(-1)
        # pred_bboxes[fg_mask] and target_bboxes[fg_mask] are of size (N, 4)
        iou=bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou=( (1.-iou)*weight ).sum() / target_scores_sum
        
        if self.dfl_loss:
            target_ltrb=bbox2dist(anchor_points=anchor_points, bbox=target_bboxes, reg_max=self.dfl_loss.reg_max-1)
            # (N, 1) = (N, 1) * (N,1)<-[(4N,bins), (N,4)]
            loss_dfl=weight * self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask])
            loss_dfl=loss_dfl.sum()/target_scores_sum
        else: loss_dfl=torch.tensor(0.).to(pred_dist.device)
        
        return loss_iou, loss_dfl
    
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
    
    def __call__(self, preds:tuple|list, batch:dict[str, Any])->tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the sum of the loss for box, cls, and dfl multiplied by batch size
        Args:
            preds (tuple|list): If tuple, the first element must be a list of features from all scales; otherwise, a list
                of BxOxHxW features from each scale where O=4*bins + nc is the number of outputs with `bins` for the 
                number of offset/distance bins for (left,top,right,center from the center) and nc for the number of classes. Typically, bins=16
            batch (dict[str,Any]): Training batch containing
                - `batch_idx` (torch.Tensor): (N,) image index of each item in the batch
                - `bboxes` (torch.Tensor): (N,4) bounding boxes in the normalized xywh format
                - `cls` (torch.Tensor): (N, 1) object classes
                - `im_file` (tuple[str]): filename of images in this batch
                - `img` (torch.Tensor): BxCxHxW where C is the number of image channels
                - `ori_shape` (tuple[tuple[int,int]]): Tuple of tuples of original (height, width) of all images in this batch 
                - `resized_shape` (tuple[tuple[int,int]]): Tuple of tuples of (height, width) of all input images in this batch 
        Returns:
            (torch.Tensor): Differentiable weighted box, classification, and DFL losses, multipled by batch_size, of size (3,)
            (torch.Tensor): Weighted box, classification, and DFL losses of size (3,)
        """
        loss=torch.zeros(3, device=self.device) # box, cls, dfl
        feats=preds[1] if isinstance(preds, tuple) else preds
        
        # list of BxOxHxW to list of BxOx(HW) -cat-> BxOx(mHW) where m is the number of feats
        # then split so one tensor is of size B x (reg_max*4) x (mHW) and the others is of size B x nc x (mHW) where reg_max is the number of discrete 
        # offset/distance bins. pred_distri is logits divided into the `reg_max` bins representing offset/distance-from-center bins for each box coordinate, 
        # including left, top, right, bottom
        pred_distri, pred_scores=torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split((self.reg_max*4, self.nc), dim=1)
        
        pred_scores=pred_scores.permute(0,2,1).contiguous() # from B x nc x (mHW) to B x (mHW) x nc
        pred_distri=pred_distri.permute(0,2,1).contiguous() # from B x (reg_max*4) x (mHW) to B x (mHW) x (reg_max*4); reg_max=number of bin of DFL
        
        dtype=pred_scores.dtype
        batch_size=pred_scores.shape[0]
        
        # multiplying feats[i].shape[2:] with stride[i] yeilds the same imgsz regardless of i
        imgsz=torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype)*self.stride[0] # image size (h, w) with shape (2,)
        anchor_points, stride_tensor=make_anchors(feats, self.stride, 0.5) # anchors are center of boxes in feature-grid units
        
        # Targets with boxes in the normalized xywh format
        targets=torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'], batch['bboxes']), 1) # Nx1, Nx1, Nx4 -> Nx6
        # BxMx5 where M is the maximum number of labels per image, and 5 is for cls, bounding box in xyxy in pixel units
        targets=self.preprocess(targets, batch_size, scale_tensor=imgsz[[1,0,1,0]]) # pad with 0 if number of labels < M
        gt_labels, gt_bboxes=targets.split((1,4), dim=2) # BxMx1 cls, BxMx4 xyxy in pixel units
        mask_gt=gt_bboxes.sum(2, keepdim=True).gt_(0.0) # BxMx1 mask valid bounding box by checking whether xyxy sum > 0
        
        # Pboxes: anchor_points mHWx2 and pred_distri BxmHWx(4reg_max)
        pred_bboxes=self.bbox_decode(anchor_points, pred_distri)  # BxmHWx4 xyxy format in feature-grid units
        
        _, target_bboxes, target_scores, fg_mask, _=self.assigner(pd_scores=pred_scores.detach().sigmoid(),
                                                                  pd_bboxes=(pred_bboxes.detach()*stride_tensor).type(gt_bboxes.dtype), 
                                                                  anc_points=anchor_points*stride_tensor,
                                                                  gt_labels=gt_labels, gt_bboxes=gt_bboxes, mask_gt=mask_gt)
        
        target_scores_sum=max(target_scores.sum(), 1)
        
        # Classification loss
        loss[1]=self.bce(pred_scores,target_scores.to(dtype=dtype)).sum() / target_scores_sum
        
        # Bbox loss
        if fg_mask.sum():
            loss[0], loss[2]=self.bbox_loss(pred_dist=pred_distri, pred_bboxes=pred_bboxes, anchor_points=anchor_points,
                                            target_bboxes=target_bboxes/stride_tensor, target_scores=target_scores, 
                                            target_scores_sum=target_scores_sum, fg_mask=fg_mask)
        loss[0]*=self.hyp.box # box loss
        loss[1]*=self.hyp.cls # cls loss
        loss[2]*=self.hyp.dfl # dfl loss
        
        return  loss*batch_size, loss.detach() # (box, cls, dfl)
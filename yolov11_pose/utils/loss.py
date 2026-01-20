from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from computer_vision.yolov11_pose.utils.metrics import OKS_SIGMA, bbox_iou
from computer_vision.yolov11_pose.utils.tal import TaskAlignedAssigner, dist2bbox, bbox2dist, make_anchors
from computer_vision.yolov11_pose.utils.ops import xywh2xyxy, xyxy2xywh

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

    def forward(self, pred_dist:torch.Tensor, pred_bboxes:torch.Tensor, anchor_points:torch.Tensor,target_bboxes:torch.Tensor, 
                target_scores:torch.Tensor, target_scores_sum:torch.Tensor, fg_mask:torch.Tensor)->tuple[torch.Tensor, torch.Tensor]:
        """Compute IoU and DFL losses for bounding boxes
        Args:
            pred_dist (torch.Tensor): Predicted distances from anchors to the left-top and right-bottom corners of boxes with shape (B,N,reg_max*4) 
                where N is the number of anchors/sum of H*W across all feature levels and reg_max is the number of distance bins
            pred_bboxes (torch.Tensor): Predicted bounding box positions in the xyxy format in feature grid units of shape (B,N,4) where N is the number
                of anchors/sum of H*W across all feature levels
            anchor_points (torch.Tensor): Anchor locations with shape (N,2) in the feature grid unit where 2 is for x and y
            target_bboxes (torch.Tensor): Ground truth bounding boxes with shape (B, N, 4) in the xyxy format in the feature grid unit
            target_scores (torch.Tensor): Soft target scores with shape (B, N, C) where C is the number of classes
            fg_mask (torch.Tensor): Foreground mask with shape (B, N) telling which anchors are associated with ground-truth boxes
        Returns:
            (torch.Tensor): IoU loss
            (torch.Tensor): DFL loss
        """
        # (B,N,C)->(B,N)->(M,)->(M,1) where M is the number of anchors having valid ground-truth boxes
        weight=target_scores.sum(-1)[fg_mask].unsqueeze(-1) 
        # pred_bboxes[fg_mask] from (B,N,4) to (M,4) and the same for target_bboxes
        iou=bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True) # (M,1)
        loss_iou=((1.-iou)*weight).sum()/target_scores_sum  # we note that target_scores_sum should be equal to weight.sum() if weight.sum()>0
        
        # DFL loss
        if self.dfl_loss:
            target_ltrb=bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max-1) # (B, N, 4)
            # pred_dist[fg_mask] from (B,N,reg_max*4) to (M,reg_max*4) and .view(-1, bbox_loss.dfl_loss.reg_max) to (M*4,reg_max)
            # target_ltrb[fg_mask] from (B, N, 4) to (M,4)
            loss_dfl=self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask])*weight # (M,1)
            loss_dfl=loss_dfl.sum()/target_scores_sum
        else: loss_dfl=torch.tensor(0.).to(pred_dist.device)
            
        return loss_iou, loss_dfl
    
class KeypointLoss(nn.Module):
    """Criterion class for computing keypoint losses"""
    def __init__(self, sigmas:torch.Tensor)->None:
        """Initialize the KeypointLoss class with keypoint sigmas"""
        super().__init__()
        self.sigmas=sigmas
        
    def forward(self, pred_kpt:torch.Tensor, gt_kpt:torch.Tensor, kpt_mask:torch.Tensor, area:torch.Tensor, eps:float=1e-9)->torch.Tensor:
        """Calculate keypoint loss factor and Euclidean distance loss for keypoints
        Args:
            pred_kpt (torch.Tensor): Predicted keypoint locations in feature-grid units with shape (M,K,d) where M is the number of instances/objects
                K is the number of keypoints per instance, and d is the dimension
            gt_kpt (torch.Tensor): Ground truth keypoint locations in feature-grid units with shape (M,K,d)
            kpt_mask (torch.Tensor): Mask indicating whether keypoints are visible or not with shape (M,K)
            area (torch.Tensor): Bounding box areas of the instances/objects with shape (M,1)
            eps (float): Small value preventing division by zero
        Returns:
            (torch.Tensor): Keypoint loss
        """
        d=(pred_kpt[...,0]-gt_kpt[...,0]).pow(2)+(pred_kpt[...,1]-gt_kpt[...,1]).pow(2) # (M,K)
        # (number of keypoints per instance)/(number of visible keypoints from all instances)
        # kpt_loss_factor makes the loss represents error per visible keypoints, while giving high weight if objects are occluded or truncated
        # (i.e., having few visible keypoints) so the loss of occluded instances is somewhat equal to the loss of highly visible instances
        kpt_loss_factor=kpt_mask.shape[1]/(torch.sum(kpt_mask!=0, dim=1)+eps)  # (M,)
        # e is of size (M,K) while sigma is of size (K,)
        e=d/((2*self.sigmas).pow(2)*(area+eps)*2) # from cocoeval
        # torch.exp(-e), where e=-d/scale (above), gives maximum of 1 when pred_kpt==gt_kpt
        # 1-torch.exp(-e) gives the loss since loss is 0 if pred_kpt==gt_kpt
        # Note: Using ((1-torch.exp(-e))*kpt_mask) only, an object with few visible keypoints contributes less loss, i.e., objects aren't equally weighted
        return (kpt_loss_factor.view(-1,1)* ((1-torch.exp(-e))*kpt_mask)).mean()
    
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
        
    def calculate_keypoints_loss(self, masks:torch.Tensor, target_gt_idx:torch.Tensor, keypoints:torch.Tensor, batch_idx:torch.Tensor,
                                 stride_tensor:torch.Tensor, target_bboxes:torch.Tensor, pred_kpts:torch.Tensor)->tuple[torch.Tensor, torch.Tensor]:
        """Calculate the keypoints loss for the model.
        
        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is based on the difference
        between the predicted keypoints and ground truth keypoints. The keypoints object loss is a binary classification loss that classifies 
        whether a keypoint is present or not.
        
        Args:
            masks (torch.Tensor): Binary mask tensor indicating, for each anchors, whether ground-truth/object exists with shape (B,N) where
                N is the number of anchors/sum of H*W from all feature levels
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects with shape (B, N), i.e., for each anchor, what 
                is the index of its associated ground truth
            keypoints (torch.Tensor): Ground truth keypoints in pixel units with shape (Gt,K,d) where K is the number of keypoints per objects,
                d is dimension, and Gt is the total number of ground-truth objects
            batch_idx (torch.Tensor): Batch index tensor with shape (Gt, 1)
            stride_tensor (torch.Tensor): Stride tensor to convert feature-grid units to pixel units and vice versa with shape (N,1)
            target_bboxes (torch.Tensor): Ground truth boxes in the xyxy format in feature grid units with shape (B, N, 4)
            pred_kpts (torch.Tensor): Predicted keypoints in feature-grid unit with shape (B, N, K, d)
        Returns:
            kpts_loss (torch.Tensor): The keypoints loss
            kpts_obj_loss (torch.Tensor): The keypoint object loss
        """
        batch_idx=batch_idx.flatten() # (Gt,1)->(Gt,)
        batch_size=len(masks)
        
        # Find the maximum number of instances/objects/keypoints in a single image
        max_kpts=torch.unique(batch_idx, return_counts=True)[1].max()
        
        # Create a tensor to hold batched keypoints
        batched_keypoints=torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )
        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i=keypoints[batch_idx==i]
            batched_keypoints[i, :keypoints_i.shape[0]]=keypoints_i
        # The following vectorization should work even with batch_idx=torch.zeros(0) and keypoints=torch.zeros(0,17,3)
        # # Count how many keypoints per batch
        # counts=torch.bincount(batch_idx.long(), minlength=batch_size)
        # # Create position indices within each batch
        # pos_idx=torch.cat([torch.arange(c, device=keypoints.device) for c in counts])
        # # Advance indexing with element-wise pairing of indices. len(batch_idx)==len(pos_idx) so Pytorch pairs
        # # the 1st element in batch_idx with the 1st element in pos_idx and use the pair as an index into the 1st two dimension of batched_keypoints
        # batched_keypoints[batch_idx.long(), pos_idx]=keypoints 
        
        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded=target_gt_idx.unsqueeze(-1).unsqueeze(-1) # (B, N, 1,1)
        
        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints=batched_keypoints.gather( # (B,N,K,d)
            1, target_gt_idx_expanded.expand(-1,-1,keypoints.shape[1], keypoints.shape[2])
        )
        # (B,N,k,d) /= (1,N,1,1)
        selected_keypoints[...,:2]/=stride_tensor.view(1,-1, 1, 1) # convert from pixel units to feature-grid units
        
        kpts_loss=0.
        kpts_obj_loss=0.
            
        if masks.any():
            gt_kpt=selected_keypoints[masks] #(B,N,K,d)[(B,N)] -> (M,K,d) where M is the number of valid ground truth key points
            area=xyxy2xywh(target_bboxes[masks])[:,2:].prod(1, keepdim=True) # (M,4) -> (M,2) -> (M,1)
            pred_kpt=pred_kpts[masks] # (B,N,K,d)[(B,N)] -> (M,K,d)
            kpt_mask=gt_kpt[...,2]!=0 if gt_kpt.shape[-1]==3 else torch.full_like(gt_kpt[...,0], True) # (M,K)
            kpts_loss=self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area) # pose loss
            if pred_kpt.shape[-1]==3: pts_obj_loss=self.bce_pose(pred_kpt[...,2], kpt_mask.float())
                
        return kpts_loss, kpts_obj_loss

    def __call__(self, preds:Any, batch:dict[str, torch.Tensor])->tuple[torch.Tensor, torch.Tensor]:
        """Calculate the total loss and detach it for pose estimation
        
        Args:
            preds (tuple[list[torch.Tensor], torch.Tensor]): (1) List of BxOxHxW features from each level where O is the output dimension
                as the sum of 4*reg_max (typically 16) and number of classes, and HxW differs according to levels; and (2) BxPxN tensor of 
                predicted keypoints where P is the number of keypoints multiplied by its dimension (e.g., 17x3=51) and N is sum of all H*W
                from all levels, e.g, if preds[0] is [BxOx80x80, BxOx40x40, BxOx20x20] then N=80**2+40**2+20**2
            batch (dict[str, torch.Tensor]): target inputs comprising at least the following keys
                - 'batch_idx': Index to images in the batch containing objects/instances with shape (Gt,) where Gt is the number of objects/instances
                - 'bboxes': Ground truth bounding boxes in the xywh format in normalized unit with shape (Gt,4)
                - 'cls': Ground truth classes with shape (Gt, nc) where nc is the number of classes
                - 'im_file': List of absolute paths to image files
                - 'img': Input image with shape (B, C, H, W) where C is the number of channels, e.g., 3 for RGB
                - 'keypoints': Ground truth keypoints in normalized unit with shape (Gt, K, d) where K is the number of keypoints per instances and 
                    d is dimension
                - 'ori_shape': List of original [height, width] of images in this batch
                - 'resized_shape': List of [height, width] of input images in this batch
        Returns:
            (torch.Tensor): Batch-aware differentiable loss of box, pose, kobj, cls, dfl
            (torch.Tensor): Non-differentiable loss of box, pose, kobj, cls, dfl
        """
        loss=torch.zeros(5, device=self.device) # box, pose, kobj, cls, dfl
        # Each feats is of size BxOxHxW where O is the output dimension, and HxW is differed depending on output levels
        #    Here O is the sum of 4*reg_max (typically 16) and nc (typically 1) so it is typically 65
        #    feats is ordered from high-resolution to lower resolutions
        # pred_kpts is of size BxPxN where P is the number of keypoints multiplied by its dimension (e.g., 51=17x3) and 
        #     N is H*W
        feats, pred_kpts=preds if isinstance(preds[0], list) else preds[1]
        # cat converts BxOxHxW where H, W differ per level to BxOxN where N is the sum of H*W from all levels
        # split gives BxRxN and BxLxN where R is 4*reg_max and L is nc
        pred_distri, pred_scores=torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max*4, self.nc), 1
        )
    
        # B, grids, ...
        pred_scores=pred_scores.permute(0,2,1).contiguous() # from BxLxN to BxNxL
        pred_distri=pred_distri.permute(0,2,1).contiguous() # from BxRxN to BxNxR
        pred_kpts=pred_kpts.permute(0,2,1).contiguous() # from BxPxN to BxNxP where P is the number of keypoints multiplied by its dimension
        
        dtype=pred_scores.dtype
        # feats[0] is corresponding to stride[0] i.e., highest feat resolution with lowest stride
        imgsz=torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype)*self.stride[0] # image size (h, w)
        anchor_points, stride_tensor=make_anchors(feats, self.stride, 0.5) # Nx2 and Nx1
    
        # Targets
        batch_size=pred_scores.shape[0]
        batch_idx=batch['batch_idx'].view(-1,1) # from shape of (Gt,) to (Gt,1) where Gt is the total number of ground-truth objects
        targets=torch.cat([batch_idx, batch['cls'], batch['bboxes']], 1) # combine (Gt,1), (Gt,1), (Gt,4) to (Gt,6)
        # BxMx5 where M is the maximum labels each image has, 5 for cls and xyxy bounding box coordinates in pixel units
        targets=self.preprocess(targets, batch_size, scale_tensor=imgsz[[1,0,1,0]]) 
        gt_labels, gt_bboxes=targets.split((1,4), 2) # BxMx1 cls, BxMx4 xyxy
        # check whether sum of box coordinates > 0.. This mask is float32 with value of 1 for a box having the sum of xyxy >0
        mask_gt=gt_bboxes.sum(2, keepdim=True).gt_(0.) 
        
        # Predicted boxes in xyxy in feature grid unit
        pred_bboxes=self.bbox_decode(anchor_points, pred_distri) # (b, N, 4) where N is the number of anchors or sum of H*W from all levels
        # keypoints are predicted relative to the cell/anchor center as normalized offsets 
        pred_kpts=self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape)) # (b, N, 17, 3)
        # Below are of size (b, N, 4), (b, N, C), (b, N) and (b,N) where C is the number of classes. We note that assigner determines targets in pixel units
        _, target_bboxes, target_scores, fg_mask, target_gt_idx=self.assigner(pred_scores.detach().sigmoid(),
                                                                              (pred_bboxes.detach()*stride_tensor).type(gt_bboxes.dtype),
                                                                              anchor_points*stride_tensor, gt_labels, gt_bboxes, mask_gt)
        target_scores_sum=max(target_scores.sum(), 1)
        # Cls loss
        loss[3]=self.bce(pred_scores, target_scores.to(dtype)).sum()/target_scores_sum  # (b,N,1)->(1,)
        
        # Bbox loss
        if fg_mask.sum():
            target_bboxes/=stride_tensor  # (b,N,4)/(N,1)=(b,N,4) box coordinates in the xyxy format in feature grid units
            loss[0],loss[4]=self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask)
            keypoints=batch['keypoints'].to(self.device).float().clone()
            keypoints[...,0]*=imgsz[-1]
            keypoints[...,1]*=imgsz[0]
            loss[1], loss[2]=self.calculate_keypoints_loss(fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, 
                                                           target_bboxes, pred_kpts)
        
        loss[0]*=self.hyp.box # box gain
        loss[1]*=self.hyp.pose # pose gain
        loss[2]*=self.hyp.kobj # kobj gain
        loss[3]*=self.hyp.cls # cls gain
        loss[4]*=self.hyp.dfl # dfl gain
        # Losses have already been normalized (computed as mean) over instances / keypoints
        # so we multiply batch_size to make it batch_size invariance 
        return loss*batch_size, loss.detach() # loss(box, pose, kobj, cls, dfl)
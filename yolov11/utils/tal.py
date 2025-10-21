#tal stands for Task-aligned learning
#Ground truth must be assigned to predicted points (as well as anchor points) during training (label assignment). TAL uses both classification confidence and #localization quality (IoU) in this assignment

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .metrics import bbox_iou

class TaskAlignedAssigner(nn.Module):
    """
    A task-aligned assigner for object detection
    This class assignes ground-truth (gt) objects to anchors based on the task-aligned metric, which combines both 
    classification and localization information
    """
    def __init__(self, topk:int=13,num_classes:int=80,alpha:float=1.,beta:float=6.,eps:float=1e-9):
        """
        Initialize a TaskAligneAssigner object with customizable hyperparameters
        Args:
            topk (int, optinal): The number of top candidates to consider
            num_classes (int, optional): The number of object classes
            alpha (float, optional): The alpha parameter for the classifiction component of the task-aligned metric
            beta (float, optional): The beta parameter for the localization component of the task-aligned metric
            eps (float, optional): A small value to prevent division by zero
        """
        super().__init__()
        self.topk=topk
        self.num_classes=num_classes
        self.alpha=alpha
        self.beta=beta
        self.eps=eps

    @staticmethod
    def select_candidates_in_gts(xy_centers:torch.Tensor, gt_bboxes:torch.Tensor, eps:float=1.e-9):
        """
        Select positive anchor centers within ground truth bounding boxes
        Args:
            xy_centers (torch.Tensor): Anchor center coordinates, shape (mHW, 2) where mHW is the sum of the 
                multiplication of height and width of all features from all scales
            gt_bboxes (torch.Tensor): Ground truth bounding boxes, shape (b, n_boxes, 4), where n_boxes is the
                maximum number of ground truth boxes in the images in this batch
            eps (float, optional): Small value representing zero  
        Returns:
            (torch.Tensor): Mask of positive anchors, shape (b, n_boxes, mHW)
        """
        n_anchors=xy_centers.shape[0]
        bs, n_boxes,_=gt_bboxes.shape
        # bsxn_boxesx4 ->(bs*n_boxes)x1x4-> (bs*n_boxes)x1x2 (bs*n_boxes)x1x2
        lt, rb=gt_bboxes.view(-1,1,4).chunk(chunks=2, dim=2) # left-top, right-bottom
        # xy_centers[None]-lt : 1xn_anchorsx2 - (bs*n_boxes)x1x2
        # cat([(bs*n_boxes) x n_anchors x 2, (bs*n_boxes) x n_anchors x 2]) -> (bs*n_boxes) x n_anchors x 4
        # view (bs*n_boxes) x n_anchors x 4 -> bs x n_boxes x n_anchors x 4
        bbox_deltas=torch.cat((xy_centers[None]-lt, rb-xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors,-1)
        return bbox_deltas.amin(dim=3).gt_(eps) # bs x n_boxes x n_anchors x 4 -> bs x n_boxes x n_anchors 
        
    def get_box_metrics(self, pd_scores:torch.Tensor, pd_bboxes:torch.Tensor, gt_labels:torch.Tensor,gt_bboxes:torch.Tensor,
                       mask_gt:torch.Tensor):
        """
        Compute alignment metric given predicted and ground truth bounding boxes
        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, n_anchors, n_classes)
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, n_anchors, 4)
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1)
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4)
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, n_anchors)
        Returns:
            (torch.Tensor): Alignment metric combining classification and localization with shape (bs, n_max_boxes, n_anchors)
            (torch.Tensor): IoU overlaps between predicted and ground truth boxes with shape (bs, n_max_boxes, n_anchors)
        """
        bs, n_max_boxes, na=mask_gt.shape # number of anchor points
        mask_gt=mask_gt.bool() # b, max_n_obj, mHW=na
        overlaps=torch.zeros([bs, n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores=torch.zeros([bs, n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)
        
        ind=torch.zeros([2, bs, n_max_boxes], dtype=torch.long) # 2, bs, max_num_obj
        ind[0]=torch.arange(end=bs).view(-1,1).expand(-1, n_max_boxes) # bs, max_num_obj
        ind[1]=gt_labels.squeeze(-1) # bs,max_num_obj,1 -> bs,max_num_obj
        # Extract the predicted score of each anchor corresponding to the class of each GT box
        # i.e., for each batch b and for each ground truth j (gt_label[b,j]), we want pd_scores[b, :, gt_labels[b,j]] 
        # which is the a vector of length n_anchors (scores of that class across all anchors)
        bbox_scores[mask_gt]=pd_scores[ind[0],:,ind[1]][mask_gt] # pd_scores (b,n_anchors,nc)->(b,max_num_obj,n_anchors)
        
        # Nx4 = (b,n_anchors,4)->(b,1,n_anchors,4)->(b,max_n_obj,n_anchors,4)[b,max_n_obj,n_anchors]
        pd_boxes=pd_bboxes.unsqueeze(1).expand(-1, n_max_boxes, -1,-1)[mask_gt]
        # Nx4 = (b, max_n_obj,4)->(b, max_n_obj,1, 4)-> (b, max_n_obj,n_anchors,4)[b,max_n_obj,n_anchors]
        gt_boxes=gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        # (bs,max_n_obj,n_anchors)->N = (Nx4,Nx4)->Nx1->N                          
        overlaps[mask_gt]=bbox_iou(gt_boxes, pd_boxes, xywh=False, CIoU=True).squeeze(-1).clamp_(min=0.)
        align_metric=bbox_scores.pow(self.alpha)*overlaps.pow(self.beta)
        return align_metric, overlaps

    def select_topk_candidates(self, metrics, topk_mask=None):
        """
        Select the top-k anchor candidates for each ground truth based on the given metrics
        Args:
            metrics (torch.Tensor): A metric of shape (b, max_num_obj, n_anchors), where b is the batch size, 
                max_num_obj is the maximum number of objects, and n_anchors is the total number of anchor points which
                equal to the number of grids from all features 
            topk_mask (torch.Tensor, optional): A mask of shape (b, max_num_obj, topk), where topk is the number of
                top candidates to be considered. If not provided, the top-k values are automatically computed based on
                the given metrics
        Returns:
            (torch.Tensor): Selected top-k candidates of shape (b, max_num_obj, n_anchors)
        """
        # (b, max_num_obj, topk), for each GT, top-k anchors
        topk_metrics, topk_idxs=torch.topk(metrics, self.topk, dim=-1, largest=True)
    
        if topk_mask is None:
            # (b, max_num_obj, topk)= (b, max_num_obj, topk)-max->(b, max_num_obj, 1)-expand->(b, max_num_obj, topk)
            topk_mask=(topk_metrics.max(-1, keepdim=True).values>self.eps).expand_as(topk_idxs)
        # (b, max_num_obj, topk)
        topk_idxs.masked_fill_(~topk_mask, 0) # remove invalid ground-truth
        
        # count_tensor (b, max_num_obj, n_anchors): for each GT, how many anchors are considered matched anchors
        count_tensor=torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        ones=torch.ones_like(topk_idxs[:,:,:1], dtype=torch.int8, device=topk_idxs.device) # (b, max_num_obj, 1)
        for k in range(self.topk):
            # Expand topk_idxs for each value of k and add 1 at the specified positions
            count_tensor.scatter_add_(dim=-1, index=topk_idxs[:,:,k:k+1], src=ones)
        # Filter invalid bboxes
        count_tensor.masked_fill_(count_tensor>1,0) # remove anchors that match the same GT more than 1 times, typically anchor index 0 from masking
        
        return count_tensor.to(dtype=metrics.dtype)
    
    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        """
        Get positive mask for each ground truth box
        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, n_anchors, n_classes)
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, n_anchors, 4)
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1)
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4)
            anc_points (torch.Tensor): Anchor points with shape (n_anchors, 2)
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1)
        Returns:
            (torch.Tensor): Positive anchor mask with shape (bs, n_max_boxes, n_anchors)
            (torch.Tensor): Alignment metric with shape (bs, n_max_boxes, n_anchors)
            (torch.Tensor): Overlaps between predicted and ground-truth boxes with shape (bs, n_max_boxes, n_anchors)
        """
        # floating-point mask Bxmax_num_objxmHW where max_num_obj is the number of max gt-boxes and mHW is the number of anchors
        mask_in_gts=self.select_candidates_in_gts(anc_points, gt_bboxes) 
        # Get anchor_align metric, (b, max_num_obj, mHW)
        align_metric, overlaps=self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts*mask_gt)
        # Get topk_metric mask (b, max_num_obj, n_anchors)
        mask_topk=self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1,-1,self.topk).bool())
        # Merge all mask to a final mask, (b, max_num_obj, n_anchors)
        mask_pos=mask_topk*mask_in_gts*mask_gt
        return mask_pos, align_metric, overlaps
    
    @staticmethod
    def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
        """
        Select anchor boxes with the highest IoU when assigned to multiple ground truths. In other words, for each anchor box,
        find the best GT for it
        Args:
            mask_pos (torch.Tensor): Positive anchor mask, shape (b, n_max_objs, n_anchors)
            overlaps (torch.Tensor): IoU overlaps, shape (b, n_max_objs, n_anchors)
            n_max_boxes (int): Maximum number of ground truth boxes
        Returns:
            (torch.Tensor): Indices of assigned ground truths, shape (b, n_anchors)
            (torch.Tensor): Foreground mask, shape (b, n_anchors)
            (torch.Tensor): Updated positive anchor mask, shape (b, n_max_boxes, n_anchors)
        """
        # Convert (b, n_max_boxes, n_anchors)->(b, n_anchors)
        fg_mask=mask_pos.sum(-2) # for each anchor, how many gt it was associated
    
        # If one anchor is assigned to multiple gt_bboxes, we pick the best gt for it
        # i.e., each anchor must be associated with at most 1 ground truth
        if fg_mask.max()>1: 
            # (b, n_anchors) -> (b, 1, n_anchors) -> (b, n_max_objs, n_anchors)
            mask_multi_gts=(fg_mask.unsqueeze(1)>1).expand(-1, n_max_boxes, -1) # anchors with multiple GT
            # for each anchors, which is the best GT (index) associated with the highest IoU
            max_overlaps_idx=overlaps.argmax(dim=1) # (b,n_max_objs, n_anchors)->(b, n_anchors)
            
            # Float anchor-mask tensor (b, n_max_objs, n_anchors) storing 1 at the gt-index (max_overlaps_idx)
            # if it is the gt with highest IoU for each anchor 
            is_max_overlaps=torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device) 
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)
            
            # Modify (b, n_max_objs, n_anchors) float32 anchor-mask, so if mask_multi_gts>1, get a new mask value from
            # is_max_overlaps, otherwise mask_pos
            mask_pos=torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float() 
            fg_mask=mask_pos.sum(-2) # update fg_mask
            
        # For each anchor, which is its corresponding GT
        target_gt_idx=mask_pos.argmax(-2) # (b, n_anchors)
        
        return target_gt_idx, fg_mask, mask_pos 

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """
        Compute target labels, target bounding boxes, and target scores for the positive anchor points
        Args:
            gt_labels (torch.Tensor): Ground truth labels of shape (b, num_max_boxes, 1) where b is the batch size,
                num_max_boxes is the maximum number of objects
            gt_bboxes (torch.Tensor): Ground truth bounding boxes of shape (b, num_max_boxes, 4)
            target_gt_idx (torch.Tensor): Indices of the assigned ground truth objects for positive anchors with shape
                (b, n_anchors), where n_anchors is the number of feature grids from all scales 
            fg_mask (torch.Tensor): Booleam tensor of shape (b, n_anchors) indicating the positive (foreground) anchor
                points
        Returns:
            (torch.Tensor): Long target labels for positive anchor points with shape (b, n_anchors)
            (torch.Tensor): Target bounding boxes for positive anchor poinst with shape (b, n_anchors, 4)
            (torch.Tensor): Target scores for positive anchor points with shape (b, n_anchors, n_classes)
        """
        bs, n_max_boxes=gt_labels.shape[:2]
    
        # Assigned target labels, changing gt_labels from (b, n_max_boxes, 1) to (b, n_anchors)
        batch_ind=torch.arange(end=bs, dtype=torch.long, device=gt_labels.device)[...,None] # (b,1)
        # (b, n_anchors) target_index incorporating the batch-index
        target_gt_idx=target_gt_idx+batch_ind*n_max_boxes  # flattened indices for (b, n_max_boxes) dimensions
        # gt_labels.long().flatten() from (b, n_max_boxes, 1)-> (b*n_max_boxes,)
        target_labels=gt_labels.long().flatten()[target_gt_idx] # (b, n_anchors)
        
        # Assigned target boxes, (b, n_max_boxes, 4)-> (b, n_anchors, 4)
        # (b, n_max_boxes, 4)->(b*n_max_boxes, 4)[(b,n_anchors)]->(b, n_anchors, 4)
        target_bboxes=gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]
        
        # Assigned target scores
        target_labels.clamp_(min=0)
        # 10x faster than F.one_hot
        target_scores=torch.zeros((bs, target_labels.shape[1], self.num_classes), dtype=torch.int64,
                                  device=target_labels.device) # (b, n_anchors, nc), for example coco nc =80
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)
        
        # (b, n_anchors) -> (b, n_anchors, nc), for example nc =80
        fg_scores_mask=fg_mask[:,:,None].repeat(1,1,self.num_classes) 
        target_scores=torch.where(fg_scores_mask>0, target_scores, 0)
        
        return target_labels, target_bboxes, target_scores
        
def make_anchors(feats, strides, grid_cell_offset=0.5):
    """
    Generate anchors from features, where anchor points are box center reference
    Args:
        feats (list | tuple): List of multiscale BxOxHxW feature tensors where O is the output size, or tuple of height, width per output scale
        strides (torch.Tensor): 1D Tensor of strides to form multiscale feats, i.e., if strides is of size (3,), feats will contain 3 features
            or 3 pairs of height and width
        grid_cell_offset (float): Offset to move anchor points outside pixel grid
    Returns:
        (torch.Tensor): mHWx2 anchor points where m is the number of strides, i.e., concatenation of all anchor points from all resolutions
        (torch.Tensor): mHWx1 stride associated with each correponding anchor point
    """
    anchor_points, stride_tensor=[],[]
    assert feats is not None
    dtype, device=feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        # feats are either list of BxOxHxW features or tuple of floating point (H, W) 
        h,w = feats[i].shape[2:] if isinstance(feats, list) else (int(feats[i][0]), int(feats[i][1]))
        sx=torch.arange(end=w, device=device, dtype=dtype)+grid_cell_offset # shift x of size (w,)
        sy=torch.arange(end=h, device=device, dtype=dtype)+grid_cell_offset # shift y of size (h,)
        sy, sx=torch.meshgrid(sy, sx, indexing='ij') # each of size (h, w)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1,2)) # (h, w, 2) -> (hw, 2)
        stride_tensor.append(torch.full((h*w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor) # (mhw,2), (mhw, 1)

def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """
    Transform distance (left-top right bottom) to box (xywh or xyxy)
    Args:
        distance (torch.Tensor): BxmHWx4 where B is the batch size, mHW is the number of anchor points (number of 
            feature grids from all scales), 4 is for left, top, right, bottom
        anchor_points (torch.Tensor): mHWx2 box center reference where mHW is the number of anchor points (number of feature grids from 
            all scales), 2 is for x, y in feature-grid units (or pixel_units)
    Returns:
        (torch.Tensor): BxmHWx4 bounding box coordinates in xywh or xyxy format in the same units as anchor_points
    """
    lt, rb=distance.chunk(2, dim=dim) # BxmHWx4 to BxmHWx2 and BxmHWx2
    x1y1=anchor_points-lt
    x2y2=anchor_points+rb
    if xywh:
        c_xy=(x1y1+x2y2)/2
        wh=x2y2-x1y1
        return torch.cat([c_xy, wh], dim=dim) # xywh in feature-grid or pixel units
    return torch.cat([x1y1, x2y2], dim=dim) # xyxy in feature-grid or pixel units

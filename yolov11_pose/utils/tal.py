from __future__ import annotations

from typing import Any
import warnings

import torch
import torch.nn as nn

from computer_vision.yolov11_pose.utils.metrics import bbox_iou

class TaskAlignedAssigner(nn.Module):
    """A task-aligned assigner for object detection
    
    This class assigns ground-truth (gt) objects to anchors based on the task-aligned metric, which combines both classification and localization 
    information
    """
    def __init__(self, topk:int=13, num_classes:int=80, alpha:float=1., beta:float=6., stride:list=[8,16,32], eps:float=1e-9, topk2:int=None):
        """Initialize a TaskAlignedAssigner object with customizable hyperparameters
        Args:
            topk (int, optional): The number of top candidates to consider
            num_classes (int, optional): The number of object classes
            alpha (float, optional): The alpha parameter for the classification component of the task aligned metric
            beta (float, optional): The beta parameter for the localization component of the task aligned metric
            stride (list, optional): List of stride values for different feature levels
            eps (float, optional): A small value to prevent division by zero
            topk2 (int, optional): Secondary topk value for additional filtering
        """
        super().__init__()
        self.topk=topk # the number of top candidates to consider
        self.topk2=topk2 or topk
        self.num_classes=num_classes # the number of object classes
        self.alpha=alpha # alpha parameter for the classification component of the task-aligned metric
        self.beta=beta # beta parameter for the localization component of the task-aligned metric
        self.stride=stride
        self.eps=eps # A small value preventing division by zero
        
    @staticmethod
    def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
        """Select positive anchor centers within ground-truth bounding boxes
        Args:
            xy_centers (torch.Tensor): Anchor center coordinates, shape (N, 2) where N is the number of anchors/sum of H*W from all levels and 
                2 for x and y in pixel units
            gt_bboxes (torch.Tensor): Ground truth bounding boxes, shape (B, n_max_boxes, 4) where n_max_boxes is the maximum number of boxes 
                for each image in this batch and 4 is for bounding boxes coordinates in xyxy format in pixel units
            eps (float, optional): Small value for numerical stability
        Returns:
            (torch.Tensor): Mask of positive anchors, shape (B, n_max_boxes, N), of type float32 with value 1 represent positive anchors and 0 otherwise
        Notes:
            - b: batch size, n_max_boxes:number of maximum ground truth boxes, h:height, w:width
            - Bounding box format: [x_min, y_min, x_max, y_max]
        """
        n_anchors=xy_centers.shape[0]
        bs, n_max_boxes,_=gt_bboxes.shape 
        # left-top and right-bottom of each boxes in pixel units
        lt, rb=gt_bboxes.view(-1, 1, 4).chunk(2, 2) # convert from (B, n_max_boxes, 4) to (B*n_max_boxes, 1, 4) to two of (B*n_max_boxes,1,2) 
        # Find anchor centers within ground-truth bounding boxes
        # xy_centers[None] is of size (1,N,2) so (xy_centers[None]-lt) is of size (B*n_max_boxes,N,2)
        bbox_deltas=torch.cat((xy_centers[None]-lt, rb-xy_centers[None]), dim=2).view(bs, n_max_boxes, n_anchors,-1) # (B,n_max_boxes, N, 4)
        # Find minimum coordinate delta for each point difference and only get those that greater than eps, i.e., min along coordinate dimension
        return bbox_deltas.amin(3).gt_(eps) # (B,n_max_boxes, N)
    
    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """Calculate IoU 
        Args:
            gt_bboxes (torch.Tensor): Groud truth boxes of size (P,4), where P is the number of boxes and 4 is for coordinates in 
                the xyxy format in pixel units
            pd_bboxes (torch.Tensor): Predicted boxes of size (P,4), where P is the number of boxes and 4 is for coordinates in 
                the xyxy format in pixel units
        Returns:
            (torch.Tensor): IoU values of shape (P,) between each pair of boxes
        """
        return bbox_iou(gt_bboxes, pd_bboxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0) # from shape of (P,1) to (P,)

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """Compute alignment metric given predicted and ground truth bounding boxes. The higher the alignment metric, the better alignment of
        the ground truth and the predicted boxes
        Args:
            pd_score (torch.Tensor): Predicted classification scores with shape (B, N, C) where N is the number of anchors/sum of H*W from all 
                feature levels and C is the number of classes
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (B, N, 4) in the xyxy format in pixel unit
            gt_labels (torch.Tensor): Ground truth labels with shape (B, n_max_boxes, 1) where n_max_boxes is the maximum number of boxes for
                each image in this batch
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (B, n_max_boxes, 4) in the xyxy format in pixel unit
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with positive anchors with shape (B, n_max_boxes, N)
        Returns:
            align_metric (torch.Tensor): Alignment metric combining classification and localization with shape (B, n_max_boxes, N)
            overlaps (torch.Tensor): IoU overlaps between predicted and ground truth boxes with shape (B, n_max_boxes, N)
        """
        na=pd_bboxes.shape[-2] # number of anchors
        mask_gt=mask_gt.bool() # (B, n_max_boxes, N)
        overlaps=torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)  # (B, n_max_boxes, N)
        bbox_scores=torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device) # (B, n_max_boxes, N)
        
        ind=torch.zeros([2,self.bs, self.n_max_boxes], dtype=torch.long) # (2,B,n_max_boxes)
        # Batch indices
        ind[0]=torch.arange(end=self.bs).view(-1,1).expand(-1,self.n_max_boxes) # from (B,) to (B,1) to (B,n_max_boxes)
        # Class indices for each ground truth object
        ind[1]=gt_labels.squeeze(-1) # (B,n_max_boxes)
        # Get the scores of each grid for each gt cls 
        # pred_scores[ind[0]] returns tensor of size (B, n_max_boxes, N, 1)
        # pred_scores[...,ind[1]] returns tensor of size (B, N, B, n_max_boxes)
        # pred_scores[ind[0],:,ind[1]] returns tensor of size (B, n_max_boxes, N)
        # What does pred_scores[ind[0],:,ind[1]] do? For each (i,j) index to (B,n_max_boxes) of ind[0] and ind[1], Pytorch pairs them together
        # i.e., (ind[0]ij,:,ind[1]ij). Also, in advance indexing, if non-adjacenet dimensions are indexed (like dim0, and dim2 with dim1 left as a
        # a slice), the indexed dimensions are pushed to the front by default. Thus, the indexing of dim0 and dim2 creates a base shape of 
        # (B,n_max_boxes) followed by the sliced dimension of size N
        # pred_scores[ind[0],:,ind[1]] mean for each batch ind[0], give me scores for ground truth box ind[1]'s class
        bbox_scores[mask_gt]=pd_scores[ind[0],:,ind[1]][mask_gt]  # (B, n_max_boxes, N)
        
        # (B,N,4)->(B,1,N,4)->(B,n_max_boxes,N,4) and mask_gt filtering yields [P,4] where P is the number of boxes that are valid across batches,
        # anchors, and ground-truth boxes
        pd_boxes=pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes,-1,-1)[mask_gt] 
        # (B,n_max_boxes,4)->(B,n_max_boxes, 1,4)->(B,n_max_boxes,N,4) and mask_gt filtering yields [P,4] 
        gt_boxes=gt_bboxes.unsqueeze(2).expand(-1,-1,na,-1)[mask_gt]
        overlaps[mask_gt]=self.iou_calculation(gt_boxes, pd_boxes)
        # both bbox_scores and overlaps are in the range [0,1]; with alpha=0.5 and beta=6 typically, we are forgiving low bbox_scores
        # but we care a lot about overlaps. If overlaps is low, power by beta will make align_metric close to zero; on the other hand,
        # if bbox_scores is low, power by 0.5 will boost/increase its value, so it means we tolerate incorrect classification but not localization
        align_metric=bbox_scores.pow(self.alpha)*overlaps.pow(self.beta) # (B, n_max_obj, N)
        return align_metric, overlaps

    def select_topk_candidates(self, metrics, topk_mask=None):
        """Select the top-k anchor/prediction candidates for each ground truth box based on the given metrics
        Args:
            metrics (torch.Tensor): A tensor of shape (B, n_max_boxes, N) where n_max_boxes is the maximum number of boxes in each image in this batch,
                and N is the number of anchors/sum of all H*W from all feature levels
            topk_mask (torch.Tensor, optional): An optional boolean tensor of shape (B, n_max_boxes, topk) where topk is the number of top candidates 
                to consider. If not provided, the top-k values are automatically computed based on the given metrics
        Returns:
            (torch.Tensor): A tensor of shape (B, n_max_boxes, N) containing the selected top-k candidates
        """
        # (B, n_max_boxes, topk)
        topk_metrics, topk_idxs=torch.topk(metrics, self.topk, dim=-1, largest=True)
        if topk_mask is None: # We pick mask based on having topk_metrics>0
            topk_mask=(topk_metrics.max(dim=-1, keepdim=True)[0]>self.eps).expand_as(topk_idxs) # from (B,n_max_boxes, 1) to (B,n_max_boxes, topk)
        # Keep only indices of topk_metrics whose values>0 or whose values selected by input topk_mask (i.e., ground-truth with boxes)
        topk_idxs.masked_fill_(~topk_mask, 0) # (B, n_max_boxes, topk)
        
        count_tensor=torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device) # (B,n_max_boxes, N)
        ones=torch.ones_like(topk_idxs[:,:,:1], dtype=torch.int8, device=topk_idxs.device) # (B,n_max_boxes, 1)
        for k in range(self.topk):
            # Expand topk_idxs for each value of k and add 1 at the specified positions
            count_tensor.scatter_add_(dim=-1, index=topk_idxs[:,:, k:(k+1)], src=ones) # topk_idxs[:,:, k:(k+1)] and ones are of shape (B,n_max_boxes, 1)
        # Filter invalid bboxes
        count_tensor.masked_fill_(count_tensor>1,0) # each anchor matches exactly 1 ground truth
        return count_tensor.to(metrics.dtype)
        
    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        """Get positive mask for each ground truth box
        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (B, N, C) where N is the number of anchors/sum of all H*W from all levels
                and C is the number of classes
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (B, N, 4) in the xyxy format in pixel units
            gt_labels (torch.Tensor): Ground truth labels with shape (B, n_max_boxes, 1) where n_max_boxes is the maximum number of boxes for each image
                in this batch
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (B, n_max_boxes, 4) in the xyxy format in pixel units
            anc_points (torch.Tensor): Anchor points with shape (N, 2) where 2 is for x, y in pixel units
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (B, n_max_boxes, 1)
        Returns:
            mask_pos (torch.Tensor):  Mask of positive anchors/predicted-boxes per ground truth box with shape (B, n_max_boxes, N)
            align_metric (torch.Tensor): Alignment metric with shape (B, n_max_boxes, N); the higher value, the better alignment
            overlaps (torch.Tensor): Overlaps between predicted boxes and ground truth boxes with shape (B, n_max_boxes, N)
        """
        # Get anchors that located inside ground truth boxes
        mask_in_gts=self.select_candidates_in_gts(anc_points, gt_bboxes) # (B,n_max_boxes, N) mask of positive anchors of type float32
        # Get anchor_align metric and overlaps CIoU, both of size (B, n_max_boxes, N)
        align_metric, overlaps=self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts*mask_gt)
        # Get topk_metric mask, (B, n_max_boxes, N)
        mask_topk=self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1,-1,self.topk).bool())
        # Merge all mask to a findal mask (B, n_max_boxes, N)
        mask_pos=mask_topk*mask_in_gts*mask_gt
        return mask_pos, align_metric, overlaps

    @staticmethod
    def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
        """Select anchor boxes with highest IoU when assigned to multiple ground truths.
        Args:
            mask_pos (torch.Tensor): Mask of positive anchors/predicted-boxes per ground truth box with shape (B, n_max_boxes, N) where N is the
                number of anchors/sum of all H*W from all feature levels
            overlaps (torch.Tensor): IoU overlaps with shape (B, n_max_boxes, N)
            n_max_boxes (int): Maximum number of ground-truth boxes per image in this batch
        Returns:
            target_gt_idx (torch.Tensor): Indices of assigned ground truth with shape (B, N)
            fg_mask (torch.Tensor): Foreground mask with shape (B, N) telling how many ground-truth boxes per anchor
            mask_pos (torch.Tensor): Updated mask of positive anchors/predicted-boxes per ground truth box with shape (B, n_max_boxes, N)
        """
        # (B, n_max_boxes, N) to (B,N)
        fg_mask=mask_pos.sum(dim=-2) # sum along the n_max_boxes/ground-truth-box direction
        if fg_mask.max()>1: # one anchor/predicted-box is assigned to more than 1 ground truth box
            mask_multi_gts=(fg_mask.unsqueeze(1)>1).expand(-1, n_max_boxes, -1) # (B, n_max_boxes, N)
            max_overlaps_idx=overlaps.argmax(dim=1) # for each anchor, find (B,N) indices to ground-truth associated with maximum IoU
        
            # is_max_overlap actually store mask (i.e., value=1) at ground-truth location offering maximum IoU for each anchors
            is_max_overlaps=torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device) # (B, n_max_boxes, N)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)
        
            # Create a new mask_pos with 
            #  - if originally 1 anchors was assigned to more than 1 ground truth. Reassigned the anchors to ground-truth having highest IoU 
            #  - otherwise, keep the original mask
            mask_pos=torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()  # (B, n_max_boxes, N)
            fg_mask=mask_pos.sum(-2)
        # Find indices of ground-truth boxes for each anchor
        target_gt_idx=mask_pos.argmax(-2) # (B, N)
        return target_gt_idx, fg_mask, mask_pos

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """Compute target labels, target bounding boxes, and target scores for the positive anchor points.
        Args:
            gt_labels (torch.Tensor): Ground truth labels of shape (B,n_max_boxes,1), where B is the batch size and n_max_boxes is the maximum
                number of boxes of each image in this batch
            gt_bboxes (torch.Tensor): Ground truth bounding boxes (B,n_max_boxes,4) in the xyxy format in pixel units
            target_gt_idx (torch.Tensor): Indices of the assigned ground truth objects for positive anchor points, with shape (B,N) where N is
                the number of anchors/sum of H*W for all feature levels
            fg_mask (torch.Tensor): A float32 tensor of shape (B,N) masking positive (foreground) anchors as 1 and 0 otherwise
        Returns:
            target_labels (torch.Tensor): Target labels for positive anchor points with shape (B,N)
            target_bboxes (torch.Tensor): Target bounding boxes for positive anchor points with shape (B,N,4)
            target_scores (torch.Tensor): Target scores for positive anchor points with shape (B, N, C) where C is the number of classes
        """
        # Assign target labels
        batch_ind=torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[...,None] # (B, 1)
        # Compute flatten 1D indices into gt_labels of shape (B,n_max_boxes, 1)
        target_gt_idx=target_gt_idx+batch_ind*self.n_max_boxes
        target_labels=gt_labels.long().flatten()[target_gt_idx]  # (B, N)
        target_labels.clamp_(0)
        
        # Assign target boxes
        target_bboxes=gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx] #  (B,n_max_boxes,4)-> (B*n_max_boxes,4)->(B,N,4)
        
        # Assign target scores: 10x faster than F.one_hot()
        target_scores=torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.num_classes),
             dtype=torch.int64, device=target_labels.device
        ) # (B, N, C) where C is the number of classes
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1) # assign value 1 at class-channel based on  target_labels
        # fg_mask whether there is a ground-truth for each anchors
        fg_scores_mask=fg_mask[:,:,None].repeat(1, 1, self.num_classes) # (B,N,C)
        target_scores=torch.where(fg_scores_mask>0, target_scores, 0)
        return target_labels, target_bboxes, target_scores

    def _forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """Compute the task-aligned assignment
        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (B, N, C) where N is the number of anchors/sum of H*W from all
                feature levels, and C is the number of classes
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (B, N, 4) in the xyxy format in pixel units
            anc_points (torch.Tensor): Anchor points with shape (N, 2)  where 2 is for x and y and in pixel units
            gt_labels (torch.Tensor): Ground truth labels with shape (B, n_max_boxes, 1)
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (B, n_max_boxes, 4) in the xyxy format in pixel units
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (B, n_max_boxes, 1)
        Returns:
            target_labels (torch.Tensor): Target labels with shape (B, N) where N is the number of anchors/sum of all H*W from all feature levels
            target_bboxes (torch.Tensor): Target bounding boxes with shape (B, N, 4) in the xyxy format in pixel units
            target_scores (torch.Tensor): Soft target scores with shape (B, N, C) where C is the number of classes
            fg_mask (torch.Tensor): Foreground mask with shape (B, N) telling which anchors are associated with ground-truth boxes
            target_gt_idx (torch.Tensor): Indices of target ground truth with shape (B, N), i.e., for each anchor, which ground-truth was assigned 
        """
        # All tensors below are of shape (B, n_max_boxes, N)
        mask_pos, align_metric, overlaps=self.get_pos_mask(pd_scores=pd_scores, pd_bboxes=pd_bboxes, gt_labels=gt_labels, gt_bboxes=gt_bboxes, 
                                                           anc_points=anc_points, mask_gt=mask_gt)
        # target_gt_idx and fg_mask are of size BxN
        target_gt_idx, fg_mask, mask_pos=self.select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)
        
        # Assigned targets with target_labels (B,N), target_bboxes (B,N,4), target_scores (B,N,C) where C is the number of class
        # target_scores is hard labels with values of 1 or 0
        target_labels, target_bboxes, target_scores=self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)
        
        
        # Normalize/soften target_scores ensures that the supervision signal is spatially smooth--anchors closer to the center of an object
        # with better shapes receive stronger positive reinforcement
        align_metric*=mask_pos  # (B, n_max_boxes, N) only positive anchors have non-zero align-metric values
        pos_align_metric=align_metric.amax(dim=-1, keepdim=True) # (B, n_max_boxes, 1) maximum metric values for each ground-truth box for each image
        pos_overlaps=(overlaps*mask_pos).amax(dim=-1, keepdim=True) # (B, n_max_boxes, 1) maximum IoU for each ground-truth box for each image
        # (B, n_max_boxes, N) -> (B, N) -> (B, N, 1)
        # pos_align_metric scales norm_align_metric to [0,1] and pos_overlaps ensures that the best anchor for a specific object has target scores equal
        # to its IoU, preventing the model from being overconfident
        norm_align_metric=(align_metric*pos_overlaps/(pos_align_metric+self.eps)).amax(dim=-2).unsqueeze(-1)
        # Soft labels/targets: if anchors are perfectly aligned (high IoU and high classification score), target_scores will be close to 1
        target_scores=target_scores*norm_align_metric #  (B, N, 1)
        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """Compute the task-aligned assignment
        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (B,N,C) where N is the number of anchors/sum of H*W from all levels
                C is the number of classes
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (B, N, 4) in the xyxy format and pixel units
            anc_points (torch.Tensor): Anchor points with shape (N, 2) where 2 is for x and y and in pixel units
            gt_labels (torch.Tensor): Ground truth labels with shape (B, n_max_boxes, 1) where n_max_boxes is the maximum number of boxes for 
                each image in this batch
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (B, n_max_boxes, 4) in the xyxy format in pixel units
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (B, n_max_boxes, 1)
        Returns:
            target_labels (torch.Tensor): Target labels with shape (B, N) where N is the number of anchors/sum of H*W from all levels
            target_bboxes (torch.Tensor): Target bounding boxes with shape (B, N, 4) in the xyxy format and pixel units
            target_scores (torch.Tensor): Soft target scores with shape (B, N, C) where C is the number of classes
            fg_mask (torch.Tensor): Foreground mask with shape (B, N) telling which anchors are associated with ground-truth boxes
            target_gt_idx (torch.Tensor): Target ground truth indices with shape (B, N)
        References:
            https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py
        """
        self.bs=pd_scores.shape[0]
        self.n_max_boxes=gt_bboxes.shape[1]
        device=gt_bboxes.device
        if self.n_max_boxes==0:
            return (
                torch.full_like(pd_scores[...,0], self.num_classes), # (B,N)
                torch.zeros_like(pd_bboxes), # (B,N,4)
                torch.zeros_like(pd_scores), # (B,N,C)
                torch.zeros_like(pd_scores[...,0]), # (B,N)
                torch.zeros_like(pd_scores[...,0]), # (B,N)
            )
        try: return self._forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
        except torch.cuda.OutOfMemoryError:
            # Move tensors to CPU, compute, then move back to original device
            warnings.warn("CUDA OutOfMemoryError in TaskAlignedAssigner, using CPU")
            cpu_tensors=[t.cpu() for t in (pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)]
            result=self._forward(*cpu_tensors)
            return tuple(t.to(device) for t in result)
        
def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors/grid cell centers from features.
    Args:
        feats (list[torch.Tensor]): Features outputted from each level, each of size BxOxHxW where O is the output dimension and 
            HxW is the size of features differing per level. O is typically the sum of 4*reg_max (typically 16) and number of classes.
            Its order is consistent with the order of stride from highest feature resolution to lower
        strides (torch.Tensor): Stride of each level, ordered from smallest stride (giving highest resolution feature) to largest stride
            (giving lowest resolution), e.g., tensor([8,16,32])
        grid_cell_offset (float): Amount of pixel shift applied to x and y directions
    Returns:
        (torch.Tensor): Nx2 anchor positions where N is the sum of H*W from all levels and 2 is for x and y
        (torch.Tensor): Nx1 strides where N is the sum of H*W from all levels
    """
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        h, w = feats[i].shape[2:] if isinstance(feats, list) else (int(feats[i][0]), int(feats[i][1]))
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x to represent the center of the grid cell
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y to represent the center of the grid cell
        # sx=[[0.5, 1.5, 2.5, ...],
        #           ...
        #  [0.5, 1.5, 2.5, ...]]
        # sy=[[0.5, 0.5, 0.5, ...],
        #           ...
        #  [79.5, 79.5, 79.5, ...]]
        sy, sx = torch.meshgrid(sy, sx, indexing="ij") # each of size (h,w)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2)) # stack yields (h,2,w) view gives (h*w, 2)
        stride_tensor.append(torch.full(size=(h * w, 1), fill_value=stride, dtype=dtype, device=device))
    return torch.cat(anchor_points, dim=0), torch.cat(stride_tensor, dim=0)
    
def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """
    Transform distance(ltrb) to box(xywh or xyxy)
    Args:
        distance (torch.Tensor): BxNx4 distribution distance from each anchor points where N is the sum of all H*W from all levels 
            or the number of anchors in feature units
        anchor_points (torch.Tensor): Nx2 where N is the sum of all H*W from all levels or the number of anchors and 2 for x,y in feature
            units
    Returns:
        (torch.Tensor): BxNx4 bounding box coordinates in feature units
    """
    lt, rb=distance.chunk(2, dim=dim) # each of size BxNx2 
    x1y1=anchor_points-lt #  BxNx2 
    x2y2=anchor_points+rb #  BxNx2 
    if xywh:
        c_xy=(x1y1+x2y2)/2
        wh=x2y2-x1y1
        return torch.cat([c_xy,wh], dim=dim) # xywh bbox
    return torch.cat([x1y1,x2y2],dim=dim) # xyxy bbox

def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox in the xyxy format to distance to ltrb (left-top and right-bottom)
    Args:
        anchor_points (torch.Tensor): Anchor locations with shape (N,2) in the feature grid unit where 2 is for x and y
        bbox (torch.Tensor): Ground truth bounding boxes with shape (B, N, 4) or (...,4) in the xyxy format in the feature grid unit
        reg_max (int): Maximum bin index/maximum distance in feature grid unit
    Returns:
        (torch.Tensor): Distance to ltrb (left-top and right-bottom) with shape (B,N,4) or (..,4) 
    """
    x1y1,x2y2=bbox.chunk(2,-1) # (B,N,4) to 2 of (B,N,2) or (...,4) to 2 of (...,2)
    return torch.cat((anchor_points-x1y1, x2y2-anchor_points), dim=-1).clamp_(0,reg_max-0.01) # dist (lt, rb) in shape (...,4) or (B,N,4)
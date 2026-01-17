from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

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
    
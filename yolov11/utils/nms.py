import torch
import torchvision

from computer_vision.yolov11.utils.ops import xywh2xyxy

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.7, classes=None, agnostic=False, multi_label=False, max_det=300, nc=0,
                        max_nms=30000, max_wh=7680, end2end= False, return_idxs= False):
    """
    Perform non-maximum suppression (NMS) on prediction results.
    
    Applies NMS to filter overlapping bounding boxes based on confidence and IoU thresholds. 
    Args:
        prediction (torch.Tensor): Prediction with shape (batch-size, 4+num_class, num_boxes), where 4 is for x,y,w,h
        conf_thres (float): Confidence threshold for NMS filtering. Valid values are between 0 and 1
        iou_thres (float): IoU threshold for NMS filtering. Valid values are between 0 and 1
        classes (list[int],optional): List of class indices to consider. If None, all classes are considered
        agnostic (bool): Whether to perform class-agnostic NMS
        multi_lable (bool): Whether each box can have multiple labels
        max_det (int): Maximum number of detection to keep per image
        nc (int): Number of classes. 
        max_nms (int): Maximum number of boxes for NMS
        max_wh (int): Maximum box width and height in pixel
        return_idxs (bool): Whether to return the indices of kept detections
    Returns:
        output (list[torch.Tensor]): List of detections per image with shape (num_boxes, 6) containing
            (x1,y1,x2,y2, confidence, class)
        keepi (list[torch.Tensor]): Indices of kept detections if return_idxs=True
    """
    
    assert 0<=conf_thres<=1, f'Invalid confidence threshold {conf_thres}, valid values are between 0. and 1.'
    assert 0<=iou_thres<=1, f'Invalid IoU {iou_thres}, valid values are between 0. and 1.'
    
    # YOLOv8 returns (inference_out, loss_out)
    if isinstance(prediction, (list, tuple)): 
        # Bx(K+4)xN where B is batch size, K is number of classes, and N is the number of boxes
        prediction=prediction[0] # select only inference
        
    if classes is not None: torch.tensor(classes, device=prediction.device)
    # prediction is of size B x (K+4) x N
    bs=prediction.shape[0] # batch size (BCN, where C=K+4 e.g., 1x84x1000)
    nc=nc or (prediction.shape[1]-4) # number of class
    mi=4+nc # mask start index
    xc=prediction[:,4:mi].amax(1)>conf_thres # B x N candidates
    xinds=torch.arange(prediction.shape[-1], device=prediction.device).expand(bs, -1)[..., None] # B x N x 1 to track idxs
    
    multi_label &= nc>1 # multiple labels per box 
    
    prediction=prediction.transpose(-1,-2) # B x N x (K+4)
    prediction[..., :4]=xywh2xyxy(prediction[..., :4]) # xywh to xyxy
    
    output=[torch.zeros(0,6,device=prediction.device)]*bs
    keepi=[torch.zeros(0,1,device=prediction.device)]*bs # to store the kept idxs
    
    for xi, (x, xk) in enumerate(zip(prediction, xinds)): # image index, (preds, preds indices)
        # Apply constraints
        filt=xc[xi] # 1D bool tensor of size N
        x=x[filt]
        if return_idxs: xk=xk[filt]
        if x.shape[0]==0: continue # if none remains, process the next image
    
        # detection matrix Nx6 (xyxy, conf, cls)
        box, cls=x.split((4, nc), 1) # Nx4 and NxK
    
        if multi_label:
            i, j=torch.where(cls>conf_thres)
            # Nx6 = Nx4, Nx1 Nx1
            x=torch.cat((box[i], x[i, 4+j,None], j[:,None].float()), 1) 
            if return_idxs: xk=xk[i]
        else: # best class only
            conf, j = cls.max(1, keepdim=True) # each Nx1
            filt=conf.view(-1)>conf_thres
            x=torch.cat((box, conf, j.float()), 1)[filt]
            if return_idxs: xk=xk[filt]
    
        # filter by class
        if classes is not None:
            filt=(x[:,5:6]==classes).any(1)
            x=x[filt]
            if return_idxs: xk=xk[filt]
        # check shape
        n=x.shape[0] # number of boxes
        if not n: continue # no boxes
        if n > max_nms: # excess boxes
            filt=x[:,4].argsort(descending=True)[:max_nms] # sort by confidence and remove excess boxes
            x=x[filt]
            if return_idxs: xk=xk[filt]
        c=x[:,5:6]*(0 if agnostic else max_wh) # classes
        scores=x[:,4] # scores
    
        boxes=x[:,:4]+c # boxes (offset by class)
        i=torchvision.ops.nms(boxes, scores, iou_thres)
        i=i[:max_det] # limit detections
        output[xi]=x[i]
        if return_idxs: keepi[xi]=xk[i].view(-1)
        
    return (output, keepi) if return_idxs else output
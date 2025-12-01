from __future__ import annotations

import math
import torch
import numpy as np

from computer_vision.yolov11_pose.utils import NOT_MACOS14

def empty_like(x):
    """Create empty torch.Tensor or np.ndarray with same shape as input and float32 dtype."""
    return (
        torch.empty_like(x, dtype=torch.float32) if isinstance(x, torch.Tensor) else np.empty_like(x, dtype=np.float32)
    )

def make_divisible(x:int, divisor:int|torch.Tensor):
    """
    Return the nearest number that is divisible by the given divisor
    Args:
        x (int): The number to make divisible
        divisor (int|torch.Tensor): The divisor
    Returns:
        (int): The nearest number divisible by the divisor
    """
    if isinstance(divisor, torch.Tensor): divisor=int(divisor.max()) # to int
    return math.ceil(x/divisor)*divisor

def xyxy2xywh(x):
    """
    Convert bounding box coordinates from (x1,y1,x2,y2) format to (x,y,width,height) format where (x1,y1) is the 
    top-left corner and (x2,y2) is the bottom right corner
    Args:
        x (np.ndarray|torch.Tensor): Input bounding box coordinates in (x1,y1,x2,y2) format
    Returns:
        (np.ndarray|torch.Tensor): Bounding box coordinates in (x,y,width,height) format
    """ 
    assert x.shape[-1]==4, f'input shape last dimension expected 4 but input shape is {x.shape}'
    y=empty_like(x) # faster than clone/copy
    x1,y1,x2,y2=x[...,0],x[...,1],x[...,2],x[...,3]
    y[...,0]=(x1+x2)/2 # x center
    y[...,1]=(y1+y2)/2 # y center
    y[...,2]=x2-x1 # width
    y[...,3]=y2-y1 # height
    return y
    
def xywh2xyxy(x):
    """Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is
    the top-left corner and (x2, y2) is the bottom-right corner. Note: ops per 2 channels faster than per channel.

    Args:
        x (np.ndarray | torch.Tensor): Input bounding box coordinates in (x, y, width, height) format.

    Returns:
        (np.ndarray | torch.Tensor): Bounding box coordinates in (x1, y1, x2, y2) format.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = empty_like(x)  # faster than clone/copy
    xy = x[..., :2]  # centers
    wh = x[..., 2:] / 2  # half width-height
    y[..., :2] = xy - wh  # top left xy
    y[..., 2:] = xy + wh  # bottom right xy
    return y

def xyxy2xywhn(x, w:int=640, h:int=640, clip:bool=False, eps:float=0.0):
    """
    Convert bounding box coordinates from (x1,y1,x2,y2) format to (x,y,w,h) normalized format, in which x, y, width,
    and height are normalized based on image size
    Args:
        x (np.ndarray | torch.Tensor): Input bounding box coordinates in (x1, y1, x2, y2) format
        w (int): Image width in pixels
        h (int): Image height in pixels
        clip (bool): Whether to clip boxes to image boundaries
        eps (float): Minimum value for box width and height
    Returns:
        (np.ndarray | torch.Tensor): Normalized bounding box coordinates in (x, y, width, height) format
    """
    if clip: x=clip_boxes(x, (h-eps, w-eps))
    assert x.shape[-1]==4, f'input shape last dimension expected to be 4 but got {x.shape}'
    y=empty_like(x) # faster than clone/copy
    x1,y1,x2,y2=x[...,0],x[...,1],x[...,2],x[...,3]
    y[...,0]=((x1+x2)/2)/w # normalized x center
    y[...,1]=((y1+y2)/2)/h # normalized y center
    y[...,2]=(x2-x1)/w # normalized width
    y[...,3]=(y2-y1)/h # normalized height
    return y

def xyxy2ltwh(x):
    """Convert bounding boxes from [x1,y1,x2,y2] to [x1, y1, w, h] format

    Args:
        x (np.ndarray | torch.Tensor): Input bounding box coordinates in xyxy format, of size (*,4)
    Returns:
        (np.ndarray | torch.Tensor): Bounding box coordinates in ltwh format, of size (*,4)
    """  
    y=x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[...,2]=x[...,2]-x[...,0] # width
    y[...,3]=x[...,3]-x[...,1] # height
    return y

def xywh2ltwh(x):
    """Convert bounding box format from [x,y,w,h] to [x1,y1,w,h] where x1, y1 are left-top coordinates

    Args:
        x (np.ndarray | torch.Tensor): Input bounding box coordinates in xywh format, of size (*, 4)
    Returns:
        (np.ndarray | torch.Tensor): Bounding box coordinates in ltwh format
    """ 
    y=x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[...,0]=x[...,0]-(x[...,2]/2) # left top x
    y[...,1]=x[...,1]-(x[...,3]/2) # top left y
    return y

def ltwh2xyxy(x):
    """Convert bounding box from [x1, y1, w, h] to [x1, y1, x2, y2] where x1,y1 is left-top and x2,y2 is right-bottom
    Args:
        x (np.ndarray | torch.Tensor): Input bounding box coordinates, of size (*, 4)
    Returns:
        (np.ndarray | torch.Tensor): Bounding box coordinates in xyxy format, of size (*,4)
    """
    y=x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[...,2]=x[...,0]+x[...,2] # left-top x + width
    y[...,3]=x[...,1]+x[...,3] # left-top y + height
    return y

def ltwh2xywh(x):
    """Convert bounding boxes from [x1,y1,w,h] to [x,y,w,h] where x1,y1 is left top and x,y is center
    Args:
        x (np.ndarray|torch.Tensor): Input bounding box coordinates, of size (*,4)
    Returns:
        (np.ndarray | torch.Tensor): Bounding box coordinates in xywh format, of size (*,4)
    """
    y=x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[...,0]=x[...,0]+(x[...,2]/2) # left top x + half width
    y[...,1]=x[...,1]+(x[...,3]/2) # left top y + half height
    return y
def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding:bool=True, xywh:bool=False):
    """
    Rescale bounding boxes from one image shape to another
    Rescale bounding boxes from img1_shape to img0_shape, accounting for padding and aspect ratio changes. Support both
    xyxy and xywh box formats
    Args:
        img1_shape (tuple): Shape of the source image (height, width)
        boxes (torch.Tensor): Bounding boxes to rescale in format (N,4)
        img0_shape (tuple): Shape of the target image (height, width)
        ratio_pad (tuple, optional): Tuple of (ratio,pad) for scaling. If None, calculate from the image shapes
        padding (bool): Whether boxes are based on YOLO-style augmented imnages with padding
        xywh (bool): Whether box format is xywh (True) or xyxy (False)
    Returns:
        (torch.Tensor): Rescaled bounding boxes in the same format as input
    """
    if ratio_pad is None: # calculate from img0_shape
        gain=min(img1_shape[0]/img0_shape[0], img1_shape[1]/img0_shape[1]) # gain=old/new
        pad_x=round((img1_shape[1]-img0_shape[1]*gain)/2 - 0.1)
        pad_y=round((img1_shape[0]-img0_shape[0]*gain)/2 - 0.1)
    else:
        gain=ratio_pad[0][0]
        pad_x, pad_y=ratio_pad[1]
    if padding:
        boxes[...,0]-=pad_x
        boxes[...,1]-=pad_y
        if not xywh:
            boxes[...,2]-=pad_x
            boxes[...,3]-=pad_y
    boxes[...,:4]/=gain
    return boxes if xywh else clip_boxes(boxes, img0_shape)

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None, normalize:bool=False, padding:bool=True):
    """ Recale segment coodinates from img1_shape to img0_shape
    Args:
        img1_shape (tuple): Source image shape as HWC or HW (support both)
        coords (torch.Tensor): Coordinates to scale with shape (N, 2) where N is the number of coordinate tuples, and 2 for (x,y)
        img0_shape (tuple): Target image shape as HWC or HW (support both)
        ratio_pad (tuple, optional): Ratio and padding values as ((ratio_h, ratio_w), (pad_w, pad_h))
        normalize (bool): Whether to normalize coordinates to range [0,1]
        padding (bool): Wehther coordinates are based on YOLO-style augmented images with padding
    Returns:
        (torch.Tensor): Scaled coordinates
    """ 
    img0_h, img0_w=img0_shape[:2] # support both HWC or HW shapes
    if ratio_pad is None: # calculate from img0_shape
        img1_h, img1_w=img1_shape[:2] # support both HWC or HW shapes
        gain=min(img1_h/img0_h, img1_w/img0_w) # gain=source/target
        pad=(img1_w-img0_w*gain)/2, (img1_h-img0_h*gain)/2 # wh padding
    else:
        gain=ratio_pad[0][0]
        pad=ratio_pad[1]
        
    if padding:
        coords[...,0]-=pad[0] # x padding
        coords[...,1]-=pad[1] # y padding
    coords[...,0]/=gain
    coords[...,1]/=gain
    coords=clip_coords(coords, img0_shape)
    if normalize:
        coords[...,0]/=img0_w # width
        coords[...,1]/=img0_h # height
    return coords

def clip_boxes(boxes, shape):
    """
    Clip bounding boxes to image boundaries
    Args:
        boxes (torch.Tensor | np.ndarray): Bounding boxes to clip
        shape (tuple): Image shape as HWC or HW (supports both)
    Returns:
        (torch.Tensor | np.ndarray): Clipped bounding boxes
    """
    h, w=shape[:2] # supports both HWC or HW shapes
    if isinstance(boxes, torch.Tensor): # faster individually
        if NOT_MACOS14:
            boxes[...,0].clamp_(0,w) # x1
            boxes[...,1].clamp_(0,h) # y1
            boxes[...,2].clamp_(0,w) # x2
            boxes[...,3].clamp_(0,h) # y2
        else: # Apple macOS14 MPS bug https://github.com/ultralytics/ultralytics/pull/21878
            boxes[...,0]=boxes[...,0].clamp(0,w) # x1
            boxes[...,1]=boxes[...,1].clamp(0,h) # y1
            boxes[...,2]=boxes[...,2].clamp(0,w) # x2
            boxes[...,3]=boxes[...,3].clamp(0,h) # y2
    else: # np.ndarray faster as a group
        boxes[...,[0,2]]=boxes[...,[0,2]].clip(0,w) # x1, x2
        boxes[...,[1,3]]=boxes[...,[1,3]].clip(0,h) # y1, y2
        
    return boxes

def clip_coords(coords, shape):
    """Clip line coordinates to image boundaries
    
    Args:
        coords (torch.Tensor | np.ndarray): Line coordinates of size (N,2) to clip
        shape (tuple): Image shape as HWC or HW (support both)
    Returns:
        (torch.Tensor | np.ndarray): Clipped coordinates
    """
    h, w=shape[:2] # support both HWC or HW shapes
    if isinstance(coords, torch.Tensor):
        if NOT_MACOS14:
            coords[...,0].clamp_(0, w) # x
            coords[...,1].clamp_(0, h) # y
        else: # Apple macOS14 MPS bug https://github.com/ultralytics/ultralytics/pull/21878
            coords[...,0]=coords[...,0].clamp(0,w)
            coords[...,1]=coords[...,1].clamp(0,h)
    else: # np.ndarray
        coords[...,0]=coords[...,0].clip(0,w) # x
        coords[...,1]=coords[...,1].clip(0,h) # y
    return coords

def segments2boxes(segments):
    """Convert segment labels to box labels, i.e., [ [x1,y1], [x2,y2],...[xn,yn]] to xywh
    Args:
        segments (list[np.ndarray]): List of segments where each segment is a Nx2 array of points for N points and [x,y] coordinates
    Returns:
        (np.ndarray): Bounding box coordinates in xywh format of shape (N,4)
    """
    boxes=[]
    for s in segments:
        x,y=s.T # segment x,y
        boxes.append([x.min(), y.min(), x.max(), y.max()]) # xyxy
    return xyxy2xywh(np.array(boxes))

def resample_segments(segments, n:int=1000):
    """Resample segments to n points each using linear interpolation
    Args:
        segments (list): List of (N, 2) arrays where N is the number of points in each segment
        n (int): Number of points to resample each segment to
    Returns:
        (list): Resampled segments with n points each
    """
    for i, s in enumerate(segments):
        if len(s)==n: continue
        s=np.concatenate((s,s[0:1,:]), axis=0)
        x=np.linspace(0, len(s)-1, n-len(s) if len(s)<n else n)
        xp=np.arange(len(s))
        x=np.insert(x, np.searchsorted(x, xp), xp) if len(s)<n else x
        segments[i]=(np.concatenate([np.interp(x, xp, s[:,i]) for i in range(2)], dtype=np.float32).reshape(2,-1).T) # segment xy
    return segments
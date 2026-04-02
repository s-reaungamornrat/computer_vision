import numbers

import cv2
import PIL
import numpy as np

import torch


def get_resize_sizes(im_h, im_w, size):
    """Get resize height and width such that the shortest side matches `size`. Thus, the function maintains aspect ratio
    Args:
        im_h (int): Image height
        im_w (int): Image width
        size (int): Desired shortest size of height or width
    Returns:
        (tuple[int, int]): Desired height and width
    """
    if im_w<im_h:
        ow=size
        oh=int(size*im_h/im_w)
    else:
        oh=size
        ow=int(size*im_w/im_h)
    return oh, ow

def resize_clip(clip, size, interpolation='bilinear'):
    """
    The function resizes the clip so that the shortest size matches `size` while maintaining the aspect ratio. It does not change the data type of the clip
    Args:
        clip (np.ndarray| list[np.ndarray] | list[PIL.Image.Image]): Video clips of shape (T,H,W,C) where T is the number of frames and C is the number 
            of channels, or list of (H,W,C) ndarray video frames or list of PIL.Image.Image video frames
        size (tuple[int,int] | int): Desired size (width, height) or single value for both
        interpolation (str): Interpolation method with choices of 'nearest', 'bilinear'. Default to 'nearest'
    Returns:
        (list[np.ndarray] | list[PIL.Image.Image]): List of resized (H,W,C) ndarray video frames or PIL.Image.Image video frames
    """
    assert isinstance(clip[0], np.ndarray) or isinstance(clip[0], PIL.Image.Image), ("clip must be a sequence of np.ndarray or PIL.Image.Image, " 
                                                                                     f"but got {type(clip[0])}")
    
    if isinstance(size, numbers.Number):
        if isinstance(clip[0], np.ndarray): im_h, im_w, _=clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image): im_w, im_h=clip[0].size
        # minimum spatial dimension already matches the desired minimum size
        if (im_w<=im_h and im_w==size) or (im_h<=im_w and im_h==size): return clip
        new_h, new_w=get_resize_sizes(im_h, im_w, size)
        size=(new_w, new_h)
        
    if isinstance(clip[0], np.ndarray):
        interp=cv2.INTER_LINEAR if interpolation=='bilinear' else cv2.INTER_NEAREST
        scaled=[cv2.resize(img, size, interpolation=interp) for img in clip]
    elif isinstance(clip[0], PIL.Image.Image):
        interp=PIL.Image.BILINEAR if interpolation=='bilinear' else PIL.Image.NEAREST
        scaled=[img.resize(size, interp) for img in clip]
    return scaled

def crop_clip(clip, y0, x0, h, w):
    """Crop images in the clip based on the top-left corner location and the crop height and width
    Args:
        clip (np.ndarray| list[np.ndarray] | list[PIL.Image.Image]): Video clips of shape (T,H,W,C) where T is the number of frames and C is the number 
            of channels, or list of (H,W,C) ndarray video frames or list of PIL.Image.Image video frames
        y0 (int): Top coordinate index
        x0 (int): Left coordinate index
        h (int): Height of the cropped region
        w (int): Width of the cropped region
    Returns:
        (list[np.ndarray] | list[PIL.Image.Image]): List of cropped (H,W,C) ndarray video frames or PIL.Image.Image video frames
    """
    if isinstance(clip[0], np.ndarray):
        cropped=[img[y0:y0+h, x0:x0+w] for img in clip]
    elif isinstance(clip[0], PIL.Image.Image):
        cropped=[img.crop((x0, y0, x0+w, y0+h)) for img in clip]
    else:
        raise TypeError(f"Expected a list of np.ndarray or PIL.Image.Image, but got a list of {type(clip[0])}")
    return cropped

def normalize(clip, mean, std):
    """
    Args:
        clip (torch.Tensor): Tensor of shape (C,T,H,W)
        mean (tuple[float]): Mean pixel value per channel
        std (tuple[float]): Standard deviation value per channel
    Returns:
        (torch.Tensor): Normalized tensor of shape (C,T,H,W)
    """
    assert torch.is_tensor(clip) and clip.ndim==4 
    
    dtype=clip.dtype
    mean=torch.tensor(mean, dtype=dtype, device=clip.device).view(clip.shape[0], *(1,)*(clip.ndim-1))
    std=torch.tensor(std, dtype=dtype, device=clip.device).view(clip.shape[0], *(1,)*(clip.ndim-1))
    return (clip-mean)/std
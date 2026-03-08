from __future__ import annotations
from typing import Optional, Union

import numbers

import cv2
import numpy as np

cv2_interp_codes = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4
}

def _scale_size(size:tuple[int,int], scale:Union[float, int, tuple[int,int]])->tuple[int,int]:
    """Rescale a size by a ratio

    Args:
        size (tuple[int]): Image size (w,h)
        scale (float | int | tuple[float] | tuple[int]): Scaling factor for width and height respectively
    Returns:
        (tuple[int]): Scaled size
    """
    if isinstance(scale, numbers.Number): scale=(scale, scale)
    w,h=size
    return int(w*float(scale[0])+0.5), int(h*float(scale[1])+0.5)
    
def rescale_size(old_size:tuple, scale:Union[float, int, tuple[int,int]], return_scale:bool=False)->tuple:
    """Calculate the new size to be rescaled to 
    Args:
        old_size (tuple[int]): The original size (width, height) of the image
        scale (float | int | tuple[int, int]): The scaling factor or maximum size. If it is a float or an integer, the image will be rescaled by this 
            factor; else if it is a tuple of 2 integers, the image will be rescaled as large as possible within the scale. 
        return_scale (bool): Whether to return the scaling factor besides the rescaled image size
    Returns:
        (tuple[int]): The new rescaled image size
    """
    w, h=old_size
    if isinstance(scale, (float, int)):
        assert scale>0, f"Invalid scale {scale}, must be positive"
        scale_factor=scale
    elif isinstance(scale, tuple):
        max_long_edge=max(scale)
        max_short_edge=min(scale)
        scale_factor=min(max_long_edge/max(h, w), max_short_edge/min(h,w))
    else: raise TypeError(f"scale must be a number or tuple of int, but got {type(scale)}")

    new_size=_scale_size((w,h), scale_factor)
    if return_scale: return new_size, scale_factor
    return new_size

def imresize(img:np.ndarray, size:tuple[int, int], return_scale:bool=False, interpolation:str='bilinear')->Union[tuple[np.ndarray, float, float], np.ndarray]:
    """Resize image to a given size
    Args:
        img (np.ndarray): Input image
        size (tuple[int]): Target size (w, h)
        return_scale (bool): Whether to return `w_scale` and `h_scale`
        interpolation (str): Interpolation method, with options of 'nearest', 'bilinear', 'bicubic', 'area', 'lanczos'
    Returns:
        (tuple[np.ndarray, float, float] | np.ndarray): (`resized_img`, `w_scale`, `h_scale`) or `resized_img`
    """
    h, w=img.shape[:2]
    resized_img=cv2.resize(img, size, interpolation=cv2_interp_codes[interpolation])
    
    if not return_scale: return resized_img

    w_scale=size[0]/w
    h_scale=size[1]/h
    return resized_img, w_scale, h_scale

def imflip(img:np.ndarray, direction:str='horizontal')->np.ndarray:
    """Inplace flip an image horizontally or vertically
    Args:
        img (np.ndarray): Image to be flipped
        direction (str): The flip direction, either 'horizontal' or 'vertical' or 'diagonal'
    Returns:
        (np.ndarray): The flipped image 
    """
    assert direction in ['horizontal', 'vertical', 'diagonal']
    if direction=='horizontal': return cv2.flip(img, 1)#, img) # originall inplace
    elif direction=='vertical': return cv2.flip(img, 0)#, img)
    return cv2.flip(img, -1)#, img)
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import ast
import math
import warnings

import cv2
import numpy as np
import torch

def check_imgsz(imgsz, stride=32, min_dim=1, max_dim=2, floor=0):
    """Verify image size is a multiple of the given stride in each dimension. If the image size is not a multiple of the stride, update
    it to the nearest multiple of the stride that is greater than or equal to the given floor value.

    Args:
        imgsz (int| list[int]): Image size
        stride (int): Stride value
        min_dim (int): Minimum number of dimensions
        max_dim (int): Maximum number of dimensions
        floor (int): Minimum allowed value for image size
    Returns:
        (list[int] | int): Updated image size
    """
    # Convert stride to integer if it is a tensor
    stride = int(stride.max() if isinstance(stride, torch.Tensor) else stride)

    # Convert image size to list if it is an integer
    if isinstance(imgsz, int): imgsz=[imgsz]
    elif isinstance(imgsz, (tuple,list)): imgsz=list(imgsz)
    elif isinstance(imgsz, str): imgsz=[int(imgsz)] if imgsz.isnumeric() else ast.literal_eval(imgsz)
    else:
        raise TypeError(f'imgsz={imgsz} is of invalid type {type(imgsz).__name__}'
                        f'Valid imgsz types are int `imgsz=640` or list `imgsz=[640,640]`')

    # Apply max_dim
    if len(imgsz)>max_dim:
        msg=("'train' and 'val' imgsz must be an integer, while 'predict' and 'export' imgsz may be a [h, w] list"
             "or an integer, i.e., 'yolo export imgsz=640,480' or 'yolo export imgsz=640'")
        if max_dim!=1: raise ValueError(f'imgsz={imgsz} is not a valid image size. {msg}')
        warnings.warn(f'updating to "imgsz={max(imgsz)}". {msg}')
        imgsz=[max(imgsz)]

    # Make image size a multiple of the stride
    sz=[max(math.ceil(x/stride)*stride, floor) for x in imgsz]

    # Print warning message if image size was updated
    if sz!=imgsz: warnings.warn(f'imgsz={imgsz} must be multiple of max stride {stride}, updating it to {sz}')

    # Add missing dimensions if necessary
    sz=[sz[0], sz[0]] if min_dim==2 and len(sz)==1 else sz[0] if min_dim==1 and len(sz)==1 else sz

    return sz
    
        
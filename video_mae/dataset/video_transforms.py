import math
import numbers
import random

import PIL
import numpy as np

from PIL import Image

from torchvision import transforms

from .rand_augment import rand_augment_transform

def _pil_interp(method):
    if method=='bicubic': return Image.BICUBIC
    elif method=='lanczos': return Image.LANCZOS
    elif method=='hamming': return Image.HAMMING
    else: return Image.BILINEAR

def create_random_augment(input_size, auto_augment=None, interpolation='bilinear'):
    """Get video random augmentation 
    Args:
        input_size (tuple[int]|int): Desired size of input video as (height, width)
        auto_augment (str): Parameter for random augmentation. 
        interpolation (str): Interpolation method
    Returns:
        (torchvision.transforms): Image transform
    """
    img_size=input_size
    if isinstance(input_size, tuple): img_size=input_size[-2:]
    
    if auto_augment:
        assert isinstance(auto_augment, str)
        img_size_min=min(img_size) if isinstance(img_size, tuple) else img_size
        aa_params={'translate_const': int(img_size_min*0.45)}
        if interpolation and interpolation!='random': aa_params['interpolation']=_pil_interp(interpolation)
        if auto_augment.startswith('rand'): return transforms.Compose([rand_augment_transform(auto_augment, aa_params)])
    raise NotImplementedError
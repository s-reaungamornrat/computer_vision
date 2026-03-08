from __future__ import annotations
from typing import Optional, Union

import numpy as np

def iminvert(img:np.ndarray):
    """Invert (negate) an image

    Args:
        img (np.ndarray): Image to be inverted of type uint8 with shape (H, W, C)
    Returns:
        (np.ndarray): The inverted image
    """
    assert np.issubdtype(img.dtype, np.uint8), f"Image must be of type uint8 but got {img.dtype}"
    return np.full_like(img, 255)-img
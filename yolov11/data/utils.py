from __future__ import annotations

import cv2
import numpy as np

def imread(filename: str, flags: int=cv2.IMREAD_COLOR)->np.ndarray | None:
    """
    Read an image from a file with multilanguage filename support
    Args:
        filename (str)L Path to the file
        flags (int, optional): Flag that can take values of cv2.IMREAD_*. Control how the image is read
    Returns:
        (np.ndarray | None): The read image array, or None if reading fails. If a color image (cv2.IMREAD_COLOR), return BGR
    Examples:
        >>> img=imread("path/to/image.jpg")
        >>> img=imread("path/to/image.jpg", cv2.IMREAD_GRAYSCALE)
    Source: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/patches.py
    """
    file_bytes=np.fromfile(filename, np.uint8)

    if filename.endswith((".tiff", ".tif")):
        success, frames=cv2.imdecodemulti(file_bytes, cv2.IMREAD_UNCHANGED)
        if success:
            # Handle RGB images in tif/tiff format
            return frames[0] if len(frames)==1 and frames[0].ndim==3 else np.stack(frames, axis=2)
        return None
    else:
        im=cv2.imdecode(file_bytes, flags)
        return im[...,None] if im is not None and im.ndim==2 else im # always ensure 3 dimensions
    
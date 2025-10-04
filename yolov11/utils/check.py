import math 
import torch

def check_imgsz(imgsz, stride=32, min_dim=1, max_dim=2, floor=0):
    """
    Verify image size is a multiple of the given stride in each dimension. If the image size is not a multiple of the
    stride, update it to the nearest multiple of the stride that is greater than or equal to the given floor value
    Args:
        imgsz (int| list[int]): Image size
        stride (int): Stride value
        min_dim (int): Mininum number of dimensions
        max_dim (int): Maximum number of dimensions
        floor (int): Minimum allowed value for image size
    Returns:
        (list[int] | int): Updated image size
    """
    # Convert stride to integer if it is a tensor
    stride=int(stride.max() if isinstance(stride, torch.Tensor) else stride)

    # Convert image size to list if it is an integer
    if isinstance(imgsz, int): imgsz=[imgsz]
    elif isinstance(imgsz, (tuple, list)): imgsz=list(imgsz)
    else: 
        raise TypeError(f"imgsz={imgsz} is of invalid type {type(imgsz).__name__}. Valid imgsz types are int or list")

    # Apply max_dim
    if len(imgsz)>max_dim: imgsz=[max(imgsz)]
    # Make image size a multiple of the stride
    sz=[max(math.ceil(x/stride)*stride, floor) for x in imgsz]

    # Add missing dimensions if necessary
    sz=[sz[0], sz[0]] if min_dim==2 and len(sz)==1 else sz[0] if min_dim==1 and len(sz)==1 else sz

    return sz
        
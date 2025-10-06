import torch
import numpy as np

def rescale(img, out_min=0., out_max=1.):
    '''
    Rescale image intensity
    Args:
        img (Tensor/ndarray): HxWxC or CxHxW
    Returns:
        img (Tensor/ndarray): HxWxC or CxHxW
    '''
    in_min, in_max=img.min(), img.max()
    return (img-in_min)*(out_max-out_min)/(in_max-in_min) + out_min

def alpha_bending(foreground, background, alpha):
    '''
    Args:
        foreground (Tensor | ndarray): NxCxHxW image tensor or HxWxC array to be on the foreground
        background (Tensor | ndarray): NxCxHxW image tensor or HxWxC array to be on the background
        alpha (float): opacity of the foreground
    Returns:
        image (Tensor | ndarray): NxCxHxW  alpha-blended image  tensor or HxWxC array
    '''
    if all(isinstance(x, torch.Tensor) for x in [foreground, background]):
        assert all(x.ndim==4 for x in [foreground, background])
        background=background.to(torch.float32)
        foreground=foreground.to(torch.float32)
        C=max(background.shape[1], foreground.shape[1])
        
        foreground=rescale(foreground, out_max=255.)
        if foreground.shape[1]!=C: 
            foreground=torch.cat([foreground, torch.zeros_like(foreground), foreground], dim=1) # NxCxHxW where C is RBG channels
        if background.shape[1]!=C: 
            background=torch.cat([background, background, background], dim=1) # NxCxHxW where C is RBG channels
        
        return (background*(1.-alpha)+foreground*alpha).to(dtype=torch.uint8)
    
    assert all(isinstance(x, np.ndarray) for x in [foreground, background])
    assert all(x.ndim==3 for x in [foreground, background])
    background=background.astype(float)
    foreground=foreground.astype(float)
    C=max(background.shape[-1], foreground.shape[-1])
        
    foreground=rescale(foreground, out_max=255.)
    if foreground.shape[-1]!=C: 
        foreground=np.concatenate([foreground, np.zeros_like(foreground), foreground], axis=2) # HxWxC where C is RBG channels
    if background.shape[-1]!=C: 
        background=np.concatenate([background, background, background], axis=2) # HxWxC where C is RBG channels
    
    return (background*(1.-alpha)+foreground*alpha).astype(np.uint8)
        
def find_free_network_port() -> int:
    """
    Find a free port on localhost, mainly for distributed training

    It is useful in single-node training when we don't want to connect to a real main node but have to set the
    `MASTER_PORT` environment variable.

    Returns:
        (int): The available network port number.
        
    Source: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/dist.py#L12
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]  # port
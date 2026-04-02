import math
import random

import torch
import numpy as np

class RandomErasing:
    """Randomly selects a rectangle region in an image and erases its pixels. 
    'Random Erasing Data Augmentation' by Zhong et al. https://arxiv.org/pdf/1708.04896.pdf
    This variant of RandomErasing is intended to be applied to either a batch or single image tensor after it has been normalized by dataset mean and std.

    Args:
        probability (float): Probability that the Random Erasing operation will be performed
        min_area (float): Minimum percentage of erased area with respect to input image area
        max_area (float): Maximum percentage of erased area with respect to input image area
        min_aspect (float): Minimum aspect ratio of erased area
        mode (str): Pixel color mode, one of 'const', 'rand', or 'pixel'
            'const' - erased block is constant color of 0 for all channels
            'rand' - erased block is same per-channel random (normal) color
            'pixel' - erased block is per-pixel random (normal) color
        max_count (int): Maximum number of erasing blocks per image, area per box is scaled by count.
            per-image count is randomly chosen between 1 and this value
    """
    def __init__(self, probability=0.5, min_area=0.02, max_area=1/3, min_aspect=0.3, max_aspect=None, mode='const', min_count=1, max_count=None, num_splits=0,
                 device='cuda', cube=True):
        self.probability=probability
        self.min_area=min_area
        self.max_area=max_area
        max_aspect=max_aspect or 1/min_aspect
        self.log_aspect_ratio=(math.log(min_aspect), math.log(max_aspect))
        self.min_count=min_count
        self.max_count=max_count or min_count
        self.num_splits=num_splits
        mode=mode.lower()
        self.rand_color=False
        self.per_pixel=False
        self.cube=cube
        if mode=='rand': self.rand_color=True # per block random normal
        elif mode=='pixel': self.per_pixel=True # per pixel random normal
        else: assert not mode or mode=='const'
        self.device=device

    def _erase_cube(self, img, batch_start):
        """In-place erase of a region (cube patch) in a video.
        Args:
            img (torch.Tensor): Video frames of shape (T,C,H,W) where T is the number of frames, C is the number of channels
            batch_start (int): Start index of batch images whose regions will be erased 
        """        
        if np.random.random()>self.probability: return 

        dtype=img.dtype
        batch_size,channels,img_h, img_w=img.shape
        
        area=img_h*img_w
        count=(self.min_count if self.min_count==self.max_count else np.random.randint(self.min_count, self.max_count)) # number of removed regions
    
        for _ in range(count):
            for _ in range(100): # try to find blocks/cubes to erase 100 times
                # area of erased blocks is scaled by count--> the larger count, the smaller area
                target_area=np.random.uniform(self.min_area, self.max_area)*area/count
                aspect_ratio=math.exp(np.random.uniform(*self.log_aspect_ratio))
                w=int(round(math.sqrt(target_area*aspect_ratio))) # aspect_ratio is w/h
                h=int(round(math.sqrt(target_area/aspect_ratio)))
                if w<img_w and h<img_h:
                    top=np.random.randint(0, img_h-h)
                    left=np.random.randint(0, img_w-w)
                    for i in range(batch_start, batch_size):
                        img[i,:, top:top+h, left:left+w]=_get_pixels(self.per_pixel, self.rand_color, (channels, h, w), 
                                                                   dtype=dtype, device=self.device) # (C,H,W)
                    break

    def _erase(self, img, bidx):
        """In-place erase of a region (cube patch) in a video.
        Args:
            img (torch.Tensor): Video frames of shape (T,C,H,W) where T is the number of frames, C is the number of channels
            bidx (int): Index of batch image/item whose regions will be erased 
        """
        if np.random.random()>self.probability: return

        dtype=img.dtype
        _, channels, img_h, img_w = img.shape

        area=img_h*img_w
        count=self.min_count if self.min_count==self.max_count else np.random.randint(self.min_count, self.max_count)
        for _ in range(count):
            for _ in range(10):  # try to find blocks/cubes to erase 10 times
                # area of erased blocks is scaled by count--> the larger count, the smaller area
                target_area=np.random.uniform(self.min_area, self.max_area)*area/count
                aspect_ratio=math.exp(np.random.uniform(*self.log_aspect_ratio))
                w=int(round(math.sqrt(target_area*aspect_ratio))) # aspect_ratio is w/h
                h=int(round(math.sqrt(target_area/aspect_ratio)))
                if w<img_w and h<img_h:
                    top=np.random.randint(0, img_h-h)
                    left=np.random.randint(0, img_w-w)
                    img[bidx, :, top:top+h, left:left+w]=_get_pixels(self.per_pixel, self.rand_color, (channels, h, w), dtype=dtype,
                                                                     device=self.device)
                    break
                        
    def __call__(self, input):
        """ Erasing parts of videos/images
        Args:
            input (torch.Tensor): Image tensor of shape (T,C,H,W)
        """
        if input.ndim==3: # (C,H,W)
            input=input.unsqueeze(0) # (1,C,H,W)
            self._erase(input, 0)
            return input.squeeze(0)  # (C,H,W)
        
        batch_size=input.shape[0]
        # skip first slice of batch if num_splits is set (for clean portion of samples)
        batch_start=batch_size//self.num_splits if self.num_splits>1 else 0
        if self.cube:
            self._erase_cube(input, batch_start)
            return input
        
        for i in range(batch_start, batch_size): self._erase(input, i)
        return input
        
def _get_pixels(per_pixel, rand_color, patch_size, dtype=torch.float32, device='cuda'):
    """ Generate the pixel values to fill an erased region (patch) in an image or video.
    
    Depending on the configuration, this returns either high-variance noise, a single random color, or a constant value
    Args:
        per_pixel (bool): If True, generate unique random noise for each pixel in the patch by drawing from a normal distribution
        rand_color (bool): If True, generate a single random color for the entire patch with unique values per channel
        patch_size (tuple[int, int, int]): Size of patch to be filled, typically of shape (C, H, W)
        dtype (torch.dtype, optional): The desired data type of the returned tensor. Default to torch.float32
        device (str|torch.device, optional): The device on which to allocate the tensor. Default to 'cuda'
    Returns:
        (torch.Tensor): A tensor containing the filled values
            - If `per_pixel`: Shape is `patch_size`
            - If `rand_color`: Shape id `(C,1,1)`
            - Otherwise: A zero-filled tensor of shape `(C,1,1)`
    """
    # Note: I've seen CUDA illegal memory access errors being caused by the normal_() paths, flip the order so normal is run on CPU if this becomes a problem. 
    # Issue has been fixed in master https://github.com/pytorch/pytorch/issues/19508
    if per_pixel: return torch.empty(patch_size, dtype=dtype, device=device).normal_()
    elif rand_color: return torch.empty((patch_size[0], 1, 1), dtype=dtype, device=device).normal_()
    else: return torch.zeros((patch_size[0],1,1),dtype=dtype, device=device)
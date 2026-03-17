import math
import numbers
import warnings
import random

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image, ImageOps

class GroupMultiScaleCrop(object):
    """
    Crop and resize video frames 
    Args:
        input_size (tuple[int, int]): (Image width, image height)
        scales (list[float]): List of factors to multiply the original image size (minimum of width and height) to get the crop sizes to randomly
            sampled from
        max_distort (int) A threshold for spatial aspect ratio consistency. By constraining abs(i-j)<=max_distort, where i and j are indices into the available 
            scale list, the function filters out extreme height-width pairings. The high values allow for more varied rectangular stretching.
        fix_crop (bool): Whether to fixed regions to crop
        more_fix_crop (bool): Whether to get 13 cropped regions instead of 5 cropped regions
    Example:
        >>> aug_op=GroupMultiScaleCrop(input_size=224, scales=[1, .875, .75, .66])
        >>> ret_imgs, label=aug_op((images, None))
    """
    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
        
        assert max_distort>0
        
        self.scales=scales if scales is not None else [1, .875, .75, .66]
        self.max_distort=max_distort
        self.fix_crop=fix_crop
        self.more_fix_crop=more_fix_crop
        self.input_size=input_size if not isinstance(input_size, int) else [input_size, input_size]
        self.interpolation=Image.BILINEAR

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        """ Determining deterministic multi-corner cropping, i.e., calculate a list of specific (w_offset, h_offset) coordinates 
        
        Create 5 or 13 fixed crops for validation/testing and for multi-view voting
    
        Args:
            more_fix_crop (bool): Whether to get 13 crops instead of 5 crops
            image_w (int): Width of image
            image_h (int): Height of image
            crop_w (int): Width of cropped region
            crop_h (int): Height of cropped region
        Returns:
            (list[tuple[int,int]]): List of tuple of start indices of each cropped region, i.e., indices of top-left corners of each
                cropped regions relative to the original image coordinates
        """
        w_step=(image_w-crop_w)//4
        h_step=(image_h-crop_h)//4
        
        ret=[(0,0), # upper left
             (4*w_step, 0), # upper right
             (0, 4*h_step), # lower left
             (4*w_step, 4*h_step), # lower right
             (2*w_step, 2*h_step) # center
            ]
        if more_fix_crop:
            ret.extend([
                (0,2*h_step), # center left
                (4*w_step,2*h_step), # center right
                (2*w_step,4*h_step), # lower center
                (2*w_step,0*h_step) # upper center
            ])
            ret.extend([
                (1*w_step, 1*h_step), # upper left quater
                (3*w_step, 1*h_step), # upper right quater
                (1*w_step, 3*h_step), # lower left quater
                (3*w_step, 3*h_step) # lower right quater
            ])
        return ret

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        """Determine `w_offset` and `h_offset` which are the top-left coordinates defining where the cropping box begins on the original image
        Args:
            image_w (int): Width of image
            image_h (int): Height of image
            crop_w (int): Width of cropped region
            crop_h (int): Height of cropped region
        Returns:
            (tuple[int,int]): A tuple of (w_offset, h_offset), representing the horizontal distance from the left edge of the original image 
                to the left edge of the crop, and the vertical distance from the top edge of the original image to the top edge of the crop
        """
        offsets=self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    def _sample_crop_size(self, im_size):
        """
        Compute the size (width, height) of crop region and the start index of crop region relative to the original image coordinate
        Args:
            im_size (tuple[int,int]): A tuple of (width, height) of an image
        Returns:
            (tuple[int,int,int,int]): Crop region width, height and start index in the x and y directions.
        """
        image_w, image_h=im_size
        
        # find a crop size
        base_size=min(image_w, image_h)
        crop_sizes=[int(base_size*x) for x in self.scales]
        # If a calculated crop size is very close to the model's expected input size (e.g., crop size is 225 and the model wants 224),
        # it snaps it exactly to `input_size` to avoid unnecessary sub-pixel interpolation artifacts during resizing step
        crop_h=[self.input_size[1] if abs(x-self.input_size[1])<3 else x for x in crop_sizes]
        crop_w=[self.input_size[0] if abs(x-self.input_size[0])<3 else x for x in crop_sizes]
        
        pairs=[]
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                # index difference of at most max_distort to prevent crop from being too skinny or flat
                if abs(i-j)<=self.max_distort:  pairs.append((w,h))
        
        crop_pair=random.choice(pairs)
        # determine `w_offset` and `h_offset` which are the top-left coordinates defining where the cropping box begins on the original image
        if not self.fix_crop:
            # making sure the cropping box stays within the original image boundary. For example, if image is 100px and crop is 60px, the 
            # offset can be anywhere between 0 and 40
            w_offset=random.randint(0, image_w-crop_pair[0]) 
            h_offset=random.randint(0, image_h-crop_pair[1])
        else:
            w_offset, h_offset=self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])
        
        return crop_pair[0], crop_pair[1], w_offset, h_offset
    
    def __call__(self, img_tuple):
        """
        Args:
            img_tuple (tuple[list[PIL.Image], int]): A tuple of list of video frames and label
        Returns:
            (list[PIL.Image]): List of images after being cropped and resized to `input_size`
            (int): Label which has been passed through
        """
        img_group, label=img_tuple
        im_size=img_group[0].size # (width, height)
        
        crop_w, crop_h, offset_w, offset_h=self._sample_crop_size(im_size)
        crop_imgs=[img.crop((offset_w, offset_h, offset_w+crop_w, offset_h+crop_h)) for img in img_group]
        ret_imgs=[img.resize(self.input_size, self.interpolation) for img in crop_imgs] # input_size=(width, height)
        
        return ret_imgs, label

class Stack(object):
    """
    Args:
        roll (bool): Whether to reverse the order of the color channels
    """
    def __init__(self, roll=False):
        self.roll=roll
        
    def __call__(self, img_tuple):
        """
        Args:
            img_tuple (tuple[list[PIL.Image], int]): A tuple of list of video frames and label
        Returns:
            (np.ndarray): Stack of images of shape (H,W,T,C) or (H,W,C*T)
            (int): Label which has been passed through
        """
        img_group, label=img_tuple

        if img_group[0].mode=='L':
            # (H,W,C) -> (H,W,1,C) -> (H,W,T,C)
            return np.concatenate([np.expand_dims(x,2) for x in img_group], axis=2),label 
        elif img_group[0].mode=='RGB':
            # roll RGB to BGR then stack T of (H,W,C) to (H,W,C*T)
            if self.roll: return np.concatenate([np.array(x)[:,:,::-1] for x in img_group], axis=2), label
            return np.concatenate(img_group, axis=2), label # stack T of (H,W,C) to (H,W,C*T)

class ToTorchFormatTensor(object):
    """Convert PIL.Image (RGB) or numpy.ndarray (H,W,C*T) in the range [0,255] to a torch.FloatTensor of shape (C,H,W) in range
    [0., 1.]"""
    def __init__(self, div=True):
        
        self.div=div
        
    def __call__(self, img_tuple):
        """
        Args:
            (tuple[np.ndarray, int]): The first element is the stack of images of shape (H,W,T,C) or (H,W,C*T) and the second is the 
                label which has been passed through
        Returns:
            (torch.Tensor): (C*T,H,W) video/image frames of type float, ranging from [0,1] if `div` is True and [0,255] otherwise
            (int): Label which has been passed through
        """
        imgs, label=img_tuple
        if isinstance(imgs, np.ndarray):
            img=torch.from_numpy(imgs).permute(2,0,1).contiguous()
        else:
            img=torch.as_tensor(imgs.tobytes(), dtype=torch.uint8)
            img=img.view(imgs.size[1], imgs.size[0], len(imgs.mode))
            # from (H,W,C) to (C,H,W)
            img=img.permute(2,0,1).contiguous()
        return img.float().div(255.) if self.div else img.float(), label


class GroupNormalize(object):

    def __init__(self, mean, std):
        self.mean=mean
        self.std=std
        
    def __call__(self, tensor_tuple):
        """
        Args:
            (tuple[torch.Tensor,int]): The first is (C*T,H,W) video/image frames of type float, and the second is the label which has been passed through
        Returns:
            (torch.Tensor): (C*T,H,W) video/image frames of type float after normalized
            (int): Label which has been passed through
        """
        tensor, label=tensor_tuple
        rep_mean=torch.tensor( self.mean * (tensor.size()[0]//len(self.mean)) )
        rep_mean=rep_mean.view(rep_mean.shape[0], *(1,)*(tensor.ndim - 1))

        rep_std=torch.tensor( self.std * (tensor.size()[0]//len(self.std)) )
        rep_std=rep_std.view(rep_std.shape[0], *(1,)*(tensor.ndim-1))

        return (tensor-rep_mean)/rep_std, label
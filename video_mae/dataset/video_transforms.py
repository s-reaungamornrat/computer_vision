import math
import numbers
import random

import PIL
import torch
import numpy as np

from PIL import Image

from torchvision import transforms

from .rand_augment import rand_augment_transform
from .functional import resize_clip, crop_clip, normalize

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

def random_short_side_scale_jitter(images, min_size, max_size, boxes=None, inverse_uniform_sampling=False):
    """Perform a spatial short scale jittering on the given images and corresponding boxes. Resize the image so that its shortest side matches a 
    randomly sampled size, thus the function maintaining aspect ratio
    Args:
        images (torch.Tensor): Images to perform scale jittering. Dimension is (T,C,H,W)
        min_size (int): Minimum size to scale the frames
        max_size (int): Maximum size to scale the frames
        boxes (np.ndarray, optional): Corresponding boxes of images. Dimension is (`num_boxes`,4)
        inverse_uniform_sampling (bool): If True, sample uniformly in [1/max_scale, 1/min_scale] and take a reciprocal to get the scale. If False,
            take a uniform sample from [min_scale, max_scale]
    Returns:
        (torch.Tensor): The scaled images with dimension of (C,T,H,W)
        (np.ndarray | None): The scaled boxes with dimension of (`num_boxes`,4)
    """
    if inverse_uniform_sampling: size=int(round(1./np.random.uniform(1./max_size, 1./min_size)))
    else: size=int(round(np.random.uniform(min_size, max_size)))
    height, width=images.shape[-2:]
    if (width<=height and width==size) or (height<=width and height==size): return images, boxes
    
    new_width=new_height=size
    if width<height:
        new_height=int(math.floor((float(height)/width)*size))
        if boxes is not None: boxes=boxes*float(new_height)/height
    else:
        new_width=int(math.floor((float(width)/height)*size))
        if boxes is not None: boxes=boxes*float(new_width)/width
    return torch.nn.functional.interpolate(images,size=(new_height,new_width),mode='bilinear', align_corners=False), boxes

def crop_boxes(boxes, x_offset, y_offset):
    """Perform crop on the bounding boxes given the offsets
    Args:
        boxes (np.ndarray, optional): Bounding boxes of shape (`num_boxes`,4)
        x_offset (int): Cropping offset in the x direction
        y_offset (int): Cropping offset in the y direction
    Returns:
        (np.ndarray|None): Cropped boxes with dimension of (`num_boxes`,4)
    """
    cropped_boxes=boxes.copy()
    cropped_boxes[...,[0,2]]=cropped_boxes[...,[0,2]]-x_offset
    cropped_boxes[...,[1,3]]=cropped_boxes[...,[1,3]]-y_offset
    return cropped_boxes

def random_crop(images, size, boxes=None):
    """Perform random spatial crop on the given images and corresponding boxes
    Args:
        images (torch.Tensor): Image to perform random crop. The dimension is (T,C,H,W)
        size (int): The size of height and width to crop the images to
        boxes (np.ndarray, optional): Corresponding boxes of the images. Dimension is (`num_boxes`, 4)
    Returns:
        (torch.Tensor): Cropped images with dimension of (T,C,H,W)
        (np.ndarray|None): Cropped boxes with dimension of (`num_boxes`, 4)
    """
    height,width=images.shape[-2:]
    #if all(s==size for s in [height,width]): return images
    y_offset=int(np.random.randint(0, height-size)) if height>size else 0
    x_offset=int(np.random.randint(0, width-size)) if width>size else 0
    cropped=images[:,:,y_offset:y_offset+size, x_offset:x_offset+size]
    cropped_boxes=crop_boxes(boxes, x_offset=x_offset, y_offset=y_offset) if boxes is not None else None
    return cropped, cropped_boxes

def _get_param_spatial_crop(scale, ratio, height, width, num_repeat=10, log_scale=True, switch_hw=False):
    """Given scale, ratio, height and width, return sampled coordinates of the videos
    Args:
        scale (tuple[float, float]): Range of scale to resize images
        ratio (tuple[float, float]): Range of aspect ratio which is width/height
        height (int): Height of input images
        width (int): Width of input images
        num_repeat (int): Number of times to repeatedly compute random cropping coordinates including the coordinates of the top-left corner and
            the height and width of the to-be-cropped area
        log_scale (bool): Whether to compute random aspect ratio (width/height) in log-scale
        switch_hw (bool): Whether to switch between the output width and output height
    Returns:
        (int): Y-coordinate of the top corner
        (int): X-coordinate of the left corner
        (int): Height of the area
        (int): Width of the area
    """
    for _ in range(num_repeat):
        area=height*width
        target_area=np.random.uniform(*scale)*area
        if log_scale:
            log_ratio=(math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio=math.exp(np.random.uniform(*log_ratio))
        else: aspect_ratio=np.random.uniform(*ratio)
        w=int(round(math.sqrt(target_area*aspect_ratio))) # aspect_ratio is w/h
        h=int(round(math.sqrt(target_area/aspect_ratio)))
        if np.random.uniform()<0.5 and switch_hw: w,h=h,w
        if 0<w<width and 0<h<height:
            i=np.random.randint(0, height-h)
            j=np.random.randint(0, width-w)
            return i,j,h,w
    
    # Fall back to the central crop
    in_ratio=float(width)/float(height)
    if in_ratio<min(ratio): # min(ratio) focuses on those with height>width
        # in_ratio<min(ratio): large height or height >> width than the ratio expected
        w=width # width is shorter than height so we compute height using width
        h=int(round(w/min(ratio)))
    elif in_ratio>max(ratio): # max(ratio) focuses on large width
        # in_ratio>max(ratio): large large width
        h=height # height is shorter than width so we compute width using height
        w=int(round(h*max(ratio)))
    else: w,h=width, height # whole image
    i=(height-h)//2
    j=(width-w)//2
    return i,j,h,w

def random_resized_crop_with_shift(images, target_height, target_width, scale=(0.8,1.0), ratio=(3./4., 4./3.)):
    """This is similar to random_resize_crop, but it samples two different boxes (for cropping) for the first and last frames. It then
    linearly interpolates the two boxes for other frames
    Args:
        images (torch.Tensor): Images to be resized and cropped of shape (T,C,H,W)
        target_height (int): Desired height after cropping
        target_width (int): Desired width after cropping
        scale (tuple[float,float]): Scale range of inception-style area based random resizing
        ratio (tuple[float, float]): Aspect ratio range of intercept-style area based random resizing
    Returns:
        (torch.Tensor): Resized and cropped images of shape (T,C,H,W)
    """
    T,C, height, width=images.shape
    i,j,h,w=_get_param_spatial_crop(scale, ratio, height, width)
    i_, j_, h_, w_=_get_param_spatial_crop(scale, ratio, height, width)
    print(f"{i=}, {j=}, {h=}, {w=}")
    print(f"{i_=},{j_=},{h_=},{w_=}")
    i_s=[int(i) for i in torch.linspace(i,i_,steps=T).tolist()]
    j_s=[int(i) for i in torch.linspace(j,j_,steps=T).tolist()]
    h_s=[int(i) for i in torch.linspace(h,h_,steps=T).tolist()]
    w_s=[int(i) for i in torch.linspace(w,w_,steps=T).tolist()]
    out=torch.zeros((T,C,target_height,target_width))
    for ind in range(T):
        out[ind:ind+1]=torch.nn.functional.interpolate(images[ind:ind+1, :, i_s[ind]:i_s[ind]+h_s[ind], j_s[ind]:j_s[ind]+w_s[ind]],
                                                       size=(target_height, target_width), mode='bilinear', align_corners=False)
    return out

def random_resized_crop(images, target_height, target_width, scale=(0.8, 1.0), ratio=(3./4., 4./3.)):
    """Crop the given images to random size and aspect ratio. A crop of random size (default of 0.08 to 1.) of the original size and a random aspect
    ratio (default of 3/4 to 4/3) of the original aspect ratio is made. This crop is finally resized to given size. This is popularly used to train the 
    inception networks
    Args:
        images (torch.Tensor): Image to be resized and cropped of shape (T,C,H,W)
        target_height (int): Desired height after cropping
        target_width (int): Desired width after cropping
        scale (tuple[float, float]): Scale range of inception-style area based random resizing
        ratio (tuple[float, float]): Aspect ratio (width/height) range of inception style area based random resizing
    Returns:
        (torch.Tensor): Resized and cropped images of shape (T,C,H,W)
    """
    height, width=images.shape[-2:]
    
    i, j, h, w=_get_param_spatial_crop(scale, ratio, height, width)
    return torch.nn.functional.interpolate(images[...,i:i+h, j:j+w], size=(target_height, target_width), mode='bilinear', align_corners=False)

def horizontal_flip(prob, images, boxes=None):
    """Perform horizontal flip on the given images and corresponding boxes
    Args:
        prob (float): Probability to flip the images
        images (torch.Tensor): Images to be flipped whose dimension is (T,C,H,W)
        boxes (np.ndarray, optional): Corresponding bounding boxes with dimension (`num_boxes`, 4)
    Returns:
        (torch.Tensor): Images with dimensio of (T,C,H,W)
        (np.ndarray|None): Bounding boxes of shape (`num_boxes`, 4)
    """
    flipped_boxes=None if boxes is None else boxes.copy()
    if np.random.uniform()<prob:
        images=images.flip(dims=(-1,))
        width=images.shape[-1]
        if boxes is not None: flipped_boxes[...,[0,2]]=width-boxes[...,[2,0]]-1
    return images, flipped_boxes

def uniform_crop(images, size, spatial_idx, boxes=None, scale_size=None):
    """Perform uniform spatial sampling on the images and corresponding boxes
    Args:
        images (torch.Tensor): Images to be uniformly cropped whose dimension is (T,C,H,W)
        size (int): Size of height and width to crop the images to
        spatial_idx (int): 0,1,or 2 for left, center, and right crop if width is larger than height. Or 0,1,2 for top, center, and bottom crop if
            height is larger than width
        boxes (np.ndarray, optional): Corresponding boxes to the images whose dimension is (`num_boxes`, 4)
        scale_size (int,optional): If not None, resize the images by making the length of the shortest side to be scale_size while maintaining aspect 
            ratio (width/height)
    Returns:
        (torch.Tensor): Cropped images with dimension of (T,C,H,W)
        (np.ndarray|None): Cropped boxes with dimension of (`num_boxes`, 4)
    """
    assert spatial_idx in [0,1,2]
    ndim=images.ndim
    if ndim==3: images=images.unsqueeze(0)
    height, width=images.shape[-2:]
    
    if scale_size is not None:
        if width<=height: width, height=scale_size, int((height/width)*scale_size)
        else: width,height=int((width/height)*scale_size), scale_size
        images=torch.nn.functional.interpolate(images, size=(height, width), mode='bilinear', align_corners=False)
    
    y_offset=int(math.ceil((height-size)/2))
    x_offset=int(math.ceil((width-size)/2))
    if height>width:
        if spatial_idx==0: y_offset=0
        elif spatial_idx==2: y_offset=height-size
    else:
        if spatial_idx==0: x_offset=0
        elif spatial_idx==2: x_offset=width-size
    cropped=images[...,y_offset:y_offset+size, x_offset:x_offset+size]
    cropped_boxes=crop_boxes(boxes, x_offset, y_offset) if boxes is not None else None
    if ndim==3: cropped=cropped.squeeze(dim=0)
    return cropped, cropped_boxes

class Compose(object):
    """Compose several transforms
    Args:
        transforms (list[callable]): List of transforms to be composed
    """
    def __init__(self, transforms): self.transforms=transforms
    def __call__(self, clip):
        for t in self.transforms: clip=t(clip)
        return clip

class Resize(object):
    """Resize a list of (H,W,C) np.ndarray to the final size

    The larger the original image us, the more times it takes to inpterpolate
    Args:
        interpolation (str): Interpolation method with choices of 'nearest', 'bilinear'. Default to 'nearest'
        size (tuple[int, int]): Desired size (width, height)
    """
    def __init__(self, size, interpolation='nearest'):
        self.size=size 
        self.interpolation=interpolation
        
    def __call__(self, clip):
        """
        The function resizes the clip so that the shortest size matches `size` while maintaining the aspect ratio. It does not change the data type of the clip
        Args:
            clip (np.ndarray| list[np.ndarray] | list[PIL.Image.Image]): Video clips of shape (T,H,W,C) where T is the number of frames and C is the number 
                of channels, or list of (H,W,C) ndarray video frames or list of PIL.Image.Image video frames
        Returns:
            (list[np.ndarray] | list[PIL.Image.Image]): List of resized (H,W,C) ndarray video frames or PIL.Image.Image video frames
        """
        return resize_clip(clip, self.size, interpolation=self.interpolation)

class CenterCrop(object):
    """Extract center crop at the same location for a sequence of images
    Args:
        size (sequence[int,int] | int): Desired output size (width, height)
    """
    def __init__(self, size):
        self.size=(size,size) if isinstance(size, numbers.Number) else size # (width, height)
        
    def __call__(self, clip):
        """
        Args:
            clip (np.ndarray | list[np.ndarray] | list[PIL.Image.Image]): (T,H,W,C) ndarray images or list of (H,W,C) ndarray images or
                list of PIL.Image.Image
        Returns: 
            (list[np.ndarray] | list[PIL.Image.Image]): List of (H,W,C) cropped ndarray images or list of cropped PIL.Image.Image
        """
        w, h=self.size
        if isinstance(clip[0], np.ndarray): im_h, im_w, _=clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image): im_w, im_h=clip[0].size
        else: raise TypeError(f"Expected list of np.ndarray or PIL.Image.Image,but got list of {type(clip[0])}")
            
        if w>im_w or h>im_h:
            raise ValueError(f"Initial size must be larger than cropped size, but got cropped size of ({w},{h}) and original size of ({im_w},{im_h})")
        
        x0=int(round((im_w-w)/2.))
        y0=int(round((im_h-h)/2.))
        
        return crop_clip(clip, y0,x0,h,w)

class ClipToTensor(object):
    """Convert a list of m (H,W,C) np.ndarray in the range of [0,255] to a torch.FloatTensor of shape (C,m,H,W) in the range [0.,1.]
    Args:
        channel_nb (int): Number of channels
        div_255 (bool): Whether to divide inputs by 255
        numpy (bool): Whether to return output as np.ndarray or as torch.Tensor
    """

    def __init__(self, channel_nb=3, div_255=True, numpy=False):
        self.channel_nb=channel_nb
        self.div_255=div_255
        self.numpy=numpy
        
    def __call__(self, clip):
        """Transform a (T,H,W,C) ndarray or a sequence of (H,W,C) np.ndarray or PIL.Image.Image to its corresponding (C,T,H,W) ndarray or tensor. The output
        can be scaled to range [0,1] of type float or maintain the range of [0,255] of type unsigned int8
        
        Args:
            clip (list[np.ndarray] | np.ndarray): (T,H,W,C) image or a list of (H,W,C) image ndarray to be converted to tensor
        Returns:
            (np.ndarray | torch.Tensor): (C,T,H,W) ndarray or tensor 
        """
        assert isinstance(clip[0], np.ndarray) or isinstance(clip[0], Image.Image), ("Clip must be a sequence of np.ndarray or PIL.Image.Image" 
                                                                                      f" but got {type(clip[0])}")
        
        dtype=clip[0].dtype if isinstance(clip[0], np.ndarray) else np.array(clip[0]).dtype
    
        if isinstance(clip[0],np.ndarray):
            h,w,c=clip[0].shape
            assert c==self.channel_nb
        elif isinstance(clip[0],Image.Image): w,h=clip[0].size
        
        np_clip=np.zeros([self.channel_nb, len(clip), int(h), int(w)], dtype=dtype) # (C,T,H,W)
        for img_idx, img in enumerate(clip):
            if isinstance(img, Image.Image): img=np.array(img, copy=False)
            if img.ndim==3: img=img.transpose(2,0,1) # (H,W,C) to (C,H,W)
            elif img.ndim==2: img=np.expand_dims(img, 0) # (H,W) to (1,H,W)
            np_clip[:,img_idx]=img
        
        if self.div_255: 
            if dtype!=np.float32: np_clip=np_clip.astype(np.float32)
            np_clip/=255.0
            
        if self.numpy: return np_clip
        return torch.from_numpy(np_clip) # (C,T,H,W)

class Normalize(object):
    """Normalize a clip with mean and standard deviation
    Args:
        mean (tuple[float]): Mean intensity value per channel
        std (tuple[float]): Standard deviation value per channel
    """
    def __init__(self, mean, std):
        self.mean=mean
        self.std=std

    def __call__(self, clip):
        """
        Args:
            clip (torch.Tensor): Tensor of shape (C, T, H, W) to be normalized
        Returns:
            (torch.Tensor): Normalized tensor of shape (C, T, H, W)
        """
        return normalize(clip, self.mean, self.std)

    def __repr__(self):
        return self.__class__.__name__ + f"(mean={self.mean}, std={self.std})"
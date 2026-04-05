# Reference: https://github.com/huggingface/pytorch-image-models/blob/main/timm/data/mixup.py
"""Mixup and Cutmix

Papers:
mixup: Beyond Empirical Risk Minimization (https://arxiv.org/abs/1710.09412)
CutMix: Regularization Strategy to Train Strong Classifiers with Localization Features (https://arxiv.org/abs/1905.04899)

Code Reference: 
CutMix: https://github.com/clovaai/CutMix-PyTorch

Hacked together by / Copyright 2019, Ross Wightman
"""

import torch
import numpy as np

def one_hot(x, num_classes, on_value=1., off_value=0.):
    """
    Args:
        x (torch.Tensor): Labels of type long with shape (batch_size, )
        num_classes (int): Number of classes
        on_value (float): Value indicating `on` or `true` with the range of (0,1], typically a value close to 1.
        off_value (float): Value indicating `off` or `false` with range of [0,1), typically a small value close to 0.
    Returns:
        (torch.Tensor): One-hot encoding of shape (batch_size, num_classes) where sum along dim=1 will be one
    """
    x=x.long().view(-1,1)
    # create a tensor of size (batch_size, num_classes) filled with off_value. Then use x (labels) as indices
    # to assign on_value on dimension 1
    return torch.full((x.size()[0], num_classes), off_value, device=x.device).scatter_(1, x, on_value)

def mixup_target(target, num_classes, lam=1., smoothing=0.):
    """Create a mixed target
    Args:
        target (torch.Tensor): Labels of type long and shape (batch_size, )
        num_classes (int): Number of classes
        lam (float): Ratio of the intensities/areas of original images maintained in MixUp/CutMix
        smoothing (float): Label smoothing value must be in the range (0.,1.), typically a small value like 0.1
    Returns:
        (torch.Tensor): Mixing one-hot encoding of shape (batch_size, num_classes) where sum along dim=1 will be one
    """
    off_value=smoothing/num_classes # so for each object, sum of one-hot encoding return 1
    on_value=1.-smoothing+off_value # equivalent to 1-smoothing*(num_classes-1)/num_classes, so for each object, sum of one-hot encoding return 1
    y1=one_hot(target, num_classes, on_value=on_value, off_value=off_value)
    y2=one_hot(target.flip(0), num_classes, on_value=on_value, off_value=off_value) # reverse order of batch
    return y1*lam + y2*(1.-lam)
    
def rand_bbox_minmax(img_shape, minmax, count=None):
    """Min-Max CutMix bounding box as top-left and bottom-right corner coordinates (x1y1x2y2)
    Inspired by Darknet cutmix implementation, generates a random rectangular bbox based on min/max percent values applied to each dimension of the input
    image
    
    Typical defaults for minmax are usually in the .2-.3 for min and .8-.9 range for max
    Args:
        img_shape (tuple[int]): Image shape as tuple, including (...,H,W) 
        minmax (tuple[float,float]|list[float,float]): Min and max bbox ratio (as percentage of image size)
        count (int): Number of bbox to generate
    Returns:
        (int): Top y coordinate (i.e., y1)
        (int): Bottom y coordinate (i.e., y2)
        (int): Left x coordinate (i.e., x1)
        (int): Right x coordinate (i.e., x2)
    """
    assert len(minmax)==2
    img_h, img_w=img_shape[-2:]
    cut_h=np.random.randint(int(img_h*minmax[0]), int(img_h*minmax[1]), size=count)
    cut_w=np.random.randint(int(img_w*minmax[0]), int(img_w*minmax[1]), size=count)
    yl=np.random.randint(0, img_h-cut_h, size=count) # top
    xl=np.random.randint(0, img_w-cut_w, size=count) # left
    yu=yl+cut_h
    xu=xl+cut_w
    return yl, yu, xl, xu

def rand_bbox(img_shape, lam, margin=0., count=None):
    """Standard CutMix bounding box
    Generate a random square bbox based on lambda value. This implementation includes support for enforcing a border margin 
    as percent of bbox dimensions
    
    Args:
        img_shape (tuple): Image shape as tuple as of (...,H,W) 
        lam (float): Cutmix lambda value
        margin (float): Percentage of bbox dimension to enforce as margin (reduce amount of box outside image)
        count (int): Number of bbox to generate
    Returns:
        (int): Top y coordinate (i.e., y1)
        (int): Bottom y coordinate (i.e., y2)
        (int): Left x coordinate (i.e., x1)
        (int): Right x coordinate (i.e., x2)
    """
    ratio=np.sqrt(1-lam) # lam is the ratio of maintaining the original so 1-lam is the ratio of cut-out
    img_h, img_w=img_shape[-2:]
    cut_h, cut_w=int(img_h*ratio), int(img_w*ratio)
    margin_y, margin_x=int(margin*cut_h), int(margin*cut_w)
    cy=np.random.randint(0+margin_y, img_h-margin_y, size=count)
    cx=np.random.randint(0+margin_x, img_w-margin_x, size=count)
    yl=np.clip(cy-cut_h//2, 0, img_h) # lower y
    yh=np.clip(cy+cut_h//2, 0, img_h) # higher y
    xl=np.clip(cx-cut_w//2, 0, img_w)
    xh=np.clip(cx+cut_w//2, 0, img_w)
    return yl, yh, xl, xh

def cutmix_bbox_and_lam(img_shape, lam, ratio_minmax=None, correct_lam=True, count=None):
    """Generate bbox and apply lambda correction
    Args:
        img_shape (tuple[int,int,int]): Image shape including (C,H,W) or (C*T,H,W)
        lam (float): Mixing/Area ratio
        count (int): Number of bbox to generate
    Returns:
        (tuple[int,int,int,int]): Cut box coordinates in the format of y1,y2,x1,x2 where (x1,y1) is the top left corner and (x2,y2) 
            is the bottom right
        (float): Ratio of the original image area that was maintained (not getting cut)
    """
    if ratio_minmax is not None: yl,yu,xl,xu=rand_bbox_minmax(img_shape=img_shape, minmax=ratio_minmax, count=count) # top/bottom,left/right
    else: yl,yu,xl,xu=rand_bbox(img_shape, lam, count=count) # top/bottom,left/right
    
    if correct_lam or ratio_minmax is not None:
        bbox_area=(yu-yl)*(xu-xl)
        lam=1.-bbox_area/float(np.prod(img_shape[-2:])) # ratio of original area maintained
    
    return (yl,yu,xl,xu), lam
    
class Mixup:
    """Mixup/Cutmix that applies different params to each element/item in a batch or to the whole batch. The images in the batch must be float, e.g., after 
    normalization
    Args:
        mixup_alpha (float): Mixup alpha value, mixup is active if > 0.
        cutmix_alpha (float): Cutmix alpha value, cutmix is active if > 0.
        cutmix_minmax (list[float]): Cutmix min/max image ratio, cutmix is active and uses this value vs alpha if not None
        prob (float): Probability of applying mixup or cutmix per batch or element
        switch_prob (float): Probability of switching to cutmix instead of mixup when both are active
        mode (str): How to apply mixup/cutmix parameters [per 'batch', 'pair' (pair of elements), 'elem' (element)]
        correct_lam (bool): Apply lambda correction when cutmix bbox clipped by image borders
        label_smoothing (float): Apply label smoothing to the mixed target tensor
        num_classes (int): Number of classes for target
    """
    def __init__(self, mixup_alpha=1., cutmix_alpha=0., cutmix_minmax=None, prob=1., switch_prob=0.5, mode='batch', correct_lam=True, 
                 label_smoothing=0.1, num_classes=101):
        self.mixup_alpha=mixup_alpha
        self.cutmix_alpha=cutmix_alpha
        self.cutmix_minmax=cutmix_minmax
        if self.cutmix_minmax is not None:
            assert len(self.cutmix_minmax)==2
            # force cutmix alpha ==1. when minmax active to keep logic simple & safe
            self.cutmix_alpha=1.
        self.mix_prob=prob
        self.switch_prob=switch_prob
        self.label_smoothing=label_smoothing
        self.num_classes=num_classes
        self.mode=mode
        self.correct_lam=correct_lam # correct lambda based on clipped area for cutmix
        self.mixup_enabled=True # set to false to disable mixing (intended to be set by train loop)

    def _params_per_elem(self, batch_size):
        """For each image in the batch, determine lambda `lam` which is mixing coefficient, ranging 0-1, and `lam ~ Beta(alpha, alpha)`is sampled from a Beta 
        distribution to ensure that the mixing is not always 50/50 but favors one image over the other. 
        
        For Mixup, new_img=lam*img1 + (1-lam)*img2
        For Cutmix, lam represents an area ratio. If lam=0.7, 70% of pixels come from the original image and 30% are replaced by a cut-out from another image
        
        Args:
            batch_size (int): Number of elements in each batch
        Returns:
            (np.ndarray): Mixing ratio or Area ratio of size (batch_size, ) and type float32
            (np.ndarray): Whether element involves cutmix of size (batch_size,) and type bool
        """
        lam=np.ones(batch_size, dtype=np.float32) # assuming not mixing
        use_cutmix=np.zeros(batch_size, dtype=bool) # assuming not using cutmix
        if self.mixup_enabled: 
            if self.mixup_alpha>0. and self.cutmix_alpha>0.:
                use_cutmix=np.random.rand(batch_size)<self.switch_prob
                # Beta with alpha < 1 (most common) gives the highest probability density near 0 and 1 (U-shape) [typically set 0.2-0.5]
                # This makes lam most of the time close to 0 or 1, meaning the augmented image still looks mostly like one of the original images
                # Only occasionally, lam is near 0.5 resulting in highly-uncertain samples which force model to smooth out its decision boundaries
                # improving generalization and making the model more robust to adversarial attack
                # We note that alpha=0, point masses are at 0 or 1, effectively turning off mixing
                #              0<alpha<1, U-shape, favors original images with slight mixing
                #              alpha=1, uniform, equal chance for any mixing ratio
                #              alpha>1, Bell-shaped, forcing heavy mixing which can be too difficult for some models
                lam_mix=np.where(use_cutmix, np.random.beta(self.cutmix_alpha, self.cutmix_alpha, size=batch_size),
                                 np.random.beta(self.mixup_alpha, self.mixup_alpha, size=batch_size)) # (batch_size,)
            elif self.mixup_alpha>0.: lam_mix=np.random.beta(self.mixup_alpha, self.mixup_alpha, size=batch_size)
            elif self.cutmix_alpha>0.: 
                use_cutmix=np.ones(batch_size, dtype=bool)
                lam_mix=np.random.beta(self.cutmix_alpha, self.cutmix_alpha, size=batch_size)
            else: raise ValueError("One of the following must be true: mixup_alpha>0., or cutmix_alpha>0., or cutmix_minmax is not None")
            lam=np.where(np.random.rand(batch_size) < self.mix_prob, lam_mix.astype(np.float32), lam)
        return lam, use_cutmix

    def _params_per_batch(self):
        """Determine a single value of MixUp/CutMix ratio (since it will be applied to the whole batch) and whether to use CutMix or not
        Returns:
            (float): Ratio of maintaining the original image in MixUp/CutMix
            (bool): Whether to use CutMix or not
        """
        lam=1.
        use_cutmix=False
        if self.mixup_enabled and np.random.rand()<self.mix_prob:
            use_cutmix=np.random.rand()<self.switch_prob
            lam_mix=np.random.beta(self.cutmix_alpha, self.cutmix_alpha) if use_cutmix else \
                    np.random.beta(self.mixup_alpha, self.mixup_alpha)
        elif self.mixup_alpha>0.: lam_mix=np.random.beta(self.mixup_alpha, self.mixup_alpha)
        elif self.cutmix_alpha>0.:
            use_cutmix=True
            lam_mix=np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        else: raise ValueError("One of mixup_alpha>0., cutmix_alpha>0., cutmix_minmax not None should be true")
        lam=float(lam_mix)
        
        return lam, use_cutmix

    def _mix_elem(self, x):
        """
        Mixing each element/item in the batch separately. This is an in-place operation of x 
        Args:
            x (torch.Tensor): Image frame tensors of shape (B,C*T,H,W)
        Returns:
            (torch.Tensor): Lambda value for MixUp and CutMix of size (B,1) where 1.0 means maintaining original image
                - For MixUp, ratio of original image as img*lam + other*(1-lam)
                - For CutMix, ratio of original image area maintained and the remaining is replaced by the other
        """
        batch_size=len(x)
        lam_batch, use_cutmix=self._params_per_elem(batch_size) # the former is (batch_size, ) float32 mixing coefficient, the latter is (batch_size, ) bool
        x_orig=x.clone() # need to keep an unmodified original as a mixing source used below
        
        for i in range(batch_size): # iterate from start-index (0) to end-index (batch_size-1)
            j=batch_size-i-1 # iterate from end-index to start-index 
            lam=lam_batch[i] # mixing coefficient of size (batch_size,)
            if lam==1.: continue
            if use_cutmix[i]:
                (yl,yh,xl,xh),lam=cutmix_bbox_and_lam(x[i].shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)
                x[i][:,yl:yh,xl:xh]=x_orig[j][:,yl:yh,xl:xh] # replace cut-out area in x[i] by x_orig[j]
                lam_batch[i]=lam
            else: x[i]=x[i]*lam + x_orig[j]*(1-lam)
        del x_orig
        return torch.tensor(lam_batch, device=x.device, dtype=x.dtype).unsqueeze(1) # of size (batch_size, 1)

    def _mix_pair(self, x):
        """ Modify elements/items in the batch in pair. For example, if there are 6 items in the batch (batch_size=6), item 0th and 5th are 
        modify concurrently, 
        
            x[0][:,yl:yh,xl:xh]=other[5][:,yl:yh,xl:xh]
            x[5][:,yl:yh,xl:xh]=other[0][:,yl:yh,xl:xh]
            
        Note: this is an in-place operation of x
        
        Args:
            x (torch.Tensor): Image frame tensors of shape (B,C*T,H,W)
        Returns:
            (torch.Tensor): Lambda value for MixUp and CutMix of size (B,1) where 1.0 means maintaining original image
                - For MixUp, ratio of original image as img*lam + other*(1-lam)
                - For CutMix, ratio of original image area maintained and the remaining is replaced by the other
        """
        batch_size=len(x)
        lam_batch, use_cutmix=self._params_per_elem(batch_size//2) # each is np.ndarray of size (batch_size//2, )
        x_orig=x.clone() # keep an unmodified original to use as a mixing source below
        for i in range(batch_size//2):# from i to batch_size//2
            j=batch_size-i-1 # from batch_size to batch_size//2
            lam=lam_batch[i]
            if lam==1.: continue
            if use_cutmix[i]:
                (yl,yh,xl,xh),lam=cutmix_bbox_and_lam(x[i].shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)
                x[i][:,yl:yh,xl:xh]=x_orig[j][:,yl:yh,xl:xh]
                x[j][:,yl:yh,xl:xh]=x_orig[i][:,yl:yh,xl:xh]
                lam_batch[i]=lam
            else:
                x[i]=x[i]*lam + x_orig[j]*(1-lam)
                x[j]=x[j]*lam + x_orig[i]*(1-lam) 
        del x_orig
        lam_batch=np.concatenate((lam_batch, lam_batch[::-1]))
        return torch.tensor(lam_batch, dtype=x.dtype, device=x.device).unsqueeze(1)
    
    def _mix_batch(self, x):
        """ Modifying the whole batch using reversed order of elements/items in the batch using the same operation (MixUp/CutMix) and mixing ratio. 
        This is an in-place operation of x 
        Args:
            x (torch.Tensor): Image frames of shape (B, C*T, H, W)
        Returns:
            (float): Ratio of the original image (intensity/areas) maintained after MixUp/CutMix
        """
        lam, use_cutmix=self._params_per_batch()
        if lam==1.: return 1.
        if use_cutmix:
            (yl,yh,xl,xh),lam=cutmix_bbox_and_lam(x.shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)
            x[:,:,yl:yh,xl:xh]=x.flip(0)[:,:,yl:yh,xl:xh] # reversing the order of batch item and assign latter items to the former items
        else:
            x_flipped=(1.-lam)*x.flip(0)
            x.mul_(lam).add_(x_flipped)
        return lam

    def __call__(self, x, target):
        """Perform MixUp/CutMix 
        Args:
            x (torch.Tensor): Images of shape (B,C,H,W) where B must be even
        Returns:
            (torch.Tensor): Images after transformation with shape (B,C,H,W) where B must be even
            (torch.Tensor): One-hot encoding after mixing with shape (B, num_classes)
        """
        assert x.shape[0]%2==0, f'Batch size must be even to use MixUp, but got the batch size of {x.shape[0]}'
        if self.mode=='elem': lam=self._mix_elem(x) # lam is the ratio of MixUp/CutMix
        elif self.mode=='pair': lam=self._mix_pair(x) 
        else: lam=self._mix_batch(x)
        target=mixup_target(target, self.num_classes, lam, self.label_smoothing)
        return x, target
from typing import Iterable
import time
import math

import torch
import torch.nn as nn

from computer_vision.video_mae.utils import SmoothedValue, MetricLogger

def train_one_epoch(model:nn.Module, data_loader:Iterable, optimizer:torch.optim.Optimizer, device:torch.device, epoch:int, max_norm:float=0,
                    tubelet_size:int=2, patch_size:tuple[int]=(16,16), normalize_target:bool=True, lr_scheduler=None, start_steps=None, 
                    lr_schedule_values=None, wd_schedule_values=None, print_freq=20, metric_logger=None, n_steps=None):
    """
    Args:
        model (nn.Module): Neural network to train
        data_loader (Iterable): Training data loader
        optimizer (torch.optim.Optimizer): Optimizer
        device (torch.device): Torch computing device
        epoch (int): Current epoch
        max_norm (float): Maximum gradient norm. If `max_norm`>0, gradient norm will be clipped to this maximum value
        tubelet_size (int): Size of tubelet along temporal dimension
        patch_size (tuple[int]): Patch size along height and width
        normalize_target (bool): Whether to normalize labels based on patch mean and standard deviation
        lr_scheduler (torch.optim.lr_scheduler): Torch learning rate scheduler
        start_steps (int): Start training steps
        lr_schedule_values (np.ndarray): Schedule of learning rates
        wd_schedule_values (np.ndarray): Schedule of weight decay
        print_freq (int): Print frequency in step unit
        metric_logger (MetricLogger): Object to record training progress
        n_steps (int): Number of steps to run. For debugging and developing code only
    Returns:
        (dict[str, float]): Average accuracy
                    
    Reference: https://github.com/OpenGVLab/VideoMAEv2/blob/master/engine_for_pretraining.py#L19
    """
    model.train()
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("min_lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header=f"Epoch: [{epoch}]"
    
    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if n_steps is not None and step>n_steps-1:
            print(f"Hit the desired number of steps {step}/{n_steps}--break")
            break
        # assign learning rate and weight decay for each step
        it=start_steps+step # global training iteration
        if any(x is not None for x in [lr_schedule_values, wd_schedule_values]):
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"]=lr_schedule_values[it]*param_group['lr_scale']
                if wd_schedule_values is not None and param_group["weight_decay"]>0.:
                    param_group["weight_decay"]=wd_schedule_values[it]
    
        # Note: when the decoder mask ratio is 0 (i.e., when decoder masking is not used), decoder_mask=~encoder_mask
        images, encoder_mask, decoder_mask=batch
        images=images.to(device, non_blocking=device.type=='cuda')
        # from (B,Tp,Hp*Wp) to (B, Tp*Hp*Wp) where Tp is the number of tube in temporal dimension, Hp and Wp is the number of patches in H and W dimensions
        encoder_mask=encoder_mask.to(device=device, dtype=torch.bool, non_blocking=device.type=='cuda').flatten(1) 
        decoder_mask=decoder_mask.to(device=device, dtype=torch.bool, non_blocking=device.type=='cuda').flatten(1)
        assert images.isfinite().all() and images.abs().sum()>0, f'images is Inf or NaN or blank'
        assert encoder_mask.isfinite().all() and encoder_mask.any(), f'encoder_mask is Inf or NaN or blank'
        assert decoder_mask.isfinite().all() and decoder_mask.any(), f'decoder_mask is Inf or NaN or blank'
    
        # calculate the label
        labels=calculate_labels(images=images, decoder_mask=decoder_mask, p0=tubelet_size, p1=patch_size[0], p2=patch_size[1],
                                normalize_target=normalize_target)
    
        outputs=model(images, encoder_mask, decoder_mask)

        loss=(outputs-labels)**2. # (B, N_vis, K) where K is the number of raw pixel values, patch_size[0]*patch_size[1]*tubelet_size*3 for RGB
        loss=loss.mean(dim=-1) # (B, N_vis) where N_vis is the number of visible tubelets
        cal_loss_mask=encoder_mask[~decoder_mask].reshape(images.shape[0],-1)  # (B, N_vis)
        loss=(loss*cal_loss_mask).sum()/cal_loss_mask.sum()
        
        loss_value=loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss {loss_value} is infinite, stop training")
            sys.exit(2)
        
        optimizer.zero_grad()
        loss.backward()
        if max_norm is not None: grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else: grad_norm=get_grad_norm(model.parameters())
        optimizer.step()
    
        metric_logger.update(loss=loss_value)
        min_lr, max_lr=10.,0.
        for group in optimizer.param_groups:
            min_lr=min(min_lr, group['lr'])
            max_lr=max(max_lr, group['lr'])
        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value=None
        for group in optimizer.param_groups:
            if group['weight_decay']>0: weight_decay_value=group['weight_decay']
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)
    
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    

@torch.no_grad()
def calculate_labels(images, decoder_mask, p0, p1, p2, normalize_target):
    """
    Determine patches that are invisible to decoder
    Args:
        images (torch.Tensor): Video frames of shape (B,C,T,H,W) of type float32 where C is the number of channels and T is the number of frames
        decoder_mask (torch.Tensor): Boolean tensor with True values for patches invisible to decoder of shape (B, t*h*w) where t is the number
            of tubes, h is the number of patches along H dimension, and w is the number of patches along W dimensiom. Thus, t*h*w is the total
            number of patches/tokens
        p0 (int): Tube size along the temporal dimension
        p1 (int): Patch size along height dimension
        p2 (int): Patch size along width dimension
    normalize_target (bool): Whether to normalize `images` based on patch mean and patch standard deviation
    
    """
    from computer_vision.video_mae.dataset.pretrained_datasets import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    
    mean=torch.as_tensor(IMAGENET_DEFAULT_MEAN)[None,:,None,None,None] # (1,3,1,1,1)
    std=torch.as_tensor(IMAGENET_DEFAULT_STD)[None,:,None,None,None] # (1,3,1,1,1)

    unnorm_images=images*std.to(device=images.device, dtype=images.dtype) + mean.to(device=images.device, dtype=images.dtype) # in [0,1]
    
    # ---- reshape images from (B,C,T,H,W) to (B, t*h*w, p0*p1*p2, C)
    B,C,T,H,W=unnorm_images.shape
    t,h,w=T//p0,H//p1,W//p2
    # split patch dimension
    unnorm_images=unnorm_images.view(B,C,t,p0,h,p1,w,p2)
    # reorder to (B,t,h,w,p0,p1,p2,C)
    unnorm_images=unnorm_images.permute(0,2,4,6,3,5,7,1).contiguous()
    # reshape to (B, t*h*w, p0*p1*p2, C) where t*h*w is the total number of patches/tokens and p0*p1*p2 is the total number of pixels per patch
    unnorm_images=unnorm_images.view(B, t*h*w, p0*p1*p2, C)
    if normalize_target:
        # -------normalize by patch mean and patch std (i.e., mean and std computed along the p0*p1*p2 dimension)
        images_norm=(unnorm_images-unnorm_images.mean(dim=-2, keepdim=True))/( unnorm_images.var(dim=-2, correction=1, keepdim=True).sqrt() +1e-6 )
        images_patch=images_norm.view(*images_norm.shape[:2], -1) # (B, t*h*w, p0*p1*p2*C)
    else:
        images_patch=unnorm_images.view(*unnorm_images.shape[:2], -1) # (B, t*h*w, p0*p1*p2*C

    B, N, C = images_patch.shape
    # decode_masked_pos of shape (B, t*h*w) and images_patch of shape (B, t*h*w, p0*p1*p2*C) so masking yield (something, p0*p1*p2*C)
    labels=images_patch[~decoder_mask].reshape(B,-1,C) # (B,gt,C) where gt is the number of patches that get masked
    return labels
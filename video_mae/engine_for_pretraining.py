from typing import Iterable
from pathlib import Path
from functools import partial

import math
import time
import math
import random
import datetime

import torch
import torch.nn as nn

import numpy as np
from PIL import Image

from computer_vision.torch_video.utils.plotting import plot_all
from computer_vision.torch_video.utils.progress import save_metrics
from computer_vision.video_mae.dataset.pretrained_datasets import HybridVideoMAE, DataAugmentationForVideoMAEv2
from computer_vision.video_mae.models.modeling_pretrain import PretrainVisionTransformer, pretrain_videomae_tiny_patch16_224
from computer_vision.video_mae.optim_factory import create_optimizer
from computer_vision.video_mae.utils import SmoothedValue, MetricLogger, save_checkpoint, form_stats, seed_worker, multiple_pretrain_samples_collate, \
cosine_scheduler, get_grad_norm

def train(args):
    
    args.output_dir=Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir=args.output_dir/"checkpoints"
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.last=args.checkpoint_dir/args.last
    args.best=args.checkpoint_dir/args.best
    print(f"{args.last=}")
    print(f"{args.best=}")

    device=torch.device(args.device) if (torch.cuda.is_available() and args.device=='cuda') else torch.device('cpu')
    print(f"{device=}")
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    model=pretrain_videomae_tiny_patch16_224(args)
    patch_size=model.encoder.patch_embed.patch_size
    print(f"Patch size {patch_size}")
    args.window_size=(args.num_frames//args.tubelet_size, args.input_size//patch_size[0], args.input_size//patch_size[1]) # T,H,W
    args.patch_size=patch_size
    
    # build dataset
    transform = DataAugmentationForVideoMAEv2(args)
    train_dataset=HybridVideoMAE(root=args.data_root, setting=args.data_path, train=True, test_mode=False, name_pattern=args.fname_tmpl, 
                           video_ext='avi', is_color=True, modality='rgb', num_segments=1, num_crop=1, new_length=args.num_frames, 
                           new_step=args.sampling_rate, transform=transform, temporal_jitter=False, lazy_init=False, num_sample=args.num_sample)
    
    num_training_steps_per_epoch=len(train_dataset)//args.batch_size
    print(f"{num_training_steps_per_epoch=}")
    
    
    collate_func=None
    if args.num_sample>1: collate_func=partial(multiple_pretrain_samples_collate, fold=False)
    num_devices=torch.cuda.device_count() # number of CUDA devices
    # MUST KEEP args.num_workers to ZERO
    train_dataloader=torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, 
                                                 pin_memory=num_devices>0 and args.pin_mem, drop_last=len(train_dataset)%args.batch_size!=0, 
                                                 worker_init_fn=seed_worker, persistent_workers=args.num_workers>0, collate_fn=collate_func)
    

    checkpoint=None
    if args.resume and args.last.is_file():
        checkpoint=torch.load(args.last, map_location='cpu', weights_only=False)
        print(f"Resume from checkpoint: {args.last}")
        model.load_state_dict(checkpoint['model'])
    
    model.to(device)
    n_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {n_parameters/1e6} million")
    print(f"Number of training steps per epoch {num_training_steps_per_epoch}")
    print(f"Number of training examples per epoch {args.batch_size*num_training_steps_per_epoch}")
    
    optimizer=create_optimizer(args, model)
    print("Use step level LR & WD scheduler!")
    lr_schedule_values=cosine_scheduler(args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch, warmup_epochs=args.warmup_epochs,
                                       warmup_steps=args.warmup_steps)
    
    if args.weight_decay_end is None: args.weight_decay_end=args.weight_decay
    wd_schedule_values=cosine_scheduler(args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print(f"Max WD={wd_schedule_values.max():.7f}, Min WD={wd_schedule_values.min():.7f}")
    
    best_loss=float('inf')
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch=checkpoint['epoch']+1
        best_loss=checkpoint['best_loss']
    
    torch.cuda.empty_cache()
    print(f"Start training for {args.epochs} epochs at {args.start_epoch}")

    start_time=time.time()
    stop=False
    for epoch in range(args.start_epoch, args.epochs):
        if args.n_epochs is not None and epoch>args.n_epochs-1:
            print(f"Hit desired number of epochs {epoch}/{args.n_epochs}--break")
            break
        start_epoch_time=time.time()
        train_metric_logger=MetricLogger(delimiter=" ")
        train_stats=train_one_epoch(model, train_dataloader, optimizer, device, epoch, max_norm=args.clip_grad,
                                    tubelet_size=args.tubelet_size, patch_size=args.patch_size, normalize_target=args.normalize_target, 
                                    lr_scheduler=None, start_steps=epoch*num_training_steps_per_epoch, lr_schedule_values=lr_schedule_values, 
                                    wd_schedule_values=wd_schedule_values, print_freq=args.print_freq, metric_logger=train_metric_logger,
                                    n_steps=args.n_steps)
    
        if args.output_dir is not None:
            save_checkpoint(args.last, model, optimizer, epoch, best_loss, scaler=None)
            stats=form_stats(train_metric_logger, mode='train')#|form_stats(val_metric_logger, mode='val')
            save_metrics(args.output_dir/"result.csv", stats, epoch=epoch, start_epoch_time=start_epoch_time)
            if args.plot_freq>0 and (epoch+1)%args.plot_freq==0: plot_all(args.output_dir/"result.csv")
    
        if best_loss>train_metric_logger.meters['loss'].global_avg:
            best_loss=train_metric_logger.meters['loss'].global_avg
            save_checkpoint(args.best, model, optimizer, epoch, best_loss, scaler=None)
        
        if args.time is not None:
            stop|=(time.time()-start_time)>(args.time*3600)
        if stop: break
                
    total_time=time.time()-start_time
    total_time_str=str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


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
    normalize_target (bool): Whether to normalize `images` based on patch mean and patch standard deviation, i.e., whether to locally normalize pixel values
        using the mean and standard deviation of each patch. By setting this to True, we ask the model to learn the texture and structure (rather than lighting 
        condition and color shifts)
    
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
        # Making the loss and model invariant to contrast and illumination changes. Forcing the model to focus on high-frequency details and geometric shape
        # By squashing raw pixel values (which can vary significantly) to a consistent distribution (zero-mean and unit variance), gradient during the 
        #     train_one_epoch loop stabilizes
        # -------normalize by patch mean and patch std (i.e., mean and std computed along the p0*p1*p2 dimension)
        images_norm=(unnorm_images-unnorm_images.mean(dim=-2, keepdim=True))/( unnorm_images.var(dim=-2, correction=1, keepdim=True).sqrt() +1e-6 )
        images_patch=images_norm.view(*images_norm.shape[:2], -1) # (B, t*h*w, p0*p1*p2*C)
    else:
        images_patch=unnorm_images.view(*unnorm_images.shape[:2], -1) # (B, t*h*w, p0*p1*p2*C

    B, N, C = images_patch.shape
    # decode_masked_pos of shape (B, t*h*w) and images_patch of shape (B, t*h*w, p0*p1*p2*C) so masking yield (something, p0*p1*p2*C)
    labels=images_patch[~decoder_mask].reshape(B,-1,C) # (B,gt,C) where gt is the number of patches that get masked
    return labels
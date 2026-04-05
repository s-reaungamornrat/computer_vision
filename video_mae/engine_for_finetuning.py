from typing import Iterable, Optional, Callable

import sys
import math

import torch
import numpy as np

from computer_vision.video_mae.utils import MetricLogger, SmoothedValue, get_grad_norm

def train_one_epoch(model:torch.nn.Module, criterion:torch.nn.Module, data_loader:Iterable, optimizer:torch.optim.Optimizer,
                    device:torch.device, epoch:int, max_norm:float=0., mixup_fn:[Callable]=None, lr_schedule_values:np.ndarray=None,
                    wd_schedule_values:np.ndarray=None, num_training_steps_per_epoch:int=None, print_freq:int=20, n_steps:int=None): # update_freq:int=None,
    """
    Args:
        model (torch.nn.Module): Model to be trained
        criterion (torch.nn.Module): Loss
        data_loader (Iterable): Data reader
        optimizer (torch.optim.Optimizer): Optimizer
        device (torch.device): Computing device
        epoch (int): Current epoch to train
        max_norm (float): Maximum allowable gradient norm (used to impose constraint on gradient)
        mixup_fn (Callable): MixUp and CutMix augmentation
        lr_schedule_values (np.ndarray): Learning rate schedule
        wd_schedule_values (np.ndarray): Weight decay schedule
        num_training_steps_per_epoch (int): Number of training iteration per epoch
        print_freq (int): How often to print progress
        n_steps (int): Maximum number of iterations allowed, for debugging and developing code only
    """
    
    start_steps=epoch*num_training_steps_per_epoch
    model.train()
    metric_logger=MetricLogger(delimiter=" ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header=f'Epoch: [{epoch}]'
    
    optimizer.zero_grad()
    for data_iter_step, (samples, targets, _, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        if n_steps is not None and data_iter_step>n_steps-1:
            print(f"Hit the desired number of steps {data_iter_step}/{n_steps}--break")
            break
            
        # samples is (B,C,T,H,W) float32 tensor and targets is (B,) long tensor
        
        step=data_iter_step #//update_freq
        # if step>=num_training_steps_per_epoch: continue
        it=start_steps+step # global training iteration
        # Update LR and WD for the first acc
        if any(x is not None for x in [lr_schedule_values,wd_schedule_values]): # and data_iter_step%update_freq==0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group['lr']=lr_schedule_values[it]*param_group['lr_scale']
                if wd_schedule_values is not None and param_group['weight_decay']>0.:
                    param_group['weight_decay']=wd_schedule_values[it]
        
        samples=samples.to(device=device, non_blocking=device.type=='cuda')
        targets=targets.to(device=device, non_blocking=device.type=='cuda')
        if mixup_fn is not None:
            # mixup handle 4D tensor
            B,C,T,H,W=samples.shape
            samples=samples.view(B,C*T,H,W)
            samples, targets=mixup_fn(samples, targets)
            samples=samples.view(B,C,T,H,W)
        
        outputs=model(samples) # (B, num_classes)
        loss=criterion(outputs, targets)
        
        loss_value=loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)
        
        optimizer.zero_grad()
        loss.backward()
        if max_norm is not None: grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else: grad_norm=get_grad_norm(model.parameters())
        optimizer.step()
    
        class_acc=(outputs.max(-1).indices==targets).float().mean() if mixup_fn is None else None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        
        min_lr,max_lr=10.,0.
        for group in optimizer.param_groups:
            min_lr=min(min_lr, group['lr'])
            max_lr=max(max_lr, group['lr'])
        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        
        weight_decay_value=None
        for group in optimizer.param_groups:
            if group['weight_decay']>0.: weight_decay_value=group['weight_decay']
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)
    
    print(f"Averaged stats: {metric_logger}")
    return {k:meter.global_avg for k, meter in metric_logger.meters.items()}
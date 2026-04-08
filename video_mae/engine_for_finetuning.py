from typing import Iterable, Optional, Callable
from pathlib import Path
from functools import partial

import os
import sys
import math
import time
import random
import datetime

import torch
import numpy as np

from computer_vision.torch_video.utils.plotting import plot_all
from computer_vision.torch_video.utils.progress import save_metrics

from computer_vision.video_mae.dataset.mixup import Mixup
from computer_vision.video_mae.dataset.datasets import VideoClsDataset
from computer_vision.video_mae.models.modeling_finetune import vit_tiny_patch16_224
from computer_vision.video_mae.models.load_pretrain import load_pretrained_model
from computer_vision.video_mae.optim_factory import LayerDecayValueAssigner, create_optimizer
from computer_vision.video_mae.models.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from .utils import (multiple_samples_collate, seed_worker, load_state_dict, get_world_size, cosine_scheduler, 
                    MetricLogger, SmoothedValue, get_grad_norm, save_checkpoint, form_stats)

def accuracy(output, target, topk=(1,)):
    """Compute accuracy of predicting targets
    Args:
        output (torch.Tensor): Logic or probability of each class of shape (B, num_classes) of type float32
        target (torch.Tensor): Label of shape (B,) of type long
        topk (tuple[int]): List of topk of accuracy to compute
    Returns:
        (tuple[float]): List of batch average of each topk accuracy specified in `topk` 
    Reference: https://github.com/huggingface/pytorch-image-models/blob/main/timm/utils/metrics.py
    """
    maxk=min(max(topk), output.shape[-1]) # topk cannot be greater than the number of classes
    batch_size=target.shape[0]
    # topk returns (values, indices),we only care for indices here. pred is (B,maxk) long tensor
    _, pred=output.topk(maxk,dim=1,largest=True,sorted=True) # return maxk largest element of the input tensor 
    pred=pred.t() # (maxk, B)
    # change target from (B,) to (1,B) to (maxk,B)
    correct=pred.eq(target.reshape(1,-1).expand_as(pred)) # (maxk, b) bool tensor
    
    # iterate through each topk and compute average correctness from all items in the batch
    return [correct[:min(k, maxk)].reshape(-1).float().sum()*(100./batch_size) for k in topk]


@torch.no_grad()
def validation_one_epoch(data_loader, model, device, metric_logger, print_freq, n_steps=None):

    criterion=torch.nn.CrossEntropyLoss()
    # metric_logger=MetricLogger(delimiter=" ")
    header='Val:'
    
    # switch to evaludation mode
    model.eval()
    
    for idx, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        if n_steps is not None and idx>n_steps-1:
            print(f"Hit the desired number of steps {idx}/{n_steps}--break")
            break
            
        images=batch[0]
        target=batch[1]
        images=images.to(device=device, non_blocking=device.type=='cuda')
        target=target.to(device=device, non_blocking=device.type=='cuda')
    
        with torch.no_grad():
            output=model(images)
            loss=criterion(output, target)
    
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
    
        batch_size=images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        
    print(f"Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f} loss {metric_logger.loss.global_avg:.3f}")
    
    return {k:meter.global_avg for k, meter in metric_logger.meters.items()}

def train_one_epoch(model:torch.nn.Module, criterion:torch.nn.Module, data_loader:Iterable, optimizer:torch.optim.Optimizer,
                    device:torch.device, epoch:int, max_norm:float=0., mixup_fn:[Callable]=None, lr_schedule_values:np.ndarray=None,
                    wd_schedule_values:np.ndarray=None, num_training_steps_per_epoch:int=None, metric_logger=None, print_freq:int=20,
                    n_steps:int=None): # update_freq:int=None,
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
    # metric_logger=MetricLogger(delimiter=" ")
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

    # data
    args.nb_classes=101
    dataset_train=VideoClsDataset(data_root=args.data_root, anno_path=args.train_data_path, mode='train', clip_len=args.num_frames, 
                            frame_sample_rate=args.sampling_rate, num_segment=1, test_num_segment=args.test_num_segment,
                            test_num_crop=args.test_num_crop, num_crop=1, keep_aspect_ratio=True, crop_size=args.input_size,
                            short_side_size=args.short_side_size, new_height=256, new_width=320, args=args)
    dataset_val=VideoClsDataset(data_root=args.data_root, anno_path=args.val_data_path, mode='validation', clip_len=args.num_frames, 
                            frame_sample_rate=args.sampling_rate, num_segment=1, test_num_segment=args.test_num_segment,
                            test_num_crop=args.test_num_crop, num_crop=1, keep_aspect_ratio=True, crop_size=args.input_size,
                            short_side_size=args.short_side_size, new_height=256, new_width=320, args=args)
    # dataset_test=VideoClsDataset(data_root=args.data_root, anno_path=args.val_data_path, mode='test', clip_len=args.num_frames, 
    #                         frame_sample_rate=args.sampling_rate, num_segment=1, test_num_segment=args.test_num_segment,
    #                         test_num_crop=args.test_num_crop, num_crop=3, keep_aspect_ratio=True, crop_size=args.input_size,
    #                         short_side_size=args.short_side_size, new_height=256, new_width=320, args=args)
    
    if args.num_sample>1: collate_func=partial(multiple_samples_collate, fold=False)
    num_devices=torch.cuda.device_count() # number of CUDA devices
    data_loader_train=torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers,
                                                  pin_memory=num_devices>0 and args.pin_mem, drop_last=len(dataset_train)%args.batch_size!=0,
                                                  collate_fn=collate_func, persistent_workers=args.num_workers>0, worker_init_fn=seed_worker,
                                                  shuffle=True)
    
    data_loader_val=torch.utils.data.DataLoader(dataset_val, batch_size=int(1.5*args.batch_size), num_workers=args.num_workers, shuffle=False,
                                                pin_memory=num_devices>0 and args.pin_mem, drop_last=len(dataset_val)%args.batch_size!=0,
                                                persistent_workers=args.num_workers>0, worker_init_fn=seed_worker)

    # mixup and cutmix
    mixup_fn=None
    mixup_active=args.mixup>0. or args.cutmix>0. or args.cutmix_minmax is not None
    if mixup_active:
        print("MixUp is activated!!!")
        mixup_fn=Mixup(mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax, prob=args.mixup_prob,
                       switch_prob=args.mixup_switch_prob, mode=args.mixup_mode, label_smoothing=args.smoothing, num_classes=args.nb_classes)

    # model
    model=vit_tiny_patch16_224(pretrained=False, img_size=args.input_size, num_classes=args.nb_classes, all_frames=args.num_frames*args.num_segments,
                              tubelet_size=args.tubelet_size, drop_rate=args.drop, drop_path_rate=args.drop_path, attn_drop_rate=args.attn_drop_rate,
                              head_drop_rate=args.head_drop_rate, use_mean_pooling=args.use_mean_pooling, init_scale=args.init_scale,
                              with_cp=args.with_checkpoint, cos_attn=args.cos_attn)
    print(f"{args.tubelet_size=}, {args.drop=}, {args.drop_path=}, {args.attn_drop_rate=}, {args.head_drop_rate=}., {args.use_mean_pooling=}")
    print(f"{args.init_scale=}, {args.with_checkpoint=}")
    args.patch_size=model.patch_embed.patch_size
    args.window_size=(args.num_frames//args.tubelet_size, args.input_size//args.patch_size[0], args.input_size//args.patch_size[1])
    print(f"Patch size: {args.patch_size}, Number of patches per dim: {args.window_size}")

    if not args.last.is_file() and isinstance(args.finetune, str) and os.path.isfile(args.finetune):
        print(f"Pretrained model file, {args.finetune}, existence is {os.path.isfile(args.finetune)}")
        load_pretrained_model(args=args, model=model, weight_fpath=args.finetune)
    
    checkpoint=None
    if args.resume and args.last.is_file():
        checkpoint=torch.load(args.last, map_location='cpu', weights_only=False)
        print(f"Resume from checkpoint: {args.last}")
        model.load_state_dict(checkpoint['model'])
        
    model.to(device)
    n_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model contains {n_parameters} parameters")

    # optimizer
    print(f"{args.lr=}, {args.min_lr=}, {args.warmup_lr=}")
    total_batch_size=args.batch_size*get_world_size() # *args.update_freq
    num_training_steps_per_epoch=len(dataset_train)//total_batch_size # == len(data_loader_train)
    args.lr=args.lr*total_batch_size/256.
    print(f"{total_batch_size=}, {args.update_freq=}, number of training examples {len(dataset_train)}, {num_training_steps_per_epoch=}, {len(data_loader_train)=}")
    #-------- scale lr ----------
    args.min_lr=args.min_lr*total_batch_size/256.
    args.warmup_lr=args.warmup_lr*total_batch_size/256.
    #-------- scale lr ----------
    print(f"{args.lr=}, {args.min_lr=}, {args.warmup_lr=}")
    num_layers=model.get_num_layers()
    print(f"{num_layers=}, {args.layer_decay=}")
    assigner=None
    if args.layer_decay<1.:
        # assigner.values lists small to large decay values
        assigner=LayerDecayValueAssigner(values=[args.layer_decay**(num_layers+1-i) for i in range(num_layers+2)])
    if assigner is not None: print(f"Assigned values={assigner.values}")
    skip_weight_decay_list=model.no_weight_decay()
    print(f"Skip weight decay list: {skip_weight_decay_list}")
    optimizer=create_optimizer(args, model, get_num_layer=assigner.get_layer_id if assigner is not None else None, 
                               get_layer_scale=assigner.get_scale if assigner is not None else None, 
                               filter_bias_and_bn=True, skip_list=skip_weight_decay_list)
    print("Use step level LR scheduler!")
    lr_schedule_values=cosine_scheduler(base_value=args.lr, final_value=args.min_lr, epochs=args.epochs, niter_per_ep=num_training_steps_per_epoch,
                                        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps)
    if args.weight_decay_end is None: args.weight_decay_end=args.weight_decay
    wd_schedule_values=cosine_scheduler(base_value=args.weight_decay, final_value=args.weight_decay_end, epochs=args.epochs, 
                                        niter_per_ep=num_training_steps_per_epoch)
    print(f"Max WD={max(wd_schedule_values):.7f}, Min WD={min(wd_schedule_values):.7f}")

    # loss
    if mixup_fn is not None: criterion=SoftTargetCrossEntropy() # smoothing is handled with mixup label transform
    elif args.smoothing>0.: criterion=LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else: criterion=torch.nn.CrossEntropyLoss()

    # checkpoint
    best_acc1=-float('inf')
    if checkpoint is not None:
        if 'optimizer' in checkpoint: optimizer.load_state_dict(checkpoint['optimizer'])
        if 'epoch' in checkpoint: args.start_epoch=checkpoint['epoch']+1
        if 'best_acc1' in checkpoint: best_acc1=checkpoint['best_acc1']
    
    print(f"Start training for {args.epochs} epoch at {args.start_epoch}")
    start_time=time.time()
    stop=False
    for epoch in range(args.start_epoch, args.epochs):
    
        if args.n_epochs is not None and epoch>args.n_epochs-1:
            print(f"Hit desired number of epochs {epoch}/{args.n_epochs}--break")
            break
                
        start_epoch_time=time.time()
        train_metric_logger=MetricLogger(delimiter=" ")
        train_stats=train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch, max_norm=args.clip_grad, mixup_fn=mixup_fn, 
                                    lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values, 
                                    num_training_steps_per_epoch=num_training_steps_per_epoch, metric_logger=train_metric_logger,
                                    print_freq=args.print_freq, n_steps=args.n_steps)
    
        val_metric_logger=MetricLogger(delimiter=" ")
        val_stats=validation_one_epoch(data_loader_val, model, device, metric_logger=val_metric_logger, print_freq=args.print_freq, n_steps=args.n_steps)
    
        if args.output_dir is not None:
            save_checkpoint(fpath=args.last, model=model, optimizer=optimizer, epoch=epoch, best_acc1=best_acc1)
            stats=form_stats(train_metric_logger, mode='train')|form_stats(val_metric_logger, mode='val')
            save_metrics(args.output_dir/"result.csv", stats, epoch=epoch, start_epoch_time=start_epoch_time)
            if args.plot_freq>0 and (epoch+1)%args.plot_freq==0: plot_all(args.output_dir/"result.csv")
    
        if best_acc1<val_metric_logger.meters['acc1'].global_avg:
            best_acc1=val_metric_logger.meters['acc1'].global_avg
            save_checkpoint(fpath=args.best, model=model, optimizer=optimizer, epoch=epoch, best_acc1=best_acc1)
    
        if args.time is not None:
            stop|=(time.time()-start_time)>(args.time*3600.)
        if stop: break
    
    total_time=time.time()-start_time
    total_time_str=str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")
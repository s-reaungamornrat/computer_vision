import time
import datetime

import numpy as np

import torch
import torch.nn as nn
import torchvision
from torchvision import tv_tensors
from torchvision.transforms import v2
from torch.utils.data import DataLoader, default_collate

from .plotting import plot_results
from .progress import MetricLogger, SmoothedValue, form_stats, save_metrics
from .metrics import accuracy
from .torch_utils import reduce_across_processes, initialize_weights, seed_worker, save_checkpoint
from computer_vision.torch_video.data.sampler import RandomClipSampler, UniformClipSampler
from computer_vision.torch_video.data.dataset import UCF101, ConvertTCHWtoCTHW, DATA_MEAN, DATA_STD, NUM_CLASSES

def train(args, device, model, model_without_ddp, criterion, optimizer, lr_scheduler, train_loader, val_loader, best_acc, train_sampler=None):
    
    
    stop=False
    for epoch in range(args.start_epoch, args.epochs):
        
        start_epoch_time=time.time()
        if args.distributed and train_sampler is not None: train_sampler.set_epoch(epoch)
            
        train_metric_logger=MetricLogger(delimiter=' ')
        stop|=train_one_epoch(model, criterion, optimizer, lr_scheduler, data_loader=train_loader, device=device, epoch=epoch, 
                              print_freq=args.print_freq, max_time=args.time, start_time=args.start_training_time, 
                              metric_logger=train_metric_logger, n_batches=args.n_batches)
        
        val_metric_logger=MetricLogger(delimiter=' ')
        acc1=evaluate(model, criterion, data_loader=val_loader, device=device, metric_logger=val_metric_logger,
                     n_batches=args.n_batches, distributed=args.distributed)

        # make sure that the model and model_without_ddp points to the same parameter values
        model_params={n:p.data for n, p in model_without_ddp.named_parameters()}
        assert all(torch.allclose(model_params[n], p.data) for n, p in model.named_parameters()), "`model` and `model_without_ddp` are not the same"

        if best_acc<acc1:
            best_acc=acc1
            if args.ouput_path: save_checkpoint(args.checkpoint_dirpath/args.best, model, optimizer, lr_scheduler, epoch, best_acc)
        if args.ouput_path:
            save_checkpoint(args.checkpoint_dirpath/args.last, model_without_ddp, optimizer, lr_scheduler, epoch, best_acc)
            # save training information
            stats=form_stats(train_metric_logger, mode='train')|form_stats(val_metric_logger, mode='val')
            save_metrics(args.ouput_path/"result.csv", stats, epoch=epoch, start_epoch_time=start_epoch_time)
            if args.plot_freq>0 and (epoch+1)%args.plot_freq==0: plot_results(args.ouput_path/"result.csv")

        if all(x is not None for x in [args.start_training_time,args.time]):
            stop|=(time.time()-args.start_training_time)>(args.time*3600)
            
        if stop: break

        

def setup_training(args, iters_per_epoch, n_classes=NUM_CLASSES, device=torch.device('cpu')):
    """Set up model, loss, optimizer, scaler and sheduler for training
    Args:
        iters_per_epoch (int): Number of batches/iterations per epoch
        n_classes (int): Number of classes
        device (torch.device): Computing device
    Returns:
        model (torch.nn.Module)
        optimizer (torch.optim)
        criterion (torch.nn.Module)
        lr_scheduler (torch.optim)
        scaler (torch.amp.GradScaler)
    """
    model=torchvision.models.get_model(args.model, weights=None)
    model.fc=nn.Linear(in_features=512, out_features=n_classes, bias=True) # modify to the right number of classes
    initialize_weights(model)
    nn.init.normal_(model.fc.weight, mean=0.0, std=0.01) # weights are also initialized to a small Gaussian
    nn.init.zeros_(model.fc.bias)
    # Below makes the initial softmax output match the dataset distribution before seeing any data.
    # but UCF101 is fairly balanced so gains are negligible; and therefore almost no papers bother
    # priors = class_counts / class_counts.sum()
    # model.fc.bias.data = torch.log(priors)
    
    model.to(device)
    if args.distributed and args.sync_bn: model=torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) # only support DDP
    
    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scaler=torch.amp.GradScaler() if (args.device=='cuda' and args.amp) else None
    
    # Convert scheduler to be per iteration, not per epoch, for warmup that lasts between different epochs
    lr_milestones=[iters_per_epoch*(m-args.lr_warmup_epochs) for m in args.lr_milestones]
    main_lr_scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=args.lr_gamma)
    
    if args.lr_warmup_epochs>0:
        warmup_iters=iters_per_epoch*args.lr_warmup_epochs
        args.lr_warmup_method=args.lr_warmup_method.lower()
        if args.lr_warmup_method=='linear':
            warmup_lr_scheduler=torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=args.lr_warmup_decay, total_iters=warmup_iters)
        elif args.lr_warmup_method=='constant':
            warmup_lr_scheduler=torch.optim.lr_scheduler.ConstantLR(optimizer, factor=args.lr_warmup_decay, total_iters=warmup_iters)
        else: raise RuntimeError(f"Invaid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supoorted")
        lr_scheduler=torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[warmup_iters])
    else: lr_scheduler=main_lr_scheduler

    return model, optimizer, criterion, lr_scheduler, scaler

def create_dataloader(args):
    """Set up dataset and data loader
    
    Returns:
        train_loader (torch.utils.data.DataLoader): Training data loader
        val_loader (torch.utils.data.DataLoader): Validation data loader
    """
    
    # Data loader
    train_dataset=UCF101(root=args.data_path, annotation_path=args.annotation_path, frame_rate=args.frame_rate, 
                         clip_duration=args.clip_duration, step_duration=args.step_duration,
                         train=True, metadata_path=args.metadata_path, fold=args.data_fold, sampling_type='random', use_audio=False,
                         decoder_transforms=[v2.RandomCrop(size=args.train_crop_size),
                                             v2.Resize(args.train_resize_size)],
                         transforms=v2.Compose([
                             v2.RandomHorizontalFlip(p=args.hflip_prob),
                             v2.ToDtype(torch.float32, scale=True),
                             v2.Normalize(mean=DATA_MEAN,std=DATA_STD),
                             ConvertTCHWtoCTHW()])
                        ) 
    
    val_dataset=UCF101(root=args.data_path, annotation_path=args.annotation_path, frame_rate=args.frame_rate, 
                         clip_duration=args.clip_duration, step_duration=args.step_duration,
                         train=False, metadata_path=args.metadata_path, fold=args.data_fold, sampling_type='regular', use_audio=False,
                         decoder_transforms=[v2.CenterCrop(size=args.train_crop_size),
                                             v2.Resize(args.train_resize_size)],
                         transforms=v2.Compose([
                             v2.ToDtype(torch.float32, scale=True),
                             v2.Normalize(mean=DATA_MEAN,std=DATA_STD),
                             ConvertTCHWtoCTHW()])
                        ) 
    
    if args.use_cutmix_mixup:
        cutmix=v2.CutMix(num_classes=NUM_CLASSES, alpha=args.cutmix_alpha)
        mixup=v2.MixUp(num_classes=NUM_CLASSES, alpha=args.mixup_alpha)
        cutmix_or_mixup=v2.RandomChoice([cutmix, mixup])
    
    def collate_fn(batch):
        
        output=default_collate(batch)
        if not args.use_cutmix_mixup: return output
            
        videos=tv_tensors.Video(output[0], dtype=output[0].dtype, device=output[0].device)
        labels=output[1] if len(output)==4 else output[2] # 4 outputs mean there is no audio
        return *cutmix_or_mixup(videos, labels), *output[-2:]
    
    train_sampler=RandomClipSampler(train_dataset.video_clip_metadata, args.max_n_clips_per_video)
    val_sampler=UniformClipSampler(val_dataset.video_clip_metadata, args.max_n_clips_per_video)
    if args.distributed:
        warning.warn("Not supported yet. Please see https://github.com/pytorch/vision/blob/main/torchvision/datasets/samplers/clip_sampler.py#L11")
    
    num_cuda=torch.cuda.device_count() # number of CUDA devices
    train_loader=DataLoader(train_dataset, batch_size=args.batch_size, sampler=None, # train_sampler,
                           num_workers=args.num_workers, pin_memory=num_cuda>0,
                           drop_last=len(train_dataset)%args.batch_size!=0, worker_init_fn=seed_worker,
                           collate_fn=collate_fn)
    val_loader=DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler,
                           num_workers=args.num_workers, pin_memory=num_cuda>0,
                           drop_last=len(train_dataset)%args.batch_size!=0, worker_init_fn=seed_worker,
                           collate_fn=None)
    return train_loader, val_loader

def train_one_epoch(model, criterion, optimizer, lr_scheduler, data_loader, device, epoch, print_freq, scaler=None, max_time=None, start_time=None,
                   metric_logger=None, n_batches=None):
    """Train a model for 1 epoch
    Args:
        model (torch.nn.Module): Deep learning model
        criterion (torch.nn.Module): Loss function
        optimizer (torch.optim): Optimizer
        lr_scheduler (torch.optim): Scheduler
        data_loader (torch.utils.data.dataloader.DataLoader): Data loader
        device (torch.device): Computing device
        epoch (int): Current epoch
        print_freq (int): Iteration print frequency
        scaler (torch.amp.GradScaler): Scaler for mix precision training
        max_time (float): Maximum training time in hours
        start_time (float): Time at the start of training program in seconds
        metric_logger (MetricLogger): Time monitoring
        n_batches (int): Number of batches to run before termination, for debugging purposes
    """
    stop=False # stop training
    
    model.train()
    # metric_logger=MetricLogger(delimiter==' ')
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("clips/s", SmoothedValue(window_size=10, fmt="{value:.3f}"))
    
    header=f"Epoch: [{epoch}]"
    for b, (video, target, _, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header, device=device)):
        
        if n_batches is not None and b>(n_batches-1): 
            print(f"Hit specified number of batches: {b}/{n_batches}! Terminate!!")
            #stop=True
            break
            
        start_time=time.time()
        video, target=video.to(device, non_blocking=device.type=='cuda'), target.to(device, non_blocking=device.type=='cuda')
        with torch.autocast(device_type="cuda", enabled=(scaler is not None and device.type=='cuda')):
            output=model(video)
            loss=criterion(output, target)
        optimizer.zero_grad()
    
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        acc1, acc5=accuracy(output, target, topk=(1,5))
        batch_size=video.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.meters['clips/s'].update(batch_size/(time.time()-start_time))
        lr_scheduler.step()       
        if all(x is not None for x in [start_time,max_time]):
            stop|=(time.time()-start_time)>(max_time*3600)
            if stop: return stop; break
                
    return stop

@torch.no_grad()
def evaluate(model, criterion, data_loader, device, print_freq=None, metric_logger=None, n_batches=None, distributed=False):
    """Evaluation
    Args:
        model (torch.nn.Module): Deep learning model
        criterion (torch.nn.Module): Loss function
        data_loader (torch.utils.data.dataloader.DataLoader): Data loader
        device (torch.device): Computing device
        print_freq (int): Iteration print frequency
        metric_logger (MetricLogger): Time monitoring
        n_batches (int): Number of batches to run before termination, for debugging purposes
        distributed (bool): Whether running as distributed system
    Returns:
        (int): Top 1 of global accuracy
    """
    model.eval()
    # metric_logger=MetricLogger(delimiter=' ')
    header="Eval: "
    
    num_processed_samples=0
    # Group and aggregate output of a video
    num_videos=len(data_loader.dataset.samples)
    num_classes=len(data_loader.dataset.classes)
    agg_preds=torch.zeros((num_videos, num_classes), dtype=torch.float32, device=device)
    agg_targets=torch.zeros((num_videos), dtype=torch.int32, device=device)
    if print_freq is None: print_freq=(len(data_loader)//4)-1
        
    with torch.inference_mode():
        for b, (video, target, video_idx, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header, device=device)):
            if n_batches is not None and b>(n_batches-1): 
                print(f"Hit specified number of batches: {b}/{n_batches}! Terminate!!")
                break
            
            video=video.to(device, non_blocking=device.type=='cuda') # (B,C,T,H,W)
            target=target.to(device, non_blocking=device.type=='cuda') # 
            output=model(video)
            loss=criterion(output, target)
    
            # Use softmax to comvert output into prediction probability
            preds=torch.softmax(output, dim=1)
            for b in range(video.size(0)):
                idx=video_idx[b].item()
                agg_preds[idx]+=preds[b].detach()
                agg_targets[idx]=target[b]
        
            acc1, acc5=accuracy(output, target, topk=(1,5))
            # FIXME need to take into account that the datasets could have been padded in the distributed setup
            batch_size=video.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples+=batch_size
            
        # gather the stats from all processes
        num_processed_samples=reduce_across_processes(num_processed_samples)
        # if isinstance(data_loader.sampler, DistributedSampler):
        #     # Get the len of UniformClipSampler inside DistributedSampler
        #     num_data_from_sampler=len(data_loader.sampler.dataset)
        # else: num_data_from_sampler=len(data_loader.sampler)
        num_data_from_sampler=len(data_loader.sampler)
        if (
            hasattr(data_loader.dataset, "__len__") and
            num_data_from_sampler!=num_processed_samples and
            distributed
        ):
            # See FIXME above
            warnings.warn(
                f"It looks like the sampler has {num_data_from_sampler} samples, but {num_processed_samples} "
                "samples were used for teh validation, which might bias the results. Try adjusting the batch size and/or the world size. "
                "Setting the world size to 1 is always a safe bet"
            )
        metric_logger.synchronize_between_processes()
        print(
            "* Clip Acc@1 {top1.global_avg:.3f} Clip Acc@5 {top5.global_avg:.3f}".format(
                top1=metric_logger.acc1, top5=metric_logger.acc5
            )
        )
        # Reduce the agg_preds and agg_targets from all gpu and show result
        agg_preds=reduce_across_processes(agg_preds)
        agg_targets=reduce_across_processes(agg_targets, op=torch.distributed.ReduceOp.MAX)
        agg_acc1, agg_acc5=accuracy(agg_preds, agg_targets, topk=(1,5))
        print("* Video Acc@1 {acc1:.3f} Video Acc@5 {acc5:.3f}".format(acc1=agg_acc1, acc5=agg_acc5))
        return metric_logger.acc1.global_avg
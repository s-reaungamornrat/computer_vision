import time
import math
import datetime
import warnings
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torchvision
from torchvision import tv_tensors
from torchvision.transforms import v2
from torch.utils.data import DataLoader, default_collate

from computer_vision.torch_video.utils.plotting import plot_all
from computer_vision.torch_video.utils.progress import MetricLogger, SmoothedValue, form_stats, save_metrics
from computer_vision.torch_video.utils.metrics import accuracy
from computer_vision.torch_video.utils.torch_utils import reduce_across_processes, initialize_weights, seed_worker, save_checkpoint, clear_memory, init_seeds,\
init_distributed_mode, load_checkpoint
from computer_vision.torch_video.data.dataset import UCF101, ConvertTCHWtoCTHW, DATA_MEAN, DATA_STD, NUM_CLASSES
from computer_vision.torch_video.data.sampler import RandomClipSampler, UniformClipSampler

class Trainer:
    def __init__(self,args):

        self.device=(torch.device(args.device) 
                if (args.device=='cuda' and torch.cuda.is_available() and torch.cuda.device_count()>0) 
                else torch.device('cpu') )

        self.args=args
        init_seeds(args.seed+1,deterministic=args.use_deterministic_algorithms)
        self.best_acc=-float('inf')
    
    def create_dataloader(self, args):
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
        
        self.train_sampler=RandomClipSampler(train_dataset.video_clip_metadata, args.max_n_clips_per_video)
        val_sampler=UniformClipSampler(val_dataset.video_clip_metadata, args.max_n_clips_per_video)
        if self.distributed:
            warning.warn("Not supported yet. Please see https://github.com/pytorch/vision/blob/main/torchvision/datasets/samplers/clip_sampler.py#L11")
        
        num_cuda=torch.cuda.device_count() # number of CUDA devices
        self.train_loader=DataLoader(train_dataset, batch_size=args.batch_size, sampler=None, # train_sampler,
                               num_workers=args.num_workers, pin_memory=num_cuda>0,
                               drop_last=len(train_dataset)%args.batch_size!=0, worker_init_fn=seed_worker,
                               collate_fn=collate_fn)
        self.val_loader=DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler,
                               num_workers=args.num_workers, pin_memory=num_cuda>0,
                               drop_last=len(train_dataset)%args.batch_size!=0, worker_init_fn=seed_worker,
                               collate_fn=None)

    def build_optimizer(self, n_classes, lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """Automatically construct an optimizer for the given model.
        Args:
            n_classes (int): Number of classes
            lr (float, optional): The learning rate for the optimizer
            momentum (float, optional): The momentum factor for the optimizer
            decay (float, optional): The weight decay for the optimizer
            iterations (float, optional): The number of iterations, which determines the optimizer if name is 'auto'
        """
        bn=tuple(v for k, v in nn.__dict__.items() if 'Norm' in k) # normalization layers, i.e., BatchNorm2d()
        # print(bn)
        
        g=[],[],[] # optimizer parameter groups
        # Automatically determine optimizer
        lr_fit=round(0.002*5/n_classes, 6)
        name, lr, momentum=('SGD', 0.01, 0.9) if iterations>10000 else ('Adam', lr_fit, 0.9)
        self.args.warmup_bias_lr=0.
        
        for module_name, module in self.model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                full_name=f'{module_name}.{param_name}' if module_name else param_name
                if 'bias' in full_name: g[2].append(param) # bias so no decay
                elif isinstance(module, bn): g[1].append(param) # weight not we do not apply decay
                else: g[0].append(param) # weight with decay
        
        if name=='Adam': self.optimizer=torch.optim.AdamW(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.)
        else: self.optimizer=torch.optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        self.optimizer.add_param_group({'params':g[0], 'weight_decay':decay})
        self.optimizer.add_param_group({'params':g[1], 'weight_decay':0.})
        print(f"'optimizer:' {type(self.optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
              f"{len(g[1])} weight(weight_decay=0.), {len(g[0])} weight(weight_decay={decay}), {len(g[2])} bias(weight_decay=0.)")

    def _setup_scheduler(self):
        """Initialize training learning rate scheduler"""
        self.lf=lambda x:max(1-x/self.args.epochs, 0)*(1.-self.args.lrf)+self.args.lrf # linear
        self.scheduler=torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def setup_training(self, args, iters_per_epoch, n_classes=NUM_CLASSES):
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
        self.model=torchvision.models.get_model(args.model, weights=None)
        self.model.fc=nn.Linear(in_features=512, out_features=n_classes, bias=True) # modify to the right number of classes
        initialize_weights(self.model)
        nn.init.normal_(self.model.fc.weight, mean=0.0, std=0.01) # weights are also initialized to a small Gaussian
        nn.init.zeros_(self.model.fc.bias)
        # Below makes the initial softmax output match the dataset distribution before seeing any data.
        # but UCF101 is fairly balanced so gains are negligible; and therefore almost no papers bother
        # priors = class_counts / class_counts.sum()
        # model.fc.bias.data = torch.log(priors)
        
        self.model.to(self.device)
        if args.distributed and args.sync_bn: self.model=torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model) # only support DDP
        
        self.criterion=nn.CrossEntropyLoss()
        # self.optimizer=torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        weight_decay=self.args.weight_decay*self.args.batch_size # scale weight decay
        iterations=math.ceil(len(self.train_loader.dataset)/self.args.batch_size)*self.args.epochs
        self.build_optimizer(n_classes=n_classes, lr=self.args.lr, momentum=self.args.momentum, decay=weight_decay, iterations=iterations)
        self.scaler=torch.amp.GradScaler() if (args.device=='cuda' and args.amp) else None

        self._setup_scheduler()
        # # Convert scheduler to be per iteration, not per epoch, for warmup that lasts between different epochs
        # lr_milestones=[iters_per_epoch*(m-args.lr_warmup_epochs) for m in args.lr_milestones]
        # main_lr_scheduler=torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=lr_milestones, gamma=args.lr_gamma)
        
        # if args.lr_warmup_epochs>0:
        #     warmup_iters=iters_per_epoch*args.lr_warmup_epochs
        #     args.lr_warmup_method=args.lr_warmup_method.lower()
        #     if args.lr_warmup_method=='linear':
        #         warmup_lr_scheduler=torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=args.lr_warmup_decay, total_iters=warmup_iters)
        #     elif args.lr_warmup_method=='constant':
        #         warmup_lr_scheduler=torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=args.lr_warmup_decay, total_iters=warmup_iters)
        #     else: raise RuntimeError(f"Invaid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supoorted")
        #     self.lr_scheduler=torch.optim.lr_scheduler.SequentialLR(self.optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], 
        #                                                             milestones=[warmup_iters])
        # else: self.lr_scheduler=main_lr_scheduler

    def train_one_epoch(self, epoch, print_freq, max_time=None, start_time=None, metric_logger=None, n_batches=None):
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

        nb=len(self.train_loader) # number of batches
        nw=max(round(self.args.warmup_epochs*nb), 100) if self.args.warmup_epochs>0 else -1 
        
        self.model.train()
        # metric_logger=MetricLogger(delimiter==' ')
        metric_logger.add_meter("lr0", SmoothedValue(window_size=1, fmt="{value}"))
        metric_logger.add_meter("lr1", SmoothedValue(window_size=1, fmt="{value}"))
        metric_logger.add_meter("lr2", SmoothedValue(window_size=1, fmt="{value}"))
        metric_logger.add_meter("clips/s", SmoothedValue(window_size=10, fmt="{value:.3f}"))
        
        header=f"Epoch: [{epoch}]"
        for b, (video, target, _, _) in enumerate(metric_logger.log_every(self.train_loader, print_freq, header, device=self.device)):
            
            if n_batches is not None and b>(n_batches-1): 
                print(f"Hit specified number of batches: {b}/{n_batches}! Terminate!!")
                #stop=True
                break

            start_time=time.time()

            # Warmup
            ni=b+nb*epoch
            if ni<=nw:
                xi=[0, nw]
                for j, x in enumerate(self.optimizer.param_groups):
                    x['lr']=np.interp(ni, xi, [self.args.warmup_bias_lr if j==0 else 0, x['initial_lr']*self.lf(epoch)])
                    if 'momentum' in x: x['momentum']=np.interp(ni,xi, [self.args.warmup_momentum, self.args.momentum])

            video=video.to(self.device, non_blocking=self.device.type=='cuda')
            target=target.to(self.device, non_blocking=self.device.type=='cuda')
            assert video.isfinite().all() and video.abs().sum()>0, f'video is Inf or NaN or blank'
            assert target.isfinite().all(), f'target is Inf or NaN'
            #with torch.autocast(device_type="cuda", enabled=(scaler is not None and device.type=='cuda')):
            output=self.model(video)
            self.loss=self.criterion(output, target)
            self.optimizer.zero_grad()

            if self.scaler is not None:
                self.scaler.scale(self.loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.)
                self.optimizer.step()
            acc1, acc5=accuracy(output, target, topk=(1,5))
            batch_size=video.shape[0]
            metric_logger.update(loss=self.loss.item())#, lr=self.optimizer.param_groups[0]['lr'])
            for j, x in enumerate(self.optimizer.param_groups): metric_logger.meters[f'lr{j}'].update(x['lr'])
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
            metric_logger.meters['clips/s'].update(batch_size/(time.time()-start_time))
            #self.lr_scheduler.step()       
            if all(x is not None for x in [start_time,max_time]):
                stop|=(time.time()-start_time)>(max_time*3600)
                if stop: return stop; break
                    
        return stop

    @torch.no_grad()
    def evaluate(self, print_freq=None, metric_logger=None, n_batches=None):
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
        self.model.eval()
        # metric_logger=MetricLogger(delimiter=' ')
        header="Eval: "
        
        num_processed_samples=0
        # Group and aggregate output of a video
        num_videos=len(self.val_loader.dataset.samples)
        num_classes=len(self.val_loader.dataset.classes)
        agg_preds=torch.zeros((num_videos, num_classes), dtype=torch.float32, device=self.device)
        agg_targets=torch.zeros((num_videos), dtype=torch.int32, device=self.device)
        if print_freq is None: print_freq=(len(self.val_loader)//4)-1
            
        with torch.inference_mode():
            for b, (video, target, video_idx, _) in enumerate(metric_logger.log_every(self.val_loader, print_freq, header, device=self.device)):
                if n_batches is not None and b>(n_batches-1): 
                    print(f"Hit specified number of batches: {b}/{n_batches}! Terminate!!")
                    break
                
                video=video.to(self.device, non_blocking=self.device.type=='cuda') # (B,C,T,H,W)
                target=target.to(self.device, non_blocking=self.device.type=='cuda') # 
                output=self.model(video)
                loss=self.criterion(output, target)
        
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
            num_data_from_sampler=len(self.val_loader.sampler)
            if (
                hasattr(self.val_loader.dataset, "__len__") and
                num_data_from_sampler!=num_processed_samples and
                self.distributed
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
            
    def _do_train(self):
        
        
        stop=False
        for epoch in range(self.args.start_epoch, self.args.epochs):
            
            start_epoch_time=time.time()
            
            with warnings.catch_warnings():
                warnings.simplefilter('ignore') # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()
                
            if self.distributed and self.train_sampler is not None: self.train_sampler.set_epoch(epoch)
                
            train_metric_logger=MetricLogger(delimiter=' ')
            stop|=self.train_one_epoch(epoch=epoch,  print_freq=self.args.print_freq, max_time=self.args.time, start_time=self.args.start_training_time, 
                                      metric_logger=train_metric_logger, n_batches=self.args.n_batches)
    
            clear_memory(device=self.device, threshold=0.5)
            val_metric_logger=MetricLogger(delimiter=' ')
            acc1=self.evaluate(metric_logger=val_metric_logger, n_batches=self.args.n_batches)
    
            # make sure that the model and model_without_ddp points to the same parameter values
            model_params={n:p.data for n, p in self.model_without_ddp.named_parameters()}
            assert all(torch.allclose(model_params[n], p.data) for n, p in self.model.named_parameters()), "`model` and `model_without_ddp` are not the same"

            if self._handle_nan_recovery(epoch): continue

            self.nan_recovery_attempts=0
            if self.best_acc<acc1:
                self.best_acc=acc1
                if self.args.ouput_path: save_checkpoint(self.args.checkpoint_dirpath/self.args.best, self.model, self.optimizer, 
                                                         self.scheduler, epoch, self.best_acc)
            if self.args.ouput_path:
                save_checkpoint(self.args.checkpoint_dirpath/self.args.last, self.model_without_ddp, self.optimizer, self.scheduler, epoch, self.best_acc)
                # save training information
                stats=form_stats(train_metric_logger, mode='train')|form_stats(val_metric_logger, mode='val')
                save_metrics(self.args.ouput_path/"result.csv", stats, epoch=epoch, start_epoch_time=start_epoch_time)
                if self.args.plot_freq>0 and (epoch+1)%self.args.plot_freq==0: plot_all(self.args.ouput_path/"result.csv")
    
            clear_memory(device=self.device, threshold=0.5) # clear if memory utilization>50%
            
            if all(x is not None for x in [self.args.start_training_time,self.args.time]):
                stop|=(time.time()-self.args.start_training_time)>(self.args.time*3600)
    
            if stop: break


    def train(self):

        self.args.start_training_time=time.time() # time at the start of training program
        self.args.ouput_path=Path(self.args.ouput_path)
        self.args.ouput_path.mkdir(parents=True, exist_ok=True)
        self.args.checkpoint_dirpath=self.args.ouput_path/"checkpoint"
        self.args.checkpoint_dirpath.mkdir(parents=True, exist_ok=True)
        
        
        init_distributed_mode(self.args)
        self.distributed=self.args.distributed

        self.create_dataloader(self.args)
        self.setup_training(self.args, iters_per_epoch=len(self.train_loader), n_classes=NUM_CLASSES)
        
        self.model_without_ddp=self.model
        if self.distributed:
            self.model=torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.args.gpu])
            self.model_without_ddp=self.model.module

        
        self.args.start_epoch=0
        if self.args.resume and (self.args.checkpoint_dirpath/self.args.last).is_file():
            self.args.start_epoch,self.best_acc=load_checkpoint(self.args.checkpoint_dirpath/self.args.last, self.model_without_ddp, self.optimizer, 
                                                                self.scheduler, self.scaler)
        
        start_time=time.time()
        print("Start training")
        
        self._do_train()
    
        clear_memory(device=self.device)
        
        if (self.args.ouput_path/"result.csv").is_file: plot_all(self.args.ouput_path/"result.csv")
            
        total_time=time.time()-start_time
        total_time_str=str(datetime.timedelta(seconds=int(total_time)))
        print(f"Training time {total_time_str}")

    def _handle_nan_recovery(self, epoch):
        """Detect and recover from NaN/Inf loss and fitness collapse by loading last checkpoint."""
        loss_nan=self.loss is not None and not self.loss.isfinite()
        if not loss_nan: return False
        if epoch==self.args.start_epoch or not (self.args.checkpoint_dirpath/self.args.last).exists():
            warnings.warn('Loss NaN/Inf detected but cannot recover from last.pt... since this is the first epoch or checkpoint file does not exist')
            return False # Cannot recover on first epoch and we let training continue
        self.nan_recovery_attempts+=1
        if self.nan_recovery_attempts>3:
            raise RuntimeError(f'Training failed: NaN persisted for {self.nan_recovery_attempts} epochs')
        warnings.warn(f'{reason} detected (attempted {self.nan_recovery_attempts}/3), recovering from last.pt...')
        self.model.train() # set model to train mode before loading checkpoints to avoid inference tensor errors
        self.args.start_epoch,self.best_acc=load_checkpoint(self.args.checkpoint_dirpath/self.args.last, self.model_without_ddp, self.optimizer, 
                                                            self.lr_scheduler, self.scaler)
        model_state=self.model.float().state_dict()
        if not all(torch.isfinite(v).all() for v in model_state.values() if isinstance(v, torch.Tensor)):
            raise RuntimeError(f'Checkpoint {self.last} is corrupted with NaN/Inf weights')
        self.scheduler.last_epoch=epoch-1
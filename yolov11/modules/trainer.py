from __future__ import annotations

from typing import Any
from pathlib import Path
from argparse import Namespace

import gc
import yaml
import time
import math
import copy
import numbers
import warnings
import datetime

import torch
import torch.nn as nn

import numpy as np

from .detector import DetectionModel
from .validator import DetectionValidator
from computer_vision.yolov11.utils.check import check_imgsz
from computer_vision.yolov11.data.dataset import YOLODataset
from computer_vision.yolov11.utils.plotting import plot_results, plot_images
from computer_vision.yolov11.utils.torch_utils import init_seeds, ModelEMA, one_cycle, EarlyStopping, unwrap_model, unset_deterministic

class DetectionTrainer:

    def __init__(self, args:Namespace, cfg:Path | str | dict, inch:int=3):
        """
        Initialize a DetectionTrainer object for training YOLO
        Args:
            args (Namespace): Training parameters
            cfg (str | dict): Configuration dict containing training parameters
            inch (int): The number of input channels
        """
         # Hyperparameters
        if isinstance(cfg, str): cfg=Path(cfg)
        if isinstance(cfg, Path):
            assert cfg.is_file(), f'{cfg} does not exist'
            with open(cfg) as f: self.cfg=yaml.load(f, Loader=yaml.SafeLoader)
        elif not isinstance(cfg, dict): raise TypeError(f'cfg must be dict/str but got {type(cfg)}')
        else: self.cfg=cfg

        data=args.data_cfg
        if isinstance(data, str): data=Path(data)
        if isinstance(data, Path):
            assert data.is_file(), f'{data} does not exist'
            with open(data, encoding="utf8") as f: self.data=yaml.load(f, Loader=yaml.SafeLoader)
        elif isinstance(data, dict): self.data=data

        self.inch=inch 
        # Merge the namespace without overriding the original args
        for k, v in vars(Namespace(**self.cfg)).items():
            if not hasattr(args, k): setattr(args, k, v)
        if Path(args.checkpoint_dirpath).is_dir():
            print(f'Resume training with checkpoint directory set to {args.checkpoint_dirpath}')
            args.resume=True
        self.args=args
        self.device=torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
        self.validator=None
        self.metrics=None
        self.plots=dict()
        init_seeds(seed=self.args.seed, deterministic=self.args.deterministic)

        # Directories
        self.save_dir=Path(self.args.output_dirpath)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.wdir=Path(self.args.checkpoint_dirpath) # weight directory
        self.wdir.mkdir(parents=True, exist_ok=True)
        self.last, self.best=self.wdir/self.args.latest_checkpoint, self.wdir/self.args.best_checkpoint
        self.save_period=self.args.save_period

        self.batch_size=self.args.batch_size
        # in case users accidentally pass epochs=None 
        self.epochs=self.args.epochs or 100 
        self.start_epoch=0

        # setting worker=0 yields faster CPU training as time dominated by inference, not dataloading
        if self.device.type in {'cpu', 'mps'}: self.args.worker=0

        # check data 
        self.args.root=Path(self.args.root)
        for dirname in [args.train_image_dirname, args.train_label_dirname, args.val_image_dirname, args.val_label_dirname]:
            assert (self.args.root/dirname).is_dir(), f'{Path(self.args.root)/dirname} does not exist'

        self.ema=None

        # Optimization utils init
        self.lf=None # learning rate fraction to be used by scheduler
        self.scheduler=None

        # Epoch level metrics
        self.best_fitness=None
        self.fitness=None
        self.loss=None
        self.tloss=None
        self.loss_names=["Loss"]
        self.csv=self.save_dir/"result.csv"
        if self.csv.exists() and not self.args.resume: self.csv.unlink(missing_ok=True)
        self.plot_idx=[0,1,2]
        self.nan_recovery_attempts=0 
        print('In modules.trainer.DetectionTrainer._init__ self.args.resume ', self.args.resume)

        self.world_size=0 # single GPU training

    def label_loss_items(self, loss_items:list[float]|None=None, prefix:str='train'):
        """
        Return a loss dict with labelled training loss items tensor
        Args:
            loss_items (list[float], optional): list of loss values
            prefix (str): Prefix for keys in the returned dict
        Returns:
            (dict | list): Dict of labeled loss items if loss_items is provided, otherwise list of keys
        Note: 
            This is not needed for classification but necessary for segmentation and detection
        """
        keys=[f'{prefix}/{x}' for x in self.loss_names]
        if loss_items is not None:
            # Convert tensors to 5 decimal place floats
            loss_items=[round(float(x), 5) for x in loss_items]
            return dict(zip(keys, loss_items))
        return keys

    def build_optimizer(self, model, name='auto', lr=0.001, momentum=0.9, decay=1.e-5, iterations=1e5):
        """
        Construct an optimizer for the given model
        Args:
            model (torch.nn.Module): The model for which to build an optimizer
            name (str, optional): The name of the optimizer to use. If `auto`, the optimizer is selected based on 
                the number of iterations
            lr (float, optional): The learning rate for the optimizer
            momentum (float, optional): The momentum factor for the optimizer
            decay (float, optional): The weight decay for the optimizer
            iterations (float, optional): The number of iterations, which determins the optimizer if name is `auto`
        Returns:
            (torch.nn.optim): The constructed optimizer
        """
        g=[],[],[] # optimizer parameter groups
        bn=tuple(v for k, v in nn.__dict__.items() if 'Norm' in k) # normalization layers, e.g., BatchNorm2d
        if name=='auto':
            print((f'optimizer: "optimizer=auto" found, ignoring "lr0={self.args.lr0}" and "momentum={self.args.momentum}" and '
                  f'determining best "optimizer", "lr0", and "momentum" automatrically....'))
            nc=self.data.get('nc', 10) # number of classes
            lr_fit=round(0.002*5/(4+nc), 6) # lr0 fit equation to 6 decimal places
            name, lr, momentum=('SGD', 0.01, 0.9) if iterations>10000 else ("AdamW", lr_fit, 0.9)
            self.args.warmup_bias_lr=0. # no higher than 0.01 for Adam
            
        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname=f'{module_name}.{param_name}' if module_name else param_name
                if 'bias' in fullname: # bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn) or "logit_scale" in fullname: # weight (no decay)
                    # ContrastiveHead and BNContrastiveHEad included here with `logit_scale`
                    g[1].append(param)
                else: # weight (with decay)
                    g[0].append(param)
        
        optimizers={'Adam', 'Adamax', 'AdamW', 'NAdam', 'RAdam', 'RMSProp', 'SGD', 'auto'}
        name={x.lower(): x for x in optimizers}.get(name.lower())
        if name in {'Adam', 'Adamax', 'AdamW', 'NAdam','RAdam'}:
            optimizer=getattr(torch.optim, name, torch.optim.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.)
        elif name=='RMSProp':  optimizer=torch.optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name=='SGD':  optimizer=torch.optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else: raise NotImplementedError(f'Optimizer "{name}" not found in list of supported optimizers {optimizers}')
        optimizer.add_param_group({'params':g[0], 'weight_decay':decay}) # add g0 with weight_decay
        optimizer.add_param_group({'params':g[1], 'weight_decay':0.}) # add g1 (normalization weights)
        
        print(f'optimizer: {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups: '
              f'{len(g[1])} weight(decy=0.), {len(g[0])} weight(decay={decay}), {len(g[2])} bias(decay=0.)' )
    
        return optimizer

    def _setup_scheduler(self):
        """
        Initialize training learning rate scheduler
        """
        # where lrf represents the final learning-rate fraction
        if self.args.cos_lr: self.lf=one_cycle(1, self.args.lrf, self.epochs) # cosine from 1 to hyp['lrf']
        else: self.lf=lambda x: max( 1 - x/self.epochs, 0)*(1.-self.args.lrf)+self.args.lrf # linear
        self.scheduler=torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def _load_checkpoint_state(self, checkpoint):
        """
        Load state_dict of optimizer, EMA, scheduler, scaler, etc.
        Args:
            checkpoint (dict): Dict of state_dict of training objects and training status information from the previous training round,
                containing keys such as 
                - `epoch` : previously trained epoch
                - `optimizer`: state_dict of optimizer so far
                - `scheduler`: state_dict of scheduler so far
                - `scaler`: state_dict of scaler so far if scaler is used
                - `ema`: state_dict of EMA and the number of updates
                - `best_fitness`: best fitness so far
        """
        
        if checkpoint.get('optimizer') is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint.get('scheduler') is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        if checkpoint.get('scaler') is not None:
            self.scaler.load_state_dict(checkpoint['scaler'])
        if self.ema and checkpoint.get('ema') is not None:
            # validation with EMA creates inference tensors that cannot be updated...
            self.ema=ModelEMA(self.model) 
            self.ema.ema.load_state_dict(checkpoint['ema'].float().state_dict())
            self.ema.updates=checkpoint['updates']
        self.best_fitness=checkpoint.get('best_fitness', 0.)
        
    def resume_training(self, checkpoint):
        """
        Resume YOLO training from given epoch and best fitness. If previos training reaches the maximum epochs `epochs`,
        the function set `stop` to True. This function does not update the state_dict of the training model. Assuming that
        has been done prior to initialize optimizer
        Args:
            checkpoint (dict): Dict of state_dict of training objects and training status information from the previous training round,
                containing keys such as 
                - `epoch` : previously trained epoch
                - `optimizer`: state_dict of optimizer so far
                - `scheduler`: state_dict of scheduler so far
                - `scaler`: state_dict of scaler so far if scaler is used
                - `ema`: state_dict of EMA and the number of updates
                - `best_fitness`: best fitness so far
        """
        if checkpoint is None or not self.args.resume: return
        start_epoch=checkpoint.get('epoch',-1)+1
        print(f'Resume training from epoch {start_epoch+1} to {self.epochs} total epochs')
        self._load_checkpoint_state(checkpoint)

        if self.epochs < start_epoch:
            print(f'Model completed training for {checkpoint["epoch"]} epochs.')
            self.stop=True
        self.start_epoch=start_epoch

    def _setup_train(self):
        """
        Build dataloaders, models, and optimizer and load their state_dicts if resume training.
        """
        # Build model and load previously-trained weights if resume
        self.model=DetectionModel(cfg=self.args.model_cfg, ch=self.inch)
        checkpoint=None
        if self.args.resume and self.last.is_file():
            checkpoint=torch.load(self.last, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model'])
        self.model=self.model.to(self.device)
        self.model.names=self.data["names"]
        
        always_freeze_names=['.dfl'] # always freeze these layers
        self.freeze_layer_names=always_freeze_names
        for k, v in self.model.named_parameters():
            if any(x in k for x in self.freeze_layer_names):
                print(f'In module.trainer.DetectionTrainer._setup_train: Freezing layer {k}')
                v.requires_grad=False
            elif not v.requires_grad and v.dtype.is_floating_point: 
                # only floating point can require gradients
                print(f'In module.trainer.DetectionTrainer._setup_train: Unfreeze layer {k}')
                v.requires_grad=True
        
        # Check imgsz
        gs=max( (int(self.model.stride.max()) if hasattr(self.model, 'stride') else 32), 32 ) # grid size / max stride
        self.args.imgsz=check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1) 
        self.stride=gs # for multiscale training
        
        # Dataloaders
        train_dataset=YOLODataset(img_path=(self.args.root/self.args.train_image_dirname),
                                  label_path=(self.args.root/self.args.train_label_dirname),
                                  data=self.data, hyp=self.cfg, imgsz=self.args.imgsz, cache=True, augment=True, rect=False,
                                  batch_size=self.args.batch_size, stride=gs, pad=0.5,  single_cls=False, classes=None, fraction=1.,
                                  channels=self.inch)
        val_dataset=YOLODataset(img_path=(self.args.root/self.args.val_image_dirname),
                                label_path=(self.args.root/self.args.val_label_dirname),
                                data=self.data, hyp=self.cfg, imgsz=self.args.imgsz, cache=True, augment=False, rect=False, 
                                batch_size=self.args.batch_size, stride=gs, pad=0.5,  single_cls=False, classes=None, fraction=1., channels=self.inch)
        self.train_loader=torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.args.batch_size, shuffle=False, sampler=None, batch_sampler=None, 
                                               num_workers=self.args.worker, collate_fn=YOLODataset.collate_fn, pin_memory=False, drop_last=True, 
                                               timeout=0, worker_init_fn=None, prefetch_factor=None, persistent_workers=False)
        self.val_loader=torch.utils.data.DataLoader(dataset=val_dataset, batch_size=self.args.batch_size, shuffle=False, sampler=None, batch_sampler=None, 
                                               num_workers=self.args.worker, collate_fn=YOLODataset.collate_fn, pin_memory=False, drop_last=False, 
                                               timeout=0, worker_init_fn=None, prefetch_factor=None, persistent_workers=False)
        # Validator
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        self.validator=DetectionValidator(hyperparam=self.cfg, data_cfg=self.data, dataloader=self.val_loader, save_dir=self.save_dir, 
                                          args=copy.deepcopy(self.args))
        self.validator.args.plots=self.args.plots
        metric_keys=self.validator.metrics.keys + self.label_loss_items(prefix='val')
        self.metrics=dict(zip(metric_keys, [0]*len(metric_keys)))
        self.ema=ModelEMA(self.model)
    
        # Optimizer
        self.accumulate=max(round(self.args.nbs/self.batch_size), 1) # accumulate loss before optimizing
        weight_decay=(self.args.weight_decay*self.batch_size*self.accumulate)/self.args.nbs # scale weight decay
        iterations=math.ceil(len(self.train_loader.dataset)/max(self.batch_size, self.args.nbs))*self.epochs
        self.optimizer=self.build_optimizer(model=self.model, name=self.args.optimizer, lr=self.args.lr0, 
                                                  momentum=self.args.momentum, decay=weight_decay, iterations=iterations)
        #Scheduler
        self._setup_scheduler()
        self.stopper, self.stop=EarlyStopping(patience=self.args.patience), False
        self.resume_training(checkpoint=checkpoint)
        self.scheduler.last_epoch=self.start_epoch-1
        # stop mosaic at trainer.args.close_mosaic before the end of training
        if self.start_epoch>(self.epochs-self.args.close_mosaic): 
            self._close_dataloader_mosaic()

    def _close_dataloader_mosaic(self):
        """ Update dataloaders to stop using mosaic augmentation """
        if hasattr(self.train_loader.dataset, 'mosaic'): 
            self.train_loader.dataset.mosaic=False
        if hasattr(self.train_loader.dataset, 'close_mosaic'):
            self.train_loader.dataset.close_mosaic(hyp=copy.deepcopy(self.args))

    def preprocess_batch(self, batch:dict)->dict:
        """
        Preprocess a batch of images by scaling and converting to float
        Args:
            batch (dict): Dict containing batch data with `img` tensor
        Returns:
            (dict): Preprocessed batch with normalized images
        Note: In Ultralytics implementation, this function allows multi-scale operation, but since
        it is disabled by default, we ignore it. 
        See https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/train.py#L111
        """
        for k, v in batch.items():
            if not isinstance(v, torch.Tensor): continue
            batch[k]=v.to(self.device, non_blocking=self.device.type=='cuda')
        batch['img']=batch['img'].float()/255
        return batch
        
    def _get_memory(self, fraction=False):
        """
        Get accelerator memory utilization in GB or as a fraction of total memory
        """
        memory,total=0,0
        if self.device.type=='mps':
            memory=torch.mps.driver_allocated_memory()
            if fraction: return __import__('psutil').virtual_memory().percent/100
        elif self.device.type!='cpu':
            memory=torch.cuda.memory_reserved()
            if fraction: 
                total=torch.cuda.get_device_properties(self.device).total_memory
        return ( (memory/total) if total>0 else 0 ) if fraction else (memory/2**30)

    def _model_train(self):
        """ Set to model to training mode """
        self.model.train()
        # Freeze BN stats if BN is in freeze layers (typically BN is not frozen layers, i.e., BN is not in self.freeze_layer_names)
        for n, m in self.model.named_modules():
            if any(filter(lambda f: f in n, self.freeze_layer_names)) and isinstance(m, nn.BatchNorm2d):
                m.eval()


    def _train_a_batch(self, i:int, nb:int, epoch:int, nw:int, batch:dict[str,Any], last_opt_step:int):
        """
        Args:
            i (int): Iteration
            nb (int): Number of batches, i.e., len(train_loader)
            epoch (int): Current epoch
            nw (int): Number of warmup iterations
            batch (dict[str,Any]): Training batch containing
                - `batch_idx` (torch.Tensor): (N,) image index of each item in the batch
                - `bboxes` (torch.Tensor): (N,4) bounding boxes in the normalized xywh format
                - `cls` (torch.Tensor): (N, 1) object classes
                - `im_file` (tuple[str]): filename of images in this batch
                - `img` (torch.Tensor): BxCxHxW where C is the number of image channels
                - `ori_shape` (tuple[tuple[int,int]]): Tuple of tuples of original (height, width) of all images in this batch 
                - `resized_shape` (tuple[tuple[int,int]]): Tuple of tuples of (height, width) of all input images in this batch 
            last_opt_step (int): Last iteration that the optimizer was updated
        Returns:
            last_opt_step (int): Last iteration that the optimizer was updated
        """
        start_it_time=time.time()
        # Warmup
        ni=i+nb*epoch
        if ni<=nw:
            xi=[0,nw]
            self.accumulate=max(1, int(np.interp(ni, xi, [1, self.args.nbs/self.batch_size]).round()))
            for j, x in enumerate(self.optimizer.param_groups):
                # if optimizer is not `auto`, bias lr falls from 0.1 to lr0 while all other lrs rise from 0. to lr0;
                # otherwise, all lrs rise from 0 to lr0
                x['lr']=np.interp(ni, xi, [self.args.warmup_bias_lr if j==0 else 0., x['initial_lr']*self.lf(epoch)])
                if 'momentum' in x: x['momentum']=np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])
        # Forward: if amp scaling is use the whole forward section has to be under with autocast(trainer.amp)
        batch=self.preprocess_batch(batch)
        loss, self.loss_items=self.model(batch, hyp=self.args)
        self.loss=loss.sum()
        # compute moving average loss
        self.tloss=self.loss_items if self.tloss is None else (self.tloss*i+self.loss_items)/(i+1)
        
        # Backward
        # self.scaler.scale(self.loss).backward()
        self.loss.backward()
        if ni-last_opt_step >= self.accumulate:
            self.optimizer.step()
            last_opt_step=ni
        
            # Timed stopping
            if self.args.time: # in hours
                self.stop=(time.time()-self.train_time_start)>(self.args.time*3600)
                # if trainer.stop: break # training time exceed we need to check this in the loop 
        
        # Log   
        loss_length=self.tloss.shape[0] if len(self.tloss.shape) else 1
        if isinstance(self.args.print_freq, numbers.Number) and (i+1)%self.args.print_freq==0:
            print('Epoch: {}, Iter: {}/{}({:.2f}%), Mem: {}GB, Box: {:.3f}, Cls: {:.3f}, DFL: {:.3f}, Time: {:.2f}s, N:{}, Img:{}'.format(epoch+1, i, nb,
                                          100*i/nb, self._get_memory(), self.tloss[0] if len(self.tloss)>0 else 0, self.tloss[1] if len(self.tloss)>1 else 0,
                                          self.tloss[2] if len(self.tloss)>2 else 0, time.time()-start_it_time, batch['cls'].shape[0], batch['img'].shape[-1]))
            # print(("%11s"*3 + "%11.4g"*(3+loss_length)) % (f'{epoch+1}/{self.epochs}', f'{i}/{nb}',
            #                                                f'{self._get_memory():.3g}G',  # (GB) GPU memory
            #                                                *(self.tloss if loss_length>1 else torch.unsqueeze(self.tloss, 0)), # losses
            #                                                time.time()-start_it_time, # time for this iteration
            #                                                batch['cls'].shape[0], # number of objects, e.g. 8
            #                                                batch['img'].shape[-1] # image size, e.g., 640
            #                                               ) )

        if self.args.plots and ni in self.plot_idx:
            print(f'In modules.trainer.DetectionTrainer._train_a_batch plot training samples: self.plot_idx {self.plot_idx}')
            self.plot_training_samples(batch, ni)
        return last_opt_step

    def _clear_memory(self, threshold:float=None):
        """
        Clear accelerator memory by calling garbage collector and emptying cache
        Args:
            threshold (float): Maximum fraction of memory usage allowed
        """
        if threshold:
            assert 0<= threshold<=1, f'Threshold must be between 0 and 1, but got {threshold}'
            if self._get_memory(fraction=True)<=threshold: return
        gc.collect()
        if self.device.type=='mps': torch.mps.empty_cache()
        elif self.device.type=='cpu': return
        else: torch.cuda.empty_cache()

    def validate(self):
        """
        Run validation on val set using self.validator
        Returns:
            metrics (dict): Dict of validation metrics
            fitness (float): Fitness score for validation
        """
        # if DDP, we have to sync EMA buffer from rank 0 to all ranks 
        # see https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/trainer.py#L690
        metrics=self.validator(trainer=self, hyp=self.args)
        if metrics is None: return None, None
        fitness=metrics.pop('fitness', -self.loss.detach().cpu().numpy()) # use loss as fitness measure if not found
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness=fitness
        return metrics, fitness
        
    def _handle_nan_recovery(self, epoch):
        """
        Detect and recover from NaN/Inf loss and fitness collapse by loading last checkpoint
        """
        loss_nan=self.loss is not None and not self.loss.isfinite()
        fitness_nan=self.fitness is not None and not np.isfinite(self.fitness)
        fitness_collapse=self.best_fitness and self.best_fitness>0 and self.fitness==0
        corrupted=loss_nan and (fitness_nan or fitness_collapse)
        reason='Loss NaN/Inf' if loss_nan else 'Fitness NaN/Inf' if fitness_nan else 'Fitness collapse'
        #if not corrupted: return False
        if epoch==self.start_epoch or not self.last.exists():
            warnings.warn(f'{reason} detected but cannot recover from last.pt since this is the first epoch; let trainig continue')
            print(f'In modules.trainer.DetectorTrainer._handle_nan_recovery self.best_fitness: {self.best_fitness}, self.fitness {self.fitness}',
                  ' self.best_fitness>0 ', self.best_fitness>0,  ' self.fitness==0 ', self.fitness==0, ' fitness_collapse ', fitness_collapse)
            return False # Cannot recover on first epoch, let training continue
        self.nan_recovery_attempts+=1
        if self.nan_recovery_attempts>3:
            raise RuntimeError(f'Training failed: NaN persisted for {self.nan_recovery_attempts} epochs')
        warnings.warn(f'{reason} detected (attempt {self.nan_recovery_attempts}/3), recovering from last.pt')
        self._model_train() # set model to train mode before loading checkpoints to avoid inference tensor error
        checkpoint=torch.load(self.last, map_location=self.device, weights_only=False)
        ema_state=checkpoint['ema'].float().state_dict()
        if not all(torch.isfinite(v).all() for v in ema_state.values() if isinstance(v, torch.Tensor)):
            raise RuntimeError(f'Checkpoint {self.last} is corrupted with NaN/Inf weights')
        unwrap_model(self.model).load_state_dict(ema_state) # Load EMA weights to model
        self._load_checkpoint_state(checkpoint) # load optimizer, scaler, scheduler, EMA, best_fitness
        del checkpoint, ema_state
        self.scheduler.last_epoch=epoch-1
        return True

    def save_metrics(self, metrics:dict[str, float]):
        """
        Save training metrics to a CSV file
        Args:
            metrics (dict[str, float]): Pairs of metric names and values
        """
        keys, vals=list(metrics.keys()), list(metrics.values())
        n=len(metrics)+2 # number of columns
        t=time.time()-self.train_time_start
        self.csv.parent.mkdir(parents=True, exist_ok=True) # ensure that parent directory exists
        s='' if self.csv.exists() else (("%s,"*n % tuple(['epoch', 'time']+keys)).rstrip(',')+'\n') # header
        with open(self.csv, 'a', encoding='utf-8') as f:
            f.write(s + ('%.6g,'*n % tuple([self.epoch+1, t]+vals)).rstrip(',') + '\n')

    def save_model(self):
        """
        Save model training checkpoints with additional metadata
        """    
        state_dict={'epoch':self.epoch, 'best_fitness':self.best_fitness, 'model':self.model.state_dict(),
                    'ema':copy.deepcopy(unwrap_model(self.ema.ema)), 'updates':self.ema.updates, 
                    'optimizer':copy.deepcopy(self.optimizer.state_dict()), 
                    'scheduler':copy.deepcopy(self.scheduler.state_dict()),
                    'train_args':vars(self.args), 'train_metrics':{**self.metrics, **{'fitness':self.fitness}},
                    'date':datetime.datetime.now().isoformat()}
        
        self.wdir.mkdir(parents=True, exist_ok=True) # ensure weights directory exists
        torch.save(state_dict, self.last)
        
        if self.best_fitness == self.fitness: torch.save(state_dict, self.best)

    def _epoch_log(self, epoch:int):
        """
        Create an epoch-log text for progress printing
        Args:
            epoch (int): Current epoch
        Returns:
            (str): Epoch training validation metrics and time
        """
        metric_maps={'precision':'P', 'recall':'R', 'fitness':'F'}
        log=f'Epoch: {epoch+1}/{self.epochs}, Time: {self.epoch_time/60:.2f} mins, F: {self.fitness:.2f}, '
        # We shorten the metrics keys from ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)',
        # 'val/box_loss', 'val/cls_loss', 'val/dfl_loss'] to [P, R, mAP50, mAP50-95, box, cls, dfl]
        for k, v in self.metrics.items():
            k=k.split('/')[-1].replace('(B)', '').replace('_loss', '')
            log+='{}:{:.2f}, '.format(metric_maps[k] if k in metric_maps else k, v)
        lr=[None,]*len(self.lr)
        for k, v in self.lr.items(): lr[int(k[-1])]=v
        log+='lr: ' + '-'.join(f'{x:.5f}' for x in lr)
        
        return log

    def final_eval(self):
        """
        Perform final evaluation and validation for object detection YOLO model
        """
        checkpoint=torch.load(self.best if self.best.exists() else self.last, map_location=self.device, weights_only=True)
        ema_state=checkpoint['ema'].float().state_dict()
        if not all(torch.isfinite(v).all() for v in ema_state.values() if isinstance(v, torch.Tensor)):
            raise RuntimeError(f'Checkpoint {self.best if self.best.exists() else self.last} is corrupted with NaN/Inf weights')
        unwrap_model(self.model).load_state_dict(ema_state) # Load EMA weights to model
        self.model.eval()
        del ema_state
        self.validator.args.plots=trainer.args.plots
        self.metrics=self.validator(model=self.model)
        self.metrics.pop('fitness', None)

    def plot_metrics(self):
        """
        Plot metrics from a CSV file
        """
        plot_results(file=self.csv) # save results.png

    def plot_training_samples(self, batch: dict[str, Any], ni: int) -> None:
        """
        Plot training samples with their annotations.

        Args:
            batch (dict[str, Any]): Dictionary containing batch data.
            ni (int): Number of iterations.
        """
        plot_images(
            labels=batch,
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg"
        )
        
    def train(self):
        """
        Train model for the specified maximum number of epochs
        Args: 
            epoch (int): Current epoch
        """
        self._setup_train()
        
        nb=len(self.train_loader) # number of batches
        nw=max(round(nb*self.args.warmup_epochs), 100) if self.args.warmup_epochs>0 else -1 # warmup iterations
        last_opt_step=-1 # schedule start position
        self.epoch_time=None
        self.epoch_time_start=time.time() # in seconds
        self.train_time_start=time.time()
        print(f'Image sizes {self.args.imgsz} train\n'
              f'Using {self.train_loader.num_workers * (self.world_size or 1)} dataloader workers\n'
              f'Logging results to {self.save_dir}\n'
              "Starting training for "+(f'{self.args.time} hours ...' if self.args.time else f'{self.epochs} epochs ...')
             )
        if self.args.close_mosaic>0: # number of epochs before the end to turning off mosaic
            base_idx=(self.epochs-self.args.close_mosaic)*nb # number of batches trained with mosaic
            self.plot_idx.extend([base_idx, base_idx+1, base_idx+2])
            
        epoch=self.start_epoch
        self.optimizer.zero_grad() # zero any resumed gradients 
        while True:
            self.epoch=epoch
            with warnings.catch_warnings():
                # Suppress `Detected lr_scheduler.step() nefore optimizer.step()`
                warnings.simplefilter('ignore') 
                self.scheduler.step()
        
            # Set to model to training mode
            self._model_train() # set model to train mode
            
            if epoch==(self.epochs-self.args.close_mosaic):
                self._close_dataloader_mosaic()
                # Do we need to call this to reset the data loader
                # self.train_loader.iterator = self.train_loader._get_iterator()
                
            self.tloss=None
            for i, batch in enumerate(self.train_loader):
                last_opt_step=self._train_a_batch(i=i, nb=nb, epoch=epoch, nw=nw, batch=batch, last_opt_step=last_opt_step)
                if self.stop: break # training time exceed 
        
            # for logger
            self.lr={f'lr/pg{ir}': x['lr'] for ir, x in enumerate(self.optimizer.param_groups)}
        
            final_epoch=epoch+1>=self.epochs
            self.ema.update_attr(self.model, include=['yaml', 'names', 'stride']) # we do not have nc, args, class_weights
            
            # Validation
            if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                self._clear_memory(threshold=0.5) # prevent VRAM spike
                self.metrics, self.fitness=self.validate()
        
            # NaN recovery
            if self._handle_nan_recovery(epoch): continue
            self.nan_recovery_attempts=0
        
            # Record metrics
            self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
        
            self.stop|=self.stopper(epoch+1, self.fitness) or final_epoch
            if self.args.time: self.stop |= (time.time()-self.train_time_start) > (self.args.time*3600)
            
            # Save model
            if self.args.save or final_epoch: self.save_model()
            
            # Scheduler
            t=time.time()
            self.epoch_time=t-self.epoch_time_start
            self.epoch_time_start=t
            # Below allow training based on training-time limit instead of maximum number of epochs
            # if trainer.args.time: 
            #     mean_epoch_time=(t-trainer.train_time_start)/(epoch-trainer.start_epoch+1) # time requires to run each epoch
            #     trainer.epochs=trainer.args.epochs=math.ceil(trainer.args.time*3600/mean_epoch_time) # updated maximum number of epochs
            #     trainer._setup_scheduler() # recreate a scheduler based on the new maximum number of epochs
            #     trainer.scheduler.last_epoch=trainer.epoch # resume from current epoch of LR curve
            #     trainer.stop |= epoch >=trainer.epochs # stop if exceed epochs
            
            # Early stopping: for DDP, must break all rank see 
            # https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/trainer.py#L355

            # Print epoch
            print(self._epoch_log(epoch), flush=True)
            
            if self.stop: break
            epoch+=1

        seconds=time.time()-self.train_time_start
        print(f'\n{epoch-self.start_epoch+1} epochs completed in {seconds/3600:.3f} hours')

        # Do final val with best.pt
        self.final_eval()
        if self.args.plots: self.plot_metrics()
        self._clear_memory()
        unset_deterministic()
from __future__ import annotations

import gc
import math
import os
import yaml
import time
import warnings
from copy import copy, deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim

from computer_vision.yolov11_pose.utils import DEFAULT_CFG_DICT
from computer_vision.yolov11_pose.cfg import get_cfg
from computer_vision.yolov11_pose.utils.files import get_latest_run
from computer_vision.yolov11_pose.utils.torch_utils import init_seeds, unwrap_model, one_cycle, EarlyStopping, unset_deterministic
from computer_vision.yolov11_pose.data.utils import check_det_dataset
from computer_vision.yolov11_pose.nn.tasks import load_checkpoint
from computer_vision.yolov11_pose.utils.checks import check_imgsz
from computer_vision.yolov11_pose.data.build import build_yolo_dataset, build_dataloader
from computer_vision.yolov11_pose.utils.plotting import plot_labels, plot_images, plot_results

class DetectionTrainer:
    """A base class for creating trainers

    This class provides the foundation for training YOLO models, handling the training loop, validation, checkpointing, and various training utilities.

    Examples:
        >>> trainer=BaseTrainer(cfg='config.yaml')
        >>> trainer.train()
    """
    def __init__(self, cfg=DEFAULT_CFG_DICT, overrides=None):
        """Initialize the BaseTrainer class
        Args:
            cfg (str | dict, optional): Path to a configuration file or a configuration dict
            overrides (dict, optional): Configuration overrides
        """

        self.model_args=get_cfg(cfg, overrides=overrides)
        # We only add settings for those not in overrides
        overrides=vars(overrides)
        for key, val in DEFAULT_CFG_DICT.items():
            if key in overrides: continue
            overrides[key]=val
        self.args=get_cfg(cfg, overrides=overrides)
        self.args.nc=self.model_args.nc
        self.check_resume()
        self.device=(torch.device(self.args.device) 
                     if (self.args.device=='cuda' and torch.cuda.is_available() and torch.cuda.device_count()>0) 
                     else torch.device('cpu') )
        self.validator=None
        self.metrics=None
        self.plots={}
        init_seeds(self.args.seed+1,deterministic=self.args.deterministic)

        self.save_dir=Path(self.args.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.wdir=Path(self.args.checkpoint_dir)
        self.wdir.mkdir(parents=True, exist_ok=True)
        # Save run args, serializing aumentations as reprs for resume compatibility
        args_dict=vars(self.args).copy()
        with open(self.save_dir/'args.yaml', 'w') as outfile: # sort_keys=False preserves key order in Python 3.7+
            yaml.dump(args_dict, outfile, default_flow_style=False, sort_keys=False) 
        self.last, self.best=self.wdir/'last.pt', self.wdir/'best.pt' # checkpoint paths
        self.save_period=self.args.save_period

        self.batch_size=self.args.batch_size
        self.epochs=self.args.epochs or 100 # in case users accidentally pass epochs=None with timed training
        self.start_epoch=0

        # Device
        if self.device.type in {'cpu', 'mps'}:
            self.args.workers=0 # faster CPU training as time dominated by inference, not dataloading

        # Dataset
        print(f'In engine.trainer.DetectionTrainer.__init__ self.args.data {self.args.data}')
        self.data=self.get_dataset()
        self.model_args.nc=self.data['nc'] 
        self.args.nc=self.data['nc']
        # merge data configuration with the hyperparameter and model configuration parameters
        for key, val in self.data.items():
            if not hasattr(self.model_args, key): setattr(self.model_args, key, val)
            if not hasattr(self.args, key): setattr(self.args, key, val)
                
        self.ema=None
        self.model=None

        # Optimization utils init
        self.lf=None # learning rate adjustment function
        self.scheduler=None

        # Epoch level metrics
        self.best_fitness=None
        self.fitness=None
        self.loss=None
        self.tloss=None
        self.loss_names=['Loss']
        self.csv=self.save_dir/'results.csv'
        if self.csv.exists() and not self.args.resume: self.csv.unlink()
        self.plot_idx=[0,1,2]
        self.nan_recovery_attempts=0

        self.world_size=1
        
        
    def check_resume(self):
        """Check if resume checkpoint exists"""
        resume=self.args.resume
        if resume: 
            exists=Path(resume).exists() if isinstance(resume, (str, Path)) else False
            last=Path(resume if exists else get_latest_run(self.args.checkpoint_dir))
            resume=True
            self.args.model=self.args.resume=str(last)
        self.resume=resume

    def get_dataset(self):
        """Get train and validation datasets from data dictionary
        Returns:
            (dict): Dict containing the training/validation/test dataset and category names
        """
        if self.args.task=='classify':  raise NotImplementedError()
        else: data=check_det_dataset(self.args.data)
        if self.args.train_image_dir is not None:
            assert os.path.isdir(self.args.train_image_dir), f'{self.args.train_image_dir} must be a directory'
            data['train']=self.args.train_image_dir 
        if self.args.val_image_dir is not None:
            assert os.path.isdir(self.args.val_image_dir), f'{self.args.val_image_dir} must be a directory'
            data['val']=self.args.val_image_dir
        if self.args.single_cls:
            print('In engine.trainer.DetectionTrainer.get_datatset: Overriding class names with single class')
            data['names']={0:'item'}
            data['nc']=1
        return data

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Get model and raise NotImplementError for loading cfg files"""
        raise NotImplementError('This task trainer does not support loading cfg files')

    def setup_model(self, verbose=True):
        """Load or create model for any tasks
        Args:
            verbose (bool):  Whether to display model information
        Returns:
            (dict): Optional checkpoint to resume training from
        """
        if isinstance(self.model, torch.nn.Module): # if the model is loaded beforehand, no setup is needed
            return
            
        ckpt=None
        if self.resume:
            if self.last.is_file(): ckpt=load_checkpoint(self.last)
            elif self.best.is_file(): ckpt=load_checkpoint(self.best)
        self.model=self.get_model(cfg=self.model_args, weights=ckpt['model'] if ckpt is not None else None, verbose=verbose)
        return ckpt

    def set_model_attributes(self):
        """Set model attributes based on dataset information"""
        self.model.nc=self.data['nc'] # attach number of classes to model
        self.model.names=self.data['names'] # attach class names to the model 
        self.model.args=self.args # attach hyperparameters to the model

    def get_dataloader(self, dataset_path:str, batch_size:int=16, mode:str='train'):
        """Construct and return dataloader for the specified mode
        Args:
            dataset_path (str): Path to the dataset
            batch_size (int): Number of images per batch
            mode (str): 'train' for training dataloader and 'val' for validation dataloader
        Returns:
            (DataLoader): PyTorch dataloader object
        """
        assert mode in {'train', 'val'}, f"Mode must be 'train' or 'val', not {mode}"
        grid_size=max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
        dataset=build_yolo_dataset(args=self.args, cfg=self.args, task=self.args.task, img_path=dataset_path,
                                   batch=batch_size,  data=self.args.data,  mode=mode, rect=mode=='val', stride=grid_size,
                                   channels=self.args.channels)
        shuffle=mode=='train'
        if getattr(dataset, 'rect', False) and shuffle:
            warnings.warn("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False" )
            shuffle=False
        return build_dataloader(dataset, batch=batch_size, workers=self.args.workers if mode=='train' else self.args.workers*2, 
                                shuffle=shuffle, drop_last=mode=='train', pin_memory=True)

    def get_validator(self):
        """Raise NotImplementError (must be implemented by subclasses)"""
        raise NotImplementedError("get_validator function not implemented in trainer")

    def label_loss_items(self, loss_items:list[float]|None=None, prefix:str='train'):
        """Return a loss dict with labeled training loss items tensor
        Args:
            loss_items (list[float], optional): List of loss values
            prefix (str): Prefix for keys in the returned dict
        Returns:
            (dict|list): Dict of labeled loss items if loss_items is provided, otherwise, list of keys
        """
        keys=[f'{prefix}/{x}' for x in self.loss_names]
        if loss_items is not None:
            loss_items=[round(float(x), 5) for x in loss_items] # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        return keys

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model"""
        boxes=np.concatenate([lb['bboxes'] for lb in self.train_loader.dataset.labels], 0)
        cls=np.concatenate([lb['cls'] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data['names'], save_dir=self.save_dir)

    def build_optimizer(self, model, name='auto', lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """Construct an optimizer for the given model.
        Args:
            model (torch.nn.Module): The model for which to build an optimizer
            name (str, optional): The name of the optimizer to use. If 'auto', the optimizer is selected based on the number of iterations
            lr (float, optional): The learning rate for the optimizer
            momentum (float, optional): The momentum factor for the optimizer
            decay (float, optional): The weight decay for the optimizer
            iterations (float, optional): The number of iterations, which determines the optimizer if name is 'auto'
        Returns:
            (torch.optim.Optimizer): The constructed optimizer
        """
        g=[],[],[] # optimizer parameter groups
        bn=tuple(v for k, v in nn.__dict__.items() if 'Norm' in k) # normalization layers, i.e., BatchNorm2d()
        if name=='auto':
            print(f"'optimizer=auto' found, "
                  f"ignoring 'lr0={self.args.lr0}' and 'momentum={self.args.momentum}' and "
                  f"determining best 'optimizer', 'lr0', and 'momentum' automatically....")
            nc=self.data.get('nc', 10) # number of classes
            # The following line is a safety mechanism. It acts as a "smart default" that prevents the optimizer from taking steps that are 
            # too aggressive when dealing with a high number of object categories, i.e., higher number of classes lower learning rate
            lr_fit=round(0.002*5/(4+nc), 6) # lr0 fit equation to 6 decimal places
            name, lr, momentum=('SGD', 0.01, 0.9) if iterations>10000 else ('AdamW', lr_fit, 0.9)
            self.args.warmup_bias_lr=0. # no higher than 0.01 for Adam
        
        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                full_name=f'{module_name}.{param_name}' if module_name else param_name
                if 'bias' in full_name: # bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn) or 'logit_scale' in full_name: # weight (no decay)
                    # logit_scale is a specialized parameter typically found in multimodal models (like CLIP)
                    # or modern Vision Transformers that use cosine similarity attention
                    # ContrastiveHead and BNContrastiveHead included here with 'logit_scale'
                    g[1].append(param)
                else: # weight (with decay)
                    g[0].append(param)
        optimizers={'Adam', 'Adamax','AdamW', 'NAdam', 'RAdam', 'RMSProp', 'SGD', 'auto'}
        name={x.lower():x for x in optimizers}.get(name.lower())
        if name in {'Adam', 'Adamax', 'AdamW', 'NAdam', 'RAdam'}:
            optimizer=getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.)
        elif name=='RMSProp':
            optimizer=optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name=='SGD':
            optimizer=optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(f"Optimizer '{name}' not found in list of available optimizers {optimizers}.")
        optimizer.add_param_group({'params':g[0], 'weight_decay':decay}) # add g0 with weight_decay
        optimizer.add_param_group({'params':g[1], 'weight_decay':0.}) # add g1 (BatchNorm2d weights)
        print(f"'optimizer:' {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups"
              f"{len(g[1])} weight(decay=0.), {len(g[0])} weight(decay={decay}), {len(g[2])} bias(decay=0.)")
        return optimizer

    def _setup_scheduler(self):
        """Initialize training learning rate scheduler"""
        if self.args.cos_lr: self.lf=one_cycle(1, self.args.lrf, self.epochs) # cosine 1->hyp['lrf']
        else: self.lf=lambda x:max(1-x/self.epochs, 0)*(1.-self.args.lrf)+self.args.lrf # linear
        self.scheduler=optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
            

    def _load_checkpoint_state(self, ckpt):
        """Load optimizer, scaler, and best_fitness from checkpoint"""
        if ckpt.get('optimizer') is not None: self.optimizer.load_state_dict(ckpt['optimizer'])
        if ckpt.get('scaler') is not None: self.scaler.load_state_dict(ckpt['scaler'])
        self.best_fitness=ckpt.get('best_fitness', 0)

    def _close_dataloader_mosaic(self):
        """Update dataloaders to stop using mosaic augmentation"""
        if hasattr(self.train_loader.dataset, 'mosaic'): self.train_loader.dataset.mosaic=False
        if hasattr(self.train_loader.dataset, 'close_mosaic'): 
            print('In engine.trainer.DetectionTrainer._close_dataloader_mosaic: closing dataloader mosaic')
            self.train_loader.dataset.close_mosaic(hyp=deepcopy(self.args))

    def resume_training(self, ckpt):
        """Resume YOLO training from given epoch and best fitness."""
        if ckpt is None or not self.resume: return
        start_epoch=ckpt.get('epoch', -1)+1
        assert start_epoch>0, (
            f"ckpt {ckpt.keys()} does not contains 'epoch', must mean training complete and the model was save without epoch.\n"
            f"Start a new training without resuming, i.e., 'yolo train model={self.args.model}'"
        )
        print(f'Resuming training {self.args.model} from epoch {start_epoch} to {self.epochs} total epochs')
        if self.epochs < start_epoch:
            print(f"{self.model} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {self.epochs} more epochs")
            trainer.epochs+=ckpt['epoch'] # finetune additional epochs
        self._load_checkpoint_state(ckpt)
        self.start_epoch=start_epoch
        if start_epoch>(self.epochs-self.args.close_mosaic): self._close_dataloader_mosaic()
    
    def _setup_train(self):
        """Build dataloader and optimizer"""
        ckpt=self.setup_model()
        self.model=self.model.to(self.device)
        self.set_model_attributes()
        
        # Freeze layers
        freeze_list=(self.args.freeze if isinstance(self.args.freeze, list)
                     else range(self.args.freeze)
                     if isinstance(self.args.freeze, int)
                     else [])
        always_freeze_names=['.dfl'] # always freeze these layers
        freeze_layer_names=[f'model.{x}.' for x in freeze_list]+always_freeze_names
        self.freeze_layer_names=freeze_layer_names
    
        for k, v in self.model.named_parameters():
            if any(x in k for x in freeze_layer_names):
                warnings.warn(f"Freezing layer '{k}'")
                v.requires_grad=False
            elif not v.requires_grad and v.dtype.is_floating_point: # only floating point tensor can require gradients
                warnings.warn(f"setting 'requires_grad=True' for frozen layer '{k}'")
                v.requires_grad=True
        
        # Check imgsz
        gs=max(int(self.model.stride.max() if hasattr(self.model, 'stride') else 32), 32) # grid size (max stride)
        self.args.imgsz=check_imgsz(self.args.imgsz, stride=gs, max_dim=1, floor=gs)
        self.stride=gs # for multiscale training
        
        # Dataloader 
        batch_size=self.batch_size//max(self.world_size, 1)
        self.train_loader=self.get_dataloader(self.data["train"], batch_size=batch_size, mode="train")
        # Note: When training DOTA dataset, double batch size could get OOM on images with >2000 objects
        self.test_loader=self.get_dataloader(self.data.get('val') or self.data.get('test'),
                                             batch_size=batch_size if self.args.task=='obb' else batch_size*2, mode='val')
        
        self.validator=self.get_validator()
        metric_keys=self.validator.metrics.keys+self.label_loss_items(prefix='val')
        self.metrics=dict(zip(metric_keys, [0]*len(metric_keys)))
        if self.args.plots: self.plot_training_labels()
        
        # Optimizer
        self.accumulate=max(round(self.args.nbs/self.batch_size),1) # accumulate loss before optimizing
        weight_decay=self.args.weight_decay*self.batch_size*self.accumulate/self.args.nbs # scale weight decay
        iterations=math.ceil(len(self.train_loader.dataset)/max(self.batch_size, self.args.nbs))*self.epochs
        self.optimizer=self.build_optimizer(model=self.model, name=self.args.optimizer, lr=self.args.lr0, 
                                            momentum=self.args.momentum, decay=weight_decay, iterations=iterations)
        
        # Scheduler
        self._setup_scheduler()
        self.stopper, self.stop=EarlyStopping(patience=self.args.patience), False
        self.resume_training(ckpt)
        self.scheduler.last_epoch=self.start_epoch-1 # do not move

    def _model_train(self):
        """Set model in training mode"""
        self.model.train()
        # Freeze BN stat
        for n, m in self.model.named_modules():
            if any(filter(lambda f: f in n, self.freeze_layer_names)) and isinstance(m, nn.BatchNorm2d):
                m.eval()

    def preprocess_batch(self, batch:dict)->dict:
        """Preprocess a batch of images by scaling and converting to float.
        Args:
            batch (dict): Dict containing batch data with 'img' tensor
        Returns:
            (dict): Preprocessed batch with normalized images
        """
        for k, v in batch.items():
            if isinstance(v, torch.Tensor): batch[k]=v.to(self.device, non_blocking=self.device.type=='cuda')
        batch['img']=batch['img'].float()/255
        if self.args.multi_scale:
            imgs=batch['img']
            sz=(
                random.randrange(int(self.args.imgsz*0.5), int(self.args.imgsz*1.5+self.stride))
                //self.stride
                *self.stride
            )
            sf=sz/max(imgs.shape[2:]) # scale factor
            if sf!=1:
                ns=[
                    math.ceil(x*sf/self.stride)*self.stride for x in imgs.shape[2:]
                ]# new shape (stretched to grid-shape multiple)
                imgs=nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)
            batch['img']=imgs
        return batch

    def optimizer_step(self):
        """Perform a single step of the training optimizer with gradient clipping and EMA update."""
        #self.scaler.unscale_(self.optimizer) # unscale gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.)
        #self.scaler.step(self.optimizer)
        #self.scaler.update()
        self.optimizer.step()
        self.optimizer.zero_grad()
        #if self.ema: self.ema.update(self.model)

    def _get_memory(self, fraction=False):
        """Get accelerator memory utilization in GB or as a fraction of total memory"""
        memory, total=0,0
        if self.device.type=='mps':
            memory=torch.mps.driver_allocated_memory()
            if fraction: return __import__("psutil").virtual_memory().percent/100
        elif self.device.type!='cpu':
            memory=torch.cuda.memory_reserved()
            if fraction: total=torch.cuda.get_device_properties(self.device).total_memory
        return ((memory/total) if total>0 else 0) if fraction else (memory/2**30)

    def plot_training_samples(self, batch:dict[str, Any], ni:int)->None:
        """Plot training samples with their annotations.
        Args:
            batch (dict[str, Any]): Dict containing batch data
            ni (int): Number of iterations/batch items trained so far
        """
        plot_images(labels=batch, paths=batch['im_file'], fname=self.save_dir/f"train_batch{ni}.jpg")

    def train_each_epoch(self, epoch, last_opt_step):
        """Training of each epoch
        Args:
            epoch (int): Current epoch
            last_opt_step (int): Last iteration that gradients were accumulated
        Returns:
            (int): Last iteration that gradients were accumulated
        """
        nb=len(self.train_loader) # number of batches
        # self.args.warmup_epochs is the number of epochs spent warming up the optimizer
        # we multiply self.args.warmup_epochs by nb since training updates happen per batch (iteration)
        # warm up over nw optimization steps/iterations, i.e., nw is the number of time we will perform learning-rate adjustment for warmup
        nw=max(round(self.args.warmup_epochs*nb), 100) if self.args.warmup_epochs>0 else -1 
        
        self.tloss=None
        for i, batch in enumerate(self.train_loader):
            # Warmup
            ni=i+nb*epoch # current number of iterations/batch items since epoch counts from 0 and nb is the total number of batches 
            if ni<=nw:
                xi=[0, nw] 
                # controling how often to accumulate
                # During warmup, starting with frequent small nominal batch size (i.e., accumulate more often) and moving toward the stable, 
                # larger nominal batch size.
                self.accumulate=max(1, int(np.interp(ni, xi, [1, self.args.nbs/self.batch_size]).round()))
                for j, x in enumerate(self.optimizer.param_groups):
                    # Bias lr falls from 0.1 to lr0, all others lrs rise from 0. to lr0
                    x['lr']=np.interp(ni, xi, [self.args.warmup_bias_lr if j==0 else 0.0, x['initial_lr']*self.lf(epoch)])
                    if 'momentum' in x:
                        x['momentum']=np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])
            # Forward
            batch=self.preprocess_batch(batch)
            if self.args.compile:
                # Decouple inference and loss calculations for improved compile performance
                preds=self.model(batch['img'])
                loss, self.loss_items=unwrap_model(self.model).loss(batch, preds)
            else: loss, self.loss_items=self.model(batch)
            self.loss=loss.sum()
            self.tloss=self.loss_items if self.tloss is None else (self.tloss*i+self.loss_items)/(i+1)
        
            # Backward
            # trainer.scaler.scale(trainer.loss).backward()
            self.loss.backward()
            if ni-last_opt_step>=self.accumulate:
                self.optimizer_step()
                last_opt_step=ni
        
                #Time stopping
                if self.args.time:
                    self.stop=(time.time()-self.train_time_start)>(self.args.time*3600)
                    if self.stop: break
            # Log
            loss_length=self.tloss.shape[0] if len(self.tloss.shape) else 1
            if self.args.print_freq is not None and self.args.print_freq>0 and ni>0 and ni%self.args.print_freq==0:
                print(("%11s"*2+"%11.4g"*(2+loss_length)) % (
                    f"{epoch+1}/{self.epochs}",
                    f"{self._get_memory():.3g}G", # (GB) GPU memory utilization
                    *(self.tloss if loss_length>1 else torch.unsqueeze(self.tloss, 0)), # losses
                    batch['cls'].shape[0], # batch size, e.g., 8
                    batch['img'].shape[-1], # imgsz, e.g., 640
                ))
            if self.args.plots and ni in self.plot_idx: self.plot_training_samples(batch, ni)
        return last_opt_step

    def _clear_memory(self, threshold:float|None=None):
        """Clear accelerator memory by calling garbage collector and emptying cache
        Args:
            threshold (float, optional): If provided, memory will be clear only the fraction of memory used above the input threshold
        """
        if threshold:
            assert 0<=threshold<=1, "Threshold must be between 0 and 1"
            if self._get_memory(fraction=True)<=threshold: return
        gc.collect()
        if self.device.type=='mps': torch.mps.empty_cache()
        elif self.device.type=='cpu': return
        else: torch.cuda.empty_cache()
            
    def validate(self):
        """Run validation on val set using self.validator
        
        Returns:
            metrics (dict): Dict of validation metrics
            fitness (float): Fitness score for the validation
        """
        metrics=self.validator(trainer=self)
        if metrics is None: return None, None
        fitness=metrics.pop('fitness', -self.loss.detach().cpu().numpy()) # use loss as fitness measure if not found
        if not self.best_fitness or self.best_fitness<fitness: self.best_fitness=fitness
        return metrics, fitness

    def _handle_nan_recovery(self, epoch):
        """Detect and recover from NaN/Inf loss and fitness collapse by loading last checkpoint."""
        loss_nan=self.loss is not None and not self.loss.isfinite()
        fitness_nan=self.fitness is not None and not np.isfinite(self.fitness)
        fitness_collapse=self.best_fitness and self.best_fitness>0 and self.fitness==0
        corrupted=loss_nan and (fitness_nan or fitness_collapse)
        reason='Loss NaN/Inf' if loss_nan else 'Fitness NaN/Inf' if fitness_nan else 'Fitness collapse'
        if not corrupted: return False
        if epoch==self.start_epoch or not self.last.exists():
            warnings.warn(f'{reason} detected but cannot recover from last.pt... since this is the first epoch or checkpoint file does not exist')
            return False # Cannot recover on first epoch and we let training continue
        self.nan_recovery_attempts+=1
        if self.nan_recovery_attempts>3:
            raise RuntimeError(f'Training failed: NaN persisted for {self.nan_recovery_attempts} epochs')
        warnings.warn(f'{reason} detected (attempted {self.nan_recovery_attempts}/3), recovering from last.pt...')
        self._model_train() # set model to train mode before loading checkpoints to avoid inference tensor errors
        ckpt=load_checkpoint(self.last)
        model_state=ckpt['model'].float().state_dict()
        if not all(torch.isfinite(v).all() for v in model_state.values() if isinstance(v, torch.Tensor)):
            raise RuntimeError(f'Checkpoint {self.last} is corrupted with NaN/Inf weights')
        unwrap_model(self.model).load_state_dict(model_state)
        self._load_checkpoint_state(ckpt)
        del ckpt, model_state
        self.scheduler.last_epoch=epoch-1
        return True

    def save_metrics(self, metrics):
        """Save training metrics to a CSV file"""
        keys, vals=list(metrics.keys()), list(metrics.values())
        n=len(metrics)+2 # number of columns
        t=time.time()-self.train_time_start
        self.csv.parent.mkdir(parents=True, exist_ok=True) # ensure parent directory exists
        s="" if self.csv.exists() else ('%s,' *n %('epoch', 'time', *keys)).rstrip(',')+"\n"
        with open(self.csv, 'a', encoding='utf-8') as f:
            f.write(s+('%.6g,'*n % (self.epoch+1, t, *vals)).rstrip(',')+"\n")

    def save_model(self):
        """Save model training checkpoints with additional metadata"""
        state_dict={"epoch":self.epoch, 
                    "best_fitness":self.best_fitness, 
                    "model":self.model.state_dict(),
                   "optimizer":self.optimizer.state_dict()}
        # Save checkpoints
        self.wdir.mkdir(parents=True, exist_ok=True) # ensure weights directory exists
        torch.save(state_dict, self.last)
        if self.best_fitness==self.fitness: torch.save(state_dict, self.best)
        if (self.save_period>0) and (self.epoch%self.save_period==0):
            torch.save(state_dict, self.wdir/f'epoch{self.epoch}.pt') # save epoch, e.g., 'epoch3.pt'

    def plot_metrics(self):
        """Plot metrics from a CSV file"""
        plot_results(file=self.csv) # save results.png
        
    def train(self):
        """Train the model"""
        self._do_train()

    def _do_train(self):
        """Train the model with the specified world size"""
        self._setup_train()
        
        nb=len(self.train_loader) # number of batches
        ## self.args.warmup_epochs is the number of epochs spent warming up the optimizer
        ## we multiply self.args.warmup_epochs by nb since training updates happen per batch (iteration)
        ## warm up over nw optimization steps/iterations, i.e., nw is the number of time we will perform learning-rate adjustment for warmup
        # nw=max(round(trainer.args.warmup_epochs*nb), 100) if trainer.args.warmup_epochs>0 else -1 
        last_opt_step=-1
        self.epoch_time=None
        self.epoch_time_start=time.time()
        self.train_time_start=time.time()
        print(f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
              f"Using {self.train_loader.num_workers *(self.world_size or 1)} dataloader workers\n"
              f"Logging results to {self.save_dir}\n"
              f"Starting training for "+(f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
        )
        if self.args.close_mosaic:
            base_idx=(self.epochs-self.args.close_mosaic)*nb # batch iterations in which close_mosaic start
            self.plot_idx.extend([base_idx, base_idx+1, base_idx+2])
        epoch=self.start_epoch
        self.optimizer.zero_grad() # zero any resumed gradients to ensure stability on training start
        while True:
            self.epoch=epoch
            with warnings.catch_warnings():
                warnings.simplefilter('ignore') # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()
            self._model_train()
            # Update dataloader attributes 
            if epoch==(self.epochs - self.args.close_mosaic): 
                # TODO: test to see whether we get non-mosaic images after we close the mosaic
                # set args.plot to true to save a few training images
                self._close_dataloader_mosaic()
                #trainer.train_loader.reset()
            last_opt_step=self.train_each_epoch(epoch, last_opt_step)
            self.lr={f'lr/pg{ir}':x['lr'] for ir, x in enumerate(self.optimizer.param_groups)}
            # self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])
        
            # Validation
            final_epoch=epoch+1>=self.epochs
            if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                self._clear_memory(threshold=0.5) # prevent VRAM spike
                self.metrics, self.fitness=self.validate()
        
            #NaN recovery
            if self._handle_nan_recovery(epoch): continue
        
            self.nan_recovery_attempts=0
            self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
            self.stop|=self.stopper(epoch+1, self.fitness) or final_epoch
            if self.args.time: self.stop|=(time.time()-self.train_time_start)>(self.args.time*3600)
                
            # Save model
            if self.args.save or final_epoch: self.save_model()
        
            # Scheduler
            t=time.time()
            self.epoch_time=t-self.epoch_time_start
            self.epoch_time_start=t
            # if self.args.time:
            #     mean_epoch_time=(t-self.train_time_start)/(epoch-self.start_epoch+1)
            #     self.epochs=self.args.epochs=math.ceil(self.args.time*3600/mean_epoch_time)
            #     self._setup_scheduler() # setup scheduler again based on total epochs
            #     self.scheduler.last_epoch=self.epoch # do not move
            #     self.stop|=epoch>=self.epochs # stop if training exceeds epochs
            self._clear_memory(0.5) # clear if memory utilization > 50%
        
            # Early stopping
            if self.stop: break
            epoch+=1
        
        # done training loop
        seconds=time.time()-self.train_time_start
        print(f"\n{epoch-self.start_epoch+1} epochs completed in {seconds/3600:.3f} hours")
        # do final val with best.pt
        if self.args.plots: self.plot_metrics()
        self._clear_memory()
        unset_deterministic()
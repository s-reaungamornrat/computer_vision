from __future__ import annotations

from typing import Any
from pathlib import Path
from argparse import Namespace

import yaml
import time
import math
import copy
import warnings

import torch
import torch.nn as nn

from .detector import DetectionModel
from .validator import DetectionValidator
from computer_vision.yolov11.utils.check import check_imgsz
from computer_vision.yolov11.data.dataset import YOLODataset
from computer_vision.yolov11.utils.torch_utils import init_seeds, ModelEMA, one_cycle, EarlyStopping

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
        self.cvs=self.save_dir/"result.csv"
        if self.cvs.exists() and not self.args.resume: self.csv.unlink(missing_ok=True)
        self.plot_idx=[0,1,2]
        self.nan_recovery_attempts=0 
        print('self.args.resume ', self.args.resume)

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
        if checkpoint.get('optimizer') is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint.get('scheduler') is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        if checkpoint.get('scaler') is not None:
            self.scaler.load_state_dict(checkpoint['scaler'])
        if self.ema and checkpoint.get('ema') is not None:
            # validation with EMA creates inference tensors that cannot be updated...
            self.ema=ModelEMA(trainer.model) 
            self.ema.ema.load_state_dict(checkpoint['ema'].float().state_dict())
            self.ema.updates=checkpoint['updates']
        self.best_fitness=checkpoint.get('best_fitness', 0.)
        
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
            checkpoint=torch.load(self.last, map_location=self.device, weights_only=True)
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
        self.validator=DetectionValidator(hyperparam=self.cfg, data_cfg=self.data, dataloader=self.val_loader, save_dir=self.save_dir, 
                                          args=copy.deepcopy(self.args))
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
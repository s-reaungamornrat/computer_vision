from __future__ import annotations

import gc
import math
import os
import yaml
import warnings
from copy import copy, deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim

from computer_vision.yolov11_pose.utils import DEFAULT_CFG_DICT
from computer_vision.yolov11_pose.cfg import get_cfg
from computer_vision.yolov11_pose.utils.files import get_latest_run
from computer_vision.yolov11_pose.utils.torch_utils import init_seeds, unwrap_model
from computer_vision.yolov11_pose.data.utils import check_det_dataset
from computer_vision.yolov11_pose.nn.tasks import load_checkpoint
from computer_vision.yolov11_pose.utils.checks import check_imgsz
from computer_vision.yolov11_pose.data.build import build_yolo_dataset, build_dataloader
from computer_vision.yolov11_pose.utils.plotting import plot_labels

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
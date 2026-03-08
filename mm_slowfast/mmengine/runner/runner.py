from __future__ import annotations

import os
import time
import copy
from functools import partial
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import distributed as torch_dist

import numpy as np

from mmengine import Config
from .utils import set_random_seed
from .loops import EpochBasedTrainingLoop, ValLoop

def is_distributed()->bool:
    """Return True if distributed environment has been initialized"""
    return torch_dist.is_available() and torch_dist.is_initialized()
    
class Runner:
    """A training helper for Pytorch

    Runner object can be built from config. We usually use the same config to launch training, testing and validation tasks. However, only
    some of these components are necessary at the same time, e.g., testing a model does not need training or validation related components.

    To avoid repeatedly modifying config, the construction of `Runner` adopts lazy initialization to only initialize components when they are
    going to be used. Therefore, the model is always initialized at the beginning, and training, validation, and testing related components are
    only initialized when calling `runner.train()`, `runner.val()`, and `runner.test()`, respectively.

    Reference: https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/runner.py#L77
    """
    cfg:Config
    _train_loop:Optional[dict]
    _val_loop:Optional[dict]
    _test_loop:Optional[dict]
    def __init__(self, model:Union[nn.Module, dict], work_dir:str, cfg:Config=None):
        
        self._work_dir=work_dir
        if not os.path.isdir(self._work_dir): os.makedirs(self._work_dir)

        self.cfg=copy.deepcopy(cfg)
        self._launcher='none'
        self._distributed=False
        self.model=model
        
        # originally calling self.setup_env()
        self._timestamp=time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
        self._randomness_cfg=cfg.randomness
        print(f"{self._randomness_cfg=}")
        self.set_randomness(**self._randomness_cfg)
        print(f"{self._seed=}")

        self._experiment_name="{}_{}".format(os.path.splitext(os.path.basename(cfg.filename))[0], self._timestamp)
        self._log_dir=os.path.join(self._work_dir, self._timestamp )
        if not os.path.isdir(self._log_dir): os.makedirs(self._log_dir)

        self._load_from=cfg.load_from
        self._resume=cfg.resume
        self._has_loaded=False # flag to mark whether checkpoint has been loaded or resumed

        # get model name from the model class
        if hasattr(self.model, 'module'): self._model_name=self.model.module.__class__.__name__
        else: self._model_name=self.model.__class__.__name__

        # dump `cfg` to `work_dir`
        #self.dump_config()
        
    def dump_config(self)->None:
        """Dump config to `work_dir`"""
        if self.cfg.filename is not None: filename=os.path.basename(self.cfg.filename)
        else: filename=f"{self.timestamp}.py"
        self.cfg.dump(os.path.join(self.work_dir, filename))
        
        
    def set_randomness(self, seed, diff_rank_seed:bool=False, deterministic:bool=False)->None:
        """Set random seed to guarantee reproducible results
        Args:
            seed (int): A number to set random modules
            diff_rank_seed (bool): Whether to use different seeds according to global rank. Default to False
            deterministic (bool): Whether to set determinic option for cudnn backend. i.e.,
                set `torch.backends.cudnn.deterministic` to True and `torch.backends.cudnn.benchmark` to False.
                Defaults to False. See https://pytorch.org/docs/stable/notes/randomness.html for more detail.
        """
        self._deterministic=deterministic
        self._seed=set_random_seed(seed=seed, deterministic=deterministic)

    def resume(self, filename:str, map_location=torch.device('cpu'))->None:
        """Resume model from checkpoint
        Args:
            filename (str): Checkpoint file
            map_location (torch.device): Device
        """
        checkpoint=torch.load(filename, map_location=map_location, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model'])
        
        self._train_loop._epoch=checkpoint['epoch']
        if 'iter' in checkpoint: self._train_loop._iter=checkpoint['iter']
        if 'optimizer' in checkpoint: 
            self.optim_wrapper.load_state_dict(checkpoint['optimizer'])

        if 'param_schedulers' in checkpoint:
            if isinstance(self.param_schedulers, dict):
                for name, schedulers in self.param_schedulers.items():
                    for scheduler, ckpt_scheduler in zip(schedulers, checkpoint['param_schedulers'][name]): 
                        scheduler.load_state_dict(ckpt_scheduler)
            else:
                for scheduler, ckpt_scheduler in zip(self.param_schedulers, checkpoint['param_schedulers']):
                    scheduler.load_state_dict(ckpt_scheduler)
        self._has_loaded=True
from __future__ import annotations
from typing import Optional, Union

import time
import copy
import bisect

import torch
from torch.utils.data import DataLoader

from computer_vision.slowfast.mmengine.dataset.utils import pseudo_collate
from computer_vision.slowfast.mmengine.runner.utils import set_random_seed, calc_dynamic_intervals
from computer_vision.slowfast.mmengine.logging.history_buffer import HistoryBuffer
from computer_vision.slowfast.mmengine.runner.amp import autocast
from computer_vision.slowfast.mmengine.structures.base_data_element import BaseDataElement
from computer_vision.slowfast.mmengine.utils.misc import is_list_of

# https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py
def _parse_losses(losses:dict[str, HistoryBuffer], stage:str)->dict[str, float]:
    """Parses the raw losses of the network
    Args:
        losses (dict): Raw losses of the network
        stage (str): The stage of loss, e.g., 'val' or 'test'
    Returns:
        (dict[str, float]): The key is the loss name, and the value is the average loss
    """
    all_loss=0
    loss_dict:dict[str, float]=dict()

    for loss_name, loss_value in losses.items():
        avg_loss=loss_value.mean()
        loss_dict[loss_name]=avg_loss
        if 'loss' in loss_name: all_loss+=avg_loss

    loss_dict[f'{stage}_loss']=all_loss
    return loss_dict

def _update_losses(outputs:list, losses:dict)->tuple[list, dict]:
    """Update and record the losses of the network
    Args:
        outputs (list): The outputs of the network
        losses (dict): The losses of the network
    Returns:
        (list): The updated outputs of the network
        (dict): The updated losses of the network
    """
    if isinstance(outputs[-1], BaseDataElement) and outputs[-1].keys==['loss']:
        loss=outputs[-1].loss
        outputs=outputs[:-1] # ignoe the last item
    else: loss=dict()

    for loss_name, loss_value in loss.items():
        if loss_name not in losses: losses[loss_name]=HistoryBuffer
        if isinstance(loss_value, torch.Tensor): losses[loss_name].update(loss_value.item())
        elif is_list_of(loss_value, torch.Tensor):
            for loss_value_i in loss_value: losses[loss_name].update(loss_value_i.item())
    return outputs, losses

class ValLoop:
    """Loop for validation
    Args:
        runner (Runner): A reference of runner
        dataloader (DataLoader): A dataloader object 
        evaluator (Evaluator): An evaluator for computing metric
        fp16 (bool): Whether to enable fp16. Default to False
    Reference: https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py
    """
    def __init__(self, runner, dataloader:DataLoader, evaluator:Evaluator, fp16:bool=False)->None:
        self.dataloader=dataloader
        self.evaluator=evaluator
        self.fp16=fp16
        self.val_loss:dict[str, HistoryBuffer]=dict()

        if hasattr(self.dataloader.dataset, 'metainfo'): self.evaluator.dataset_meta=self.dataloader.dataset.metainfo
        else: warnings.warn(f"Dataset {self.dataloader.dataset.__class__.__name__} has no metainfo `dataset_meta` in evaluator,"
                            " metric and visualizer will be None")
        
    @property
    def runner(self): return self._runner

    def run(self)->dict:
        """Launch validation"""
        self.runner.model.eval()
        
        # clear val loss
        self.val_loss.clear()
        for idx, data_batch in enumerate(self.dataloader): self.run_iter(idx, data_batch)
        
        # compute metrics
        metrics=self.evaluator.evaluate(len(self.dataloader.dataset))

        if self.val_loss:
            loss_dict=_parse_losses(self.val_loss, 'val')
            metrics.update(loss_dict)
        return metrics

    @torch.no_grad()
    def run_iter(self, idx, data_batch:Sequence[dict]):
        """Iterate one mini-batch
        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader
        """
        # outputs should be sequence of BaseDataElement
        with autocast(enabled=self.fp16): outputs=self.runner.model.val_step(data_batch)
        outputs, self.val_loss=_update_losses(outputs, self.val_loss)
        self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        

class EpochBasedTrainingLoop:
    """Loop for epoch-based training
    Args:
        runner (Runner): A reference to runner
        dataloader (DataLoader): An iterator to generate a batch for each iteration
        max_epochs (int): Total training epochs
        val_begin (int): The epoch that validation begins. Default to 1
        val_interval (int): Validation interval. Default to 1
        dynamic_intervals (list[tuple[int, int]], optional): The first element in the tuple is a milestone and the second element is 
            an interval. The interval is used after the corresponding milestone. Default to None.
    Reference: https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py
    """
    def __init__(self, runner, dataloader:DataLoader, max_epochs:int, val_begin:int=1, val_interval:int=1, 
                 dynamic_intervals:Optional[list[tuple[int, int]]]=None)->None:

        self._runner=runner
        self.dataloader=dataloader
        
        self._max_epochs=int(max_epochs)
        self._max_iters=self._max_epochs*len(self.dataloader)
        self._epoch=0
        self._iter=0
        self.val_begin=val_begin
        self.val_interval=val_interval

        # This attribute will be updated by `EarlyStoppingHook` when it is enabled
        self.stop_training=False

        self.dynamic_milestones, self.dynamic_intervals=calc_dynamic_intervals(self.val_interval, dynamic_intervals)

    @property
    def runner(self): return self._runner
        
    @property
    def max_epochs(self)->int:
        """Total epochs to train a model"""
        return self._max_epochs

    @property
    def max_iters(self)->int:
        """Total iterations to train a model"""
        return self._max_iters

    @property
    def epoch(self)->int:
        """Current epoch"""
        return self._epoch

    @property
    def iter(self)->int:
        """Current iteration"""
        return self._iter

    def _decide_current_val_interval(self)->None:
        """Dynamically modify the `val_interval` """
        
        step=bisect.bisect(self.dynamic_milestones, (self.epoch+1))
        self.val_interval=self.dynamic_intervals[step-1]
        
    def run_iter(self, idx:int, data_batch:Sequence[dict])->None:
        """Iterate one mini-batch
        Args:
            idx (int): Batch index
            data_batch (Sequence[dict]): Batch of data from dataloader
        """
        # Enable gradient accumulation mode and avoid unnecessary gradient synchronization during gradient accumulation process. 
        # Outputs should be a dict of loss
        outputs = self.runner.model.train_step(data_batch, optim_wrapper=self.runner.optim_wrapper)
        # schedule step???
        self._iter+=1
        
    def run_epoch(self)->None:
        """Iterate one epoch"""
        
        self.epoch_start_time=time.time()
        
        if hasattr(self.dataloader, 'sampler') and hasattr(self.dataloader.sampler, 'set_epoch'):
            self.dataloader.sampler.set_epoch(self.runner.epoch)

        self.runner.model.train()
        for idx, data_batch in enumerate(self.dataloader): 
            self.run_iter(idx, data_batch)
        
        # schedule step???
        # save checkpoint!!!
        self._epoch+=1
        
    def run(self)->torch.nn.Module:
        """Launch training"""
        while self._epoch < self._max_epochs and not self.stop_training:
            self.run_epoch()
            self._decide_current_val_interval()
            if (self.runner.val_loop is not None and self._epoch>=self.val_begin and 
                (self._epoch%self.val_interval==0 or self._epoch==self._max_epochs)):
                self.runner.val_loop.run()
        return self.runner.model
            
            
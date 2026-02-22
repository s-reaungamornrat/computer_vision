from __future__ import annotations
from typing import Optional
from contextlib import contextmanager

import torch
import torch.nn as nn
from torch.optim import Optimizer

from .base import BaseOptimWrapper
from computer_vision.slowfast.mmengine.utils.dl_utils.misc import has_batch_norm

class OptimWrapper(BaseOptimWrapper):
    """Optimizer wrapper provides a common interface for updating parameters.

    Optimizer wrapper provides a unified interface for single precision training and automatic mixed precision training with different hardware.
    OptimWrapper encapsulates optimizer to provide simplified interfaces for commonly used training techniques such as gradient accumulative and grad
    clips. OptimWrapper implements the basic logic of gradient accumulation and gradient clipping based on `torch.optim.Optimizer`. The subclasses only
    need to override some methods to implement the mixed precision training. See more in `AmpOptimWrapper`

    Args:
        optimizer (Optimizer): Optimizer used to update model parameters.
        accumulative_counts (int): The number of iterations to accumulate gradients. The parameters will be updated per `accumulative_counts`
        clip_grad (dict, optional): If `clip_grad` is not None, it will be argument of `torch.nn.utils.clip_grad_norm_` or 
            `torch.nn.utils.clip_grad_value_`. `clip_grad` should be a dict with keys.
            If the key `type` is not set or set to 'norm',
                - 'max_norm' (float | int): Maximum norm of gradients
                - 'norm_type' (float | int): Type of the used p-norm. Can be 'inf' for infinity norm
                - 'error_if_nonfinite' (bool): If True, an error is throw if the total norm of the gradients is `nan`, `inf`, or `-inf`. 
                    Default to False
            If the key `type` is set to 'value', 
                - 'clip_value' (float|int): Maximum allowed value of the gradient. The gradients are clipped in the range (-clip_value, clip_value)
    Note:
        If `accumulative_counts` is larger than 1, perform `update_params` under the context of `optim_context` could avoid unnecessary gradient
        synchronization
    Note:
        If you use `IterBasedRunner` and enable gradient accumulation, the original `max_iters` should be multipled by `accumulative_counts`
    Note:
        The subclass should make sure that once `update_params` is called, `inner_count+=1` is automatically performed.
    Reference: https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/optimizer/optimizer_wrapper.py#L17
    """
    def __init__(self,optimizer:Optimizer, accumulative_counts:int=1, clip_grad:Optional[dict]=None):
        assert accumulative_counts>0, f"accumulative_counts must be greater than or equal to 1, but got {accumulative_counts}"
        self._accumulative_counts=accumulative_counts
        self.optimizer=optimizer

        if clip_grad is not None:
            # clip_grad_kwargs should not be non-empty dict
            assert isinstance(clip_grad, dict) and clip_grad, ("If `clip_grad` is not None, it shoyld be a dict of arguments for "
                                                               "`torch.nn.utils.clip_grad_norm_` or `torch.nn.utils.clip_grad_value_ ")
            clip_type=clip_grad.pop('type', 'norm')
            if clip_type=='norm':
                self.clip_func=torch.nn.utils.clip_grad_norm_
                self.grad_name='grad_norm'
            elif clip_type=='value':
                self.clip_func=torch.nn.utils.clip_grad_value_
                self.grad_name='grad_value'
            else: raise ValueError(f'Type of clip_grad should be "norm" or "value", but got {clip_type}')
            assert clip_grad, ("`clip_grad` should contain other arguments besides `type`. The arguments should match with "
                               "`torch.nn.utils.clip_grad_norm_ or torch.nn.utils.clip_grad_value_")
        self.clip_grad_kwargs=clip_grad
        self._inner_count=0
        # `_max_counts` means the total number of parameter updates. It ensures that the gradient of the last few iterations will not be lost when the
        # `_max_counts` is not divisible by `accumulative_counts`
        self._max_counts=-1
        # The `_remainder_counts' is used for calculating loss factor at the last few iterations. If `_max_counts` has not been initialized, the loss
        # factor will always be the same as `_accumulative_counts`
        self._remainder_counts=-1

        # The following code is used to initialize `base_param_settings`. `base_param_settings` is used to store parameters that are not updated by the
        # optimizer. 
        # The `base_param_settings` used for tracking the base learning in the optimizer. If the optimizer has multiple parameter groups, this params 
        # will not be scaled by the loss factor
        if len(optimizer.param_groups)>1:
            self.base_param_settings={'params':torch.tensor([0.,], dtype=torch.float)}
            self.base_param_settings.update(**self.optimizer.defaults)
        else: self.base_param_settings=None 

    def scale_loss(self, loss:torch.Tensor)->torch.Tensor:
        """Get scaled loss according to `_accumulative_counts`, `inner_counts` and `max_counts`
        Args:
            loss (torch.Tensor): Original loss value
        Returns:
            loss (torch.Tensor): Scaled loss
        """
        if self._accumulative_counts==1:
            # update parameters without gradient accumulation. The gradient should not be rescaled and `loss_factor=1`
            loss_factor=1
        elif self._max_counts==-1: loss_factor=self._accumulative_counts
        else:
            # if `self._accumulative_counts>1`, the gradient needs to be rescaled and accumulated. In most cases, `loss_factor` equals to 
            # `self._accumulative_counts`. However, `self._max_counts` may not be divisible by `self._accumulative_counts`, so the `loss_scale`
            # for the last few iterations needs to be recaculated
            if self._inner_count<self._max_counts-self._remainder_counts: loss_factor=self._accumulative_counts
            else: loss_factor=self._remainder_counts
            assert loss_factor>0, ("loss_factor should be larger than zero! This error could happened when initialize_iter_status called with "
                                   "an error `init_counts` or `max_counts`")
        loss=loss/loss_factor
        return loss

    def backward(self,loss:torch.Tensor, **kwargs)->None:
        """Perform gradient back propagation

        Provide unified `backward` interface compatible with automatic mixed precision training. Subclass can overload this method to implement the
        required logic. For example, `torch.cuda.amp` require some extra operation on GradScaler during backward process

        Note:
            If subclasses inherit from `OptiWrapper` override `backward`, `_inner_counts+=1` must be implemented.
        Args:
            loss (torch.Tensor): Loss of current iteration
            kwargs (dict): Keyward arguments passed to `torch.Tensor.backward`
        """
        loss.backward(**kwargs)
        self._inner_count+=1

    def should_update(self)->bool:
        """Decide whether the parameters should be updated at the current iteration

        Called by `update_params` and check whether the optimizer wrapper should update parameters at current iteration

        Returns:
            (bool): Whether to update parameters
        """
        return (self._inner_count%self._accumulative_counts == 0 or self._inner_count==self._max_counts)

    def step(self, **kwargs)->None:
        """A wrapper of optimizer.step
        Args:
            kwargs (dict): Keyword arguments of the `step` function
        """
        if self.clip_grad_kwargs: self._clip_grad()
        self.optimizer.step(**kwargs)

    def _clip_grad(self):
        """Clip the gradients of parameters
        
        Returns:
            (torch.Tensor|None): Total norm of the parameter gradients (viewed as a single vector) if torch.nn.utils.clip_grad_norm_ is used;
                otherwise, return None
         """
        params:list[torch.Tensor]=[]
        for param_group in self.optimizer.param_groups: params.extend(param_group['params'])

        params=list(filter(lambda p: p.requires_grad and p.grad is not None, params))
        grad=None
        if len(params):
            grad=self.clip_func(params, **self.clip_grad_kwargs)
            # torch.nn.utils.clip_grad_value_ will return None
        return grad

    def zero_grad(self, **kwargs)->None:
        """A wrapper of `optimizer.zero_grad`
        Args:
            kwargs (dict): Keyword arguments to optimizer.zero_grad
        """
        self.optimizer.zero_grad(**kwargs)
        
    def update_params(self, loss:torch.Tensor, step_kwargs:Optional[dict]=None, zero_kwargs:Optional[dict]=None)->None:
        """Update parameters
        Args:
            loss (torch.Tensor): A tensor for back propagation
            step_kwargs (dict, optional): Arguments for optimizer.step. Default to None
            zero_kwargs (dict, optional): Argument to optimizer.zero_grad. Default to None
        """
        if step_kwargs is None: step_kwargs={}
        if zero_kwargs is None: zero_kwargs={}
        loss=self.scale_loss(loss)
        self.backward(loss)
        # Update parameters only if `self._inner_count` is divisible by `self._accumulative_counts` or `self._inner_count` equal to `self._max_counts`
        if self.should_update():
            self.step(**step_kwargs)
            self.zero_grad(**zero_kwargs)
            
    def should_sync(self)->bool:
        """Decide whether the automatic gradient synchronization should be allowed at the current iteration

        It takes effect when gradient accumulation is used to skip synchronization at the iterations where the parameter is not updated.

        Since `should_sync` is called by `optim_context` and it is called before `backward` which means `self._inner_count+=1` has nt happened yet.
        Therefore, `self._inner_count+=1` should be performed manually here

        Returns:
            (bool): Whethere to block automatic gradient synchronization
        """
        return ((self._inner_count+1)%self._accumulative_counts ==0) or ((self._inner_count+1)==self._max_counts)

    @contextmanager
    def optim_context(self, model:nn.Module):
        """Context for gradient accumulation and automatic mix precision training

        If subclasses need to enable the context for mix precision training, e.g., `AmpOptimWrapper`, the corresponding context should be enabled in 
        `optim_context`. Since `OptimWrapper` uses default fp32 training, `optim_context` will only enable the context for blocking the unnecessary
        gradient synchronization during gradient accumulation

        If model is an instance with `no_sync` method (which means blocking the gradient syncrhonization) and `self._accumulative_counts!=1`. The model
        will not automatically synchronize gradients if `cur_iter` is divisible by `self._accumulative_counts`. Otherwise, this method will enable an 
        empty context.

        Args:
            model (nn.Module): Trainable model
        """
        # During gradient accumulation process, the gradient synchronization should only happen before updating parameters
        if not self.should_sync() and hasattr(model, 'no_sync'):
            with model.no_sync(): yield
        else: yield

    def initialize_count_status(self, model:nn.Module, init_counts:int, max_counts:int)->None:
        """Initialize gradient accumulation related attributes
        
        `OptimWrapper` can be used without calling `initialize_count_status`. However, consider the case of `len(dataloader)==10` and the 
        `accumulative_counts==3`. Since 10 is not divisible by 3, the last iteratioon will not trigger `optimizer.step()`, resulting in one less
        parameter update.

        Args:
            model (nn.Module): Trainable model
            init_counts (int): The initial value of inner count
            max_counts (int): The maximum value of inner count
        """
        self._inner_count=init_counts
        self._max_counts=max_counts
        if self._inner_count%self._accumulative_counts!=0:
            print("Resumed iteration number is not divisible by `_accumulative_counts` in `GradientCumulativeOptimizerHook`, which means the "
                 "gradient of some iterations is lost and the result may be influened slightly")
        if has_batch_norm(model) and self._accumulative_counts>1:
            print("Gradient accumulative may slightly decrease performance because the model has BatchNorm layer")
        self._remainder_counts=self._max_counts%self._accumulative_counts

    @property
    def inner_count(self):
        """Get the number of updating parameters of optimizer wrapper"""
        return self._inner_count

    def __repr__(self):
        wrapper_info=(f"Type: {type(self).__name__}\n"
                      f"_accumulative_counts: {self._accumulative_counts}\n"
                      "optimizer:\n")
        optimizer_str=repr(self.optimizer)+"\n"
        return wrapper_info+optimizer_str
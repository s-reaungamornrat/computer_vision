from __future__ import annotations
from abc import ABCMeta, abstractmethod
from typing import Dict,List

import torch

class BaseOptimWrapper(metaclass=ABCMeta):
    """
    Reference: https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/optimizer/base.py#L32
    """
    def __init__(self, optimizer):
        self.optimizer=optimizer
        # Following code is used to initialized `base_param_settings`. The `base_param_settings` is used to store the parameters that are not updated
        # by the optimizer
        # The `base_param_settings` used for trainig the base learning in the optimizer. If the optimizer has multiple parameter groups, this params will
        # not be scaled by the loss factor. The example of settings stored in `base_param_settings` for SGD include 
        # {'lr': 0.0003, 'momentum': 0, 'dampening': 0, 'weight_decay': 0, 'nesterov': False, 'maximize': False, 'foreach': None, 'differentiable': False}
        if len(optimizer.param_groups)>1:
            self.base_param_settings={'params':torch.tensor([0.], dtype=torch.float)}
            self.base_param_settings.update(**self.optimizer.defaults) # deepcopy initial settings of optimizer
        else: self.base_param_settings=None

    @abstractmethod
    def update_params(self, *args, **kwargs):
        """Update parameters in `optimizer`"""

    @abstractmethod
    def backward(self,loss:torch.Tensor, **kwargs)->None:
        """Perform gradient back propagation"""

    @abstractmethod
    def zero_grad(self, **kwargs)->None:
        """A wrapper of `optimizer.zero_grad`"""

    @abstractmethod
    def step(self,**kwargs):
        """Call the step method of optimizer"""

    def state_dict(self)->dict:
        """A wrapper of optimizer.state_dict"""
        state_dict=self.optimizer.state_dict()
        if self.base_param_settings is not None: state_dict['base_param_settings']=self.base_param_settings
        return state_dict

    def load_state_dict(self,state_dict:dict)->None:
        """A wrapper for optimizer.load_state_dict. 
        
        Provide unified `load_state_dict` interface compatible with automatic mixed precision training. Subclass can overload this method to implememt
        the required logic. For example, the state dict of GradScaler should be loaded when training woth `torch.cuda.amp`.

        Args:
            state_dict (dict): The state dict of `optimizer`
        """
        base_param_settings=state_dict.pop('base_param_settings', None)

        if base_param_settings is not None: self.base_param_settings=base_param_settings

        # load state_dict
        self.optimizer.load_state_dict(state_dict)

    @property
    def param_groups(self)->list[dict]:
        """A wrapper of `optimizer.param_groups`
        Make optimizer wrapper compatible with `_ParamScheduler`
        Returns:
            (dict): `param_groups` of `optimizer`
        """
        if self.base_param_settings is not None: return self.optimizer.param_groups+[self.base_param_settings]
        return self.optimizer.param_groups

    @property
    def defaults(self)->dict:
        """A wrapper of `optimizer.defaults`

        Returns:
            (dict): the `param_groups` of `optimizer`
        """
        return self.optimizer.defaults

    def get_lr(self):
        """Get the learning rate of the optimizer
        
        Provide unified interface to get learning rate of the optimizer

        Returns:
            (dict[str, list[float]]): param_groups learning rate of the optimizer
        """
        res=dict()
        if self.base_param_settings is not None: res['base_lr']=[self.base_param_settings['lr']]
            
        res['lr']=[groups['lr'] for group in self.optimizer.param_groups]
        return res

    def get_momentum(self)->dict[str, list[float]]:
        """Get momentum of the optimizer
        
        Provide unified interface to get momentum of optimizer

        Returns:
            (dict[str, list[float]]): Momentum of optimizer
        """
        momentum=[]
        for group in self.optimizer.param_groups:
            # Get momentum of SGD
            if 'momentum' in group.keys(): momentum.append(group['momentum'])
            elif 'betas' in group.keys(): momentum.append(group['betas'][0]) # get momentum of Adam
            else: momentum.append(0)
        return dict(momentum=momentum)
        
        
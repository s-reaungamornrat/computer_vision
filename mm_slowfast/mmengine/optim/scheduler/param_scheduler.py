from __future__ import annotations

import math
import warnings
import weakref
from collections import Counter
from functools import wraps
from typing import Callable, List, Optional, Sequence, Union

from torch.optim import Optimizer

from computer_vision.slowfast.mmengine.optim.optimizer.base import BaseOptimWrapper

INF=int(1e9)

OptimizerType=Union[BaseOptimWrapper, Optimizer]

class _ParamScheduler:
    """Base class for parameter schedulers

    It should be inherited by all schedulers that schedule parameters in the optimizer's `param_groups`. All subclasses should overwrite the `get_value()` 
    according to their own schedule strategy. The implementation is motivated by https://github.com/pytorch/pytorch/blob/master/torch/optim/lr_scheduler.py.

    Args:
        optimizer (BaseOptimWrapper | Optimizer): Wrapped optimizer
        param_name (str): Name of the parameter to be adjusted, such as `lr` and `momentum`
        begin (int): Step at which to start updating the parameters. Default to 0
        end (int): Step at which to stop updating the parameters. Default to INF
        last_step (int): The index of last step. Used for resuming without state dict. Default value `-1` means the `step` function is never called before.
            Default to -1.
        by_epoch (bool): Whether to scheduled parameters are updated by epoch. Default to True
        verbose (bool): Whether to print the value for each update. Default to False.
    """
    def __init__(self, optimizer:OptimizerType, param_name:str='lr', begin:int=0, end:int=INF, last_step:int=-1, by_epoch:bool=True, verbose:bool=False):
        
        assert isinstance(optimizer, (Optimizer, BaseOptimWrapper)), f"`optimizer` should be an Optimizer, but got {type(optimizer).__name__}"
        self.optimizer=optimizer
        self.param_name=param_name
        if end<=begin: raise ValueError(f"end should be larger than begin, but got begin={begin} and end={end}")
        self.begin=begin
        self.end=end
        self.by_epoch=by_epoch
        assert isinstance(last_step, int) and last_step>=-1
        # Initialize valid step count and base values
        if last_step==-1:
            for group in optimizer.param_groups:
                # if the param is never scheduled, record the current value as the initial value
                group.setdefault(f"initial_{param_name}", group[param_name])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if f"initial_{param_name}" not in group:
                    raise KeyError(f"param 'initial_{param_name}' is not specified in param_group[{i}] when resuming an optimizer")
        self.base_values=[group[f'initial_{param_name}'] for group in optimizer.param_groups]
        self.last_step=last_step
        # Following Following https://github.com/pytorch/pytorch/issues/20124, we would like to ensure that `scheduler.step()`is called after `optimizer.step()`
        def with_counter(method:Callable):
            # the function injects a step counter ($\_global\_step$) directly into the optimizer's .step() method.
            if getattr(method, '_with_counter', False): # checks _with_counter to ensure it doesn't wrap the same optimizer twice
                #optimizer.step has already been replaced, return 
                return method
            # Keep a weak reference to the optimizer instance to prevent cyclic references
            instance_ref=weakref.ref(method.__self__) 
            # Get the unbound method for the same purpose
            func=method.__func__
            cls=instance_ref().__class__
            del method
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                instance=instance_ref()
                instance._global_step+=1
                wrapped=func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method, so attributes like `__func__` and `__self__` no longer exist
            wrapper._with_counter=True
            return wrapper
                
        # add counter to optimizer
        self.optimizer.step=with_counter(self.optimizer.step) # replaces the standard PyTorch method with this "tracked" version
        self.optimizer._global_step=-1
        self._global_step=-1
        self.verbose=verbose
        self.step()

    def state_dict(self)->dict:
        """Return the state of the scheduler as a dict

        It contains an entry for every variable in self.__dict__ which is not the optimizer
        Returns:
            (dict): Scheduler state
        """
        return {key:value for key, value in self.__dict__.items() if key!='optimizer'}

    def load_state_dict(self, state_dict:dict):
        """Load the scheduler state
        Args:
            state_dict (dict): Scheduler state. Should be an object returned from a acall to `state_dict`
        """
        self.__dict__.update(state_dict)

    def get_last_value(self):
        """Return the last computed value by current scheduler
        Returns:
            (list): A list of the last computed value of the optimizer's `param_group`
        """
        return self._last_value

    def _get_value(self):
        """Computed value using chainable form of the scheduler"""
        raise NotImplementedError

    def print_value(self, is_verbose:bool, group:int, value:float):
        """Display the current parameter value
        Args:
            is_verbose (bool): Whether to print the value
            group (int): The index of the current `param_group`
            value (float): The parameter value
        """
        if is_verbose:
            print(f"Adjusting parameter value of group {group} to {value:.4e}")

    def step(self):
        """Adjust the parameter value of each parameter group based on the specified schedule"""
        # Raise a warning if old pattern is detected. ttps://github.com/pytorch/pytorch/issues/20124
        if self._global_step==0:
            if not hasattr(self.optimizer.step, '_with_counter'):
                warnings.warn("Seem like `optimizer.step()` has been override after parameter value scheduler initialization. Please make sure to call "
                              "`optimizer.step()` before `scheduler.step()`. See more detials at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
            # Just check if there were two first scheduler.step() calls before optimizer.step()
            elif self.optimizer._global_step<0:
                warnings.warn("Detect call of `scheduler.step` before `optimizer.step`. In pytorch 1.1.0+. you should call them in the opposite oder: "
                             "`optimizer.step` before `scheduler.step`. Failure to do this will result in pytorch skipping the first value of the paramter "
                             "value schedule. See more details at "
                             "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
        self._global_step+=1
        # Compute parameter value per param group in the effective range
        if self.begin <=self._global_step < self.end:
            self.last_step+=1
            values=self._get_value()

            for i, data in enumerate(zip(self.optimizer.param_groups, values)):
                param_group, value=data
                param_group[self.param_name]=value
                self.print_value(self.verbose, i, value)

        self._last_value=[group[self.param_name] for group in self.optimizer.param_groups]


class LinearParamScheduler(_ParamScheduler):
    """Decays the parameter value of each parameter group by linearly changing small multiplicative factor until the number of epoch reaches a pre-defined
    milestone `end`.

    Notice that such decay can happen simultaneously with other changes to the parameter value from outside this scheduler
    Args:
        optimizer (Optimizer | BaseOptimWrapper): Optimizer or wrapped optimizer
        param_name (str): Name of the parameter to be adjusted, such as `lr` or `momentum`
        start_factor (float): The number we multiply parameter value in the first epoch. The multiplication factor changes towards end_factor in the following
            epochs. Default to 1./3.
        end_factor (float): The number we multiply parameter value at the end of linear changing process. Default to 1.
        begin (int): Step at which to start updating the parameters. Default to 0
        end (int): Steo at which to stop updating the parameters. Default to INF
        last_step (int): The index of last step. Used for resume without state dict. Defaults to -1
        by_epoch (bool): Whether the scheduled parameters are updated by epochs. Default to True
        verbose (bool): Whether to print the value for each update. Default to False
    Reference: https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/scheduler/param_scheduler.py#L708
    """
    def __init__(self,optimizer:Union[Optimizer,BaseOptimWrapper], param_name:str, start_factor:float=1./3., end_factor:float=1., begin:int=0, end:int=INF,
                 last_step:int=-1, by_epoch:bool=True, verbose:bool=False):
        
        assert 0.<start_factor<=1.0, f"Start multiplicative factor should be between 0 and 1, but got {start_factor}" 
        assert 0.<end_factor<=1.0, f"End multiplicative factor should be between 0 and 1, but got {end_factor}" 

        self.start_factor=start_factor
        self.end_factor=end_factor
        self.total_iters=end-begin-1
        super().__init__(optimizer, param_name=param_name, begin=begin, end=end, last_step=last_step, by_epoch=by_epoch, verbose=verbose)

    @classmethod
    def build_iter_from_epoch(cls, *args, begin=0, end=INF, by_epoch=True, epoch_length=None, **kwargs):
        """Build an iter-based instance of this scheduler from an epoch-based config"""
        assert by_epoch, "Only epoch-based kwargs whose `by_epoch=True` can be converted to iter-based"
        assert epoch_length is not None and epoch_length>0, f"`epoch_length` must be a positive integer, but got {epoch_length}"
        by_epoch=False
        begin=int(begin*epoch_length)
        if end!=INF: end=int(end*epoch_length)
        return cls(*args, begin=begin, end=end, by_epoch=by_epoch, **kwargs)

    def _get_value(self):
        """Compute value using chainable form of the scheduler"""
        if self.last_step==0:
            return [group[self.param_name]*self.start_factor for group in self.optimizer.param_groups]
        return [group[self.param_name]*(1.+(self.end_factor-self.start_factor)/
                                       (self.total_iters*self.start_factor+(self.last_step-1)*(self.end_factor-self.start_factor))
                                       )]

class CosineAnnealingParamScheduler(_ParamScheduler):
    """Set the parameter value of each parameter group using a cosine annealing schedule, where `eta_{max}` is set to the initial value and `T_{cur}` is
    the number of epochs since the last restart in Stochastic Gradient Descent with Warm Restarts (SGDR)
    .. math::
        \begin{aligned}
            \eta_t & = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1
            + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right),
            & T_{cur} \neq (2k+1)T_{max}; \\
            \eta_{t+1} & = \eta_{t} + \frac{1}{2}(\eta_{max} - \eta_{min})
            \left(1 - \cos\left(\frac{1}{T_{max}}\pi\right)\right),
            & T_{cur} = (2k+1)T_{max}.
        \end{aligned}

    Notice that because the schedule is defined recursively, the parameter value can be simultaneously modified outside this scheduler by other operators. 
    If the parameter value is set solely by this scheduler, the parameter value at each step becomes:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)

    It has been proposed in `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only implements the cosine annealing part of SGDR, 
    and not the restarts.
    Args:
        optimizer (Optimizer | BaseOptimWrapper): Optimizer or wrapped optimizer
        param_name (str): Name of parameter to be adjusted, such as `lr`, `momentum`
        T_max (int, optional): Maximum number of iterations. If not specified, use `end-begin`. Default to None
        eta_min (float, optional): Minimum parameter value, Default to None.
        begin (int): Step at which to start updating the parameters. Default to 0
        end (int): Step at which to stop updating the parameters. Default to INF
        last_step (int): The index of last step. Used for resume without state dict. Defaults to -1
        by_epoch (bool): Whether the scheduled parameters are updated by epochs. Default to True
        verbose (bool): Whether to print the value for each update. Default to False
        eta_min_ratio (float, optional): The ratio of the minimum parameter valye to the base parameter value. Euther `eta_min` or `eta_min_ratio` should
            be specified. Defaults to None
    Reference: https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/scheduler/param_scheduler.py#L567
    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """
    def __init__(self, optimizer:Union[Optimizer, BaseOptimWrapper], param_name:str, T_max:Optional[int]=None, eta_min:Optional[float]=None, begin:int=0,
                end:int=INF, last_step:int=-1, by_epoch:bool=True, verbose:bool=False, eta_min_ratio:Optional[float]=None):
        # to preserve backwards compatibility
        if eta_min is None and eta_min_ratio is None: eta_min=0.
        assert not all(x is None for x in [eta_min, eta_min_ratio]), "Either `eta_min` or `eta_min_ratio` must be specified"
        self.T_max=T_max
        self.eta_min=eta_min
        self.eta_min_ratio=eta_min_ratio
        super().__init__(optimizer, param_name=param_name, begin=begin, end=end, last_step=last_step, by_epoch=by_epoch, verbose=verbose)

    @classmethod
    def build_iter_from_epoch(cls, *args, T_max=None, begin=0, end=INF, by_epoch=True, epoch_length=None, **kwargs):
        """Build an iter-based instance of this scheduler from an epoch-based config"""
        assert by_epoch, "Only epoch-based kwargs whose `by_epoch=True` can be converted to iter-based"
        assert epoch_length is not None and epoch_length>0, f"`epoch_length` must be a positive integer, but got {epoch_length}"
        by_epoch=False
        if T_max is not None: T_max=T_max*epoch_length
        begin=int(begin*epoch_length)
        if end!=INF: end=int(end*epoch_length)
        return cls(*args, T_max=T_max, begin=begin, end=end, by_epoch=by_epoch, **kwargs)

    def _get_value(self)->list:
        """Compute value using chainable form of the scheduler"""
        def _get_eta_min(base_value):
            if self.eta_min_ratio is None: return self.eta_min
            return base_value*self.eta_min_ratio
        if self.last_step==0: return [group[self.param_name] for group in self.optimizer.param_groups]
        elif (self.last_step-1 - self.T_max) % (2*self.T_max)==0:
            return [
                group[self.param_name]+(base_value-_get_eta_min(base_value))*(1-math.cos(math.pi/self.T_max))/2 
                for base_value, group in zip(self.base_values, self.optimizer.param_groups)
            ]
        return [(1+math.cos(math.pi*self.last_step/self.T_max)) /
                (1+math.cos(math.pi*(self.last_step-1)/self.T_max)) *
                (group[self.param_name]-_get_eta_min(base_value))+_get_eta_min(base_value) for base_value, group in zip(self.base_values, 
                                                                                                                       self.optimizer.param_groups)]
        
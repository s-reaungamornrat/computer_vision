import os
import math
import copy
import random
import warnings

import torch
import torch.nn as nn
import numpy as np

class EarlyStopping:
    """
    Early stopping class that stops training when a specified number of epochs have passed without improvement
    """
    def __init__(self, patience=50):
        """
        Initialize early stopping object
        Args:
            patience (int, optional): Number of epochs to wait after fitness stops improving before stopping 
        """
        self.best_fitness=0. # i.e., mAP
        self.best_epoch=0.
        self.patience=patience or float('inf') # epochs to wait after fitness stops improving before stopping
        self.possible_stop=False # possible stop may occur next epoch

    def __call__(self, epoch, fitness):
        """
        Check whether to stop training
        Args:
            epoch (int): Current epoch of training
            fitness (float): Fitness value of current epoch
        Returns:
            (bool): True if training should stop, False otherwise
        """
        if fitness is None: return False 

        if fitness >self.best_fitness or self.best_fitness==0: # allow for early zero-fitness stage of training
            self.best_epoch=epoch
            self.best_fitness=fitness
        delta=epoch-self.best_epoch # epochs without improvement
        self.possible_stop=delta>=(self.patience-1) # possible stop may occur next epoch
        stop=delta>=self.patience # stop if patience exceeds
        if stop:
            print(f'EarlyStopping: Training stopped early as no improvement observed in last {self.patience} epochs'
                  f'Best results observed at epoch {self.best_epoch}, best model saved as best.pt\n'
                  f'To update EarlyStopping(patience={self.patience}) pass a new patience value,'
                  f'i.e., `patience=300` or use `patience=0` to disable EarlyStopping')
        return stop
    
def one_cycle(y1=0., y2=1., steps=100):
    """
    Return a lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf.
    Args:
        y1 (float, optional): Initial value
        y2 (float, optional): Final value
        steps (int, optional): Number of steps
    Returns:
        (function): Lambda function for computing sinusoidal ramp
    """
    return lambda x: max(0, (1-math.cos(x*math.pi/steps))/2)*(y2-y1)+y1
    
def init_seeds(seed=0, deterministic=False):
    """
    Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html.
    Args:
        seed (int, optinal): Random seed.
        deterministic (bool, optional): Whether to set deterministic algorithms
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        if int(torch.__version__[0])>=2:
            torch.use_deterministic_algorithms(True, warn_only=True) # warn if deterministic is not possible
            torch.backends.cudnn.deterministic=True
            os.environ['CUBLAS_WORKSPACE_CONFIG']=":4096:8"
            os.environ["PYTHONHASHSEED"]=str(seed)
        else: warnings.warn(f'Upgrade to torch>=2.0.0 for deterministic training')
    else:
        # Unset all the configurations applied for deterministic training
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic=False
        os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
        os.environ.pop('PYTHONHASHSEED', None)

def unwrap_model(m:nn.Module)->nn.Module:
    """
    Unwrap complied and parallel models to get the base model
    Args:
        m (nn.Module): A model that may be wrapped by torch.compile (._orig_mod) or parallel wrappers such as
            DataParallel/DistributedDataParallel (.module)
    Returns:
        (nn.Module): The unwrapped base model without compile or parallel wrappers
    """
    while True:
        if hasattr(m, "_orig_mod") and isinstance(m._orig_mod, nn.Module): m=m._orig_mod
        elif hasattr(m, "module") and isinstance(m.module, nn.Module): m=m.module
        else: return m

def copy_attr(a, b, include=(), exclude=()):
    """
    Copy attributes from object `b` to object `a`, with options to include/exclude certain attributes
    Args:
        a (Any): Destination object to copy attributes to 
        b (Any): Source object to copy attributes from
        include (tuple, optional): Attributes to include. If empty, all attributes are included
        exclude (tuple, optional): Attribute to exclude
    """
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude: continue
        setattr(a, k, v)
        
class ModelEMA:
    """
    Update Exponential Moving Average (EMA) implementation
    
    Keep a moving average of everything in the model state_dict (parameters and buffers). 
    To disable EMA set the `enabled` to `False`
    References:
        - https://github.com/rwightman/pytorch-image-models
        - https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """
    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        """
        Initialize EMA for `model1 with given arguments
        Args:
            model (nn.Module): Model to create EMA for
            decay (float, optional): Maximum EMA decay rate
            tau (int, optional): EMA decay time constant
            updates (int, optional): Initial number of updates
        """
        self.ema=copy.deepcopy(unwrap_model(model)).eval()
        self.updates=updates # number of EMA updates
        self.decay=lambda x: decay*(1-math.exp(-x/tau)) # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters(): p.requires_grad_(False)
        self.enabled=True

    def update(self, model):
        """
        Update EMA parameters.
        Args:
            model (nn.Module): Model to update EMA from.
        """
        if self.enabled:
            self.updates+=1
            d=self.decay(self.updates)
            msd=unwrap_model(model).state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point: # true for FP16, FP32, FP64
                    v*=d
                    v+=(1.-d)*msd[k].detach()
                    
    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        """
        Update attributes
        Args:
            model (nn.Module): Model to update attributes from
            include (tuple, optional): Attributes to include
            exclude (tuple, optional): Attributes to exclude
        """
        if self.enabled: copy_attr(self.ema, model, include, exclude)
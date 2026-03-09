from __future__ import annotations

from typing import Optional, Union

import copy
import inspect

import torch
import torch.nn as nn

def build_conv_layer(cfg:Optional[dict], *args, **kwargs)->nn.Module:
    """Build convolution layer
    Args:
        cfg (dict|None): The conv layer config, which should contain: 
            - type (str): Layer type
            - layer args: Args needed to instantiate an conv layer
        args (argument list): Arguments passed to the `__init__` method of the corresponding conv layer
        kwargs (keyword arguments): Keywaord arguments passed to the `__init__` method of the corresponding conv layer
    Returns:
        (nn.Module): Created conv layer
    Reference: https://github.com/open-mmlab/mmcv/blob/main/mmcv/cnn/bricks/conv.py
    """
    cfg_=copy.deepcopy(cfg)
    layer_type=cfg_.pop('type')
    return getattr(nn, layer_type)(*args, **kwargs, **cfg_)

def infer_abbr(class_type):
    """Infer abbrevation from the class name
    When we build a norm layer with `build_norm_layer()`, we want to preserve the norm type in variable names, 
    e.g., self.bn1, self.gn. This method will infer the abbreviation to map class types to abbreviations.
    Rule 1: If the class has the property '_abbr_'. return the property
    Rule 2: If the parent class is _BatchNorm, GroupNorm, LayerNorm, or InstanceNorm, the abbreviation of this 
        layer will be 'bn', 'gn', 'ln', and 'in', respectively.
    Rule 3: If the class name contains "batch", "group", "layer", or "instance", the abbreviation of this layer
        will be 'bn', 'gn', 'ln', and 'in', respectively.
    Rule 4: Otherwise, the abbreviation falls back to 'norm'

    Args:
        class_type (type): The norm layer type, e.g., <class 'torch.nn.modules.batchnorm.BatchNorm3d'>
    Returns:
        (str): The inferred abbreviation
    Reference: https://github.com/open-mmlab/mmcv/blob/main/mmcv/cnn/bricks/norm.py
    """
    from torch.nn.modules.batchnorm import _BatchNorm
    from torch.nn.modules.instancenorm import _InstanceNorm
    
    assert inspect.isclass(class_type), f"`class_type` must be a type, but got {type(class_type)}"
    if hasattr(class_type, '_abbr_'): return class_type._abbr_
    if issubclass(class_type, _InstanceNorm): return 'in' # IN is a subclass of BN
    if issubclass(class_type, _BatchNorm): return 'bn'
    if issubclass(class_type, nn.GroupNorm): return 'gn'
    if issubclass(class_type, nn.LayerNorm): return 'ln'
    class_name=class_type.__name__.lower()
    if 'batch' in class_name: return 'bn'
    if 'group' in class_name: return 'gn'
    if 'layer' in class_name: return 'ln'
    if 'instance' in class_name: return 'in'
    return 'norm_layer'
    
def build_norm_layer(cfg:dict, num_features:int, postfix:Union[int, str]='')->tuple[str, nn.Module]:
    """Build normalization layer
    Args:
        cfg (dict): The norm layer config, which should contain
            - type (str): Layer type
            - layer args : Args needed to instantiate a norm layer
            - requires_grad (bool, optional): Whether requires parameter value updates
        num_features (int): Number of input channels
        postfix (int|str): The postfix to be append into norm abbrevation to create named layer
    Returns:
        (str): Layer name consisting of abbreviation and postfix, e.g., bn1, gn
        (nn.Module): Norm layer
    Reference: https://github.com/open-mmlab/mmcv/blob/main/mmcv/cnn/bricks/norm.py
    """
    normalizations={'BN':nn.BatchNorm2d, 'BN1d':nn.BatchNorm1d, 'BN2d':nn.BatchNorm2d, 'BN3d':nn.BatchNorm3d,
                   'SyncBN':nn.SyncBatchNorm, 'GN':nn.GroupNorm, "LN":nn.LayerNorm, "IN":nn.InstanceNorm2d,
                    'IN1d':nn.InstanceNorm1d, "IN2d":nn.InstanceNorm2d, "IN3d":nn.InstanceNorm3d}

    assert isinstance(cfg, dict), f'cfg must be a dict, but got {type(cfg)}'
    assert 'type' in cfg, "cfg dict must contain the key 'type'"

    cfg_=copy.deepcopy(cfg)
    layer_type=cfg_.pop('type')
    norm_layer=normalizations[layer_type]

    abbr=infer_abbr(norm_layer)
    assert isinstance(postfix, (int, str))
    name=abbr+str(postfix)

    requires_grad=cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)
    if norm_layer is not nn.GroupNorm:
        layer=norm_layer(num_features, **cfg_)
        if layer_type=='SyncBN' and hasattr(layer, '_specify_ddp_gpu_num'):
            raise NotImplementedError("Please see https://github.com/open-mmlab/mmcv/blob/main/mmcv/cnn/bricks/norm.py#L121")
    else:
        assert 'num_groups' in cfg_
        layer=norm_layer(num_channels=num_features, **cfg)

    for param in layer.parameters(): param.requires_grad=requires_grad
    return name, layer
    
class Clamp(nn.Module):
    """Clamp activation layer

    This activation function is to clamp the feature map value within [`min`, `max`]. 
    See torch.clamp() for more details

    Args:
        min (float, optional): Lower bound of the range to be clamped to. Default to -1
        max (float, optional): Upper bound of the range to be clamped to. Default to 1
    Reference: https://github.com/open-mmlab/mmcv/blob/main/mmcv/cnn/bricks/activation.py#L102
    """
    def __init__(self, min:float=-1., max:float=1.):
        super().__init__()
        self.min=min
        self.max=max

    def forward(self, x)->torch.Tensor:
        """Forward function
        Args:
            x (torch.Tensor): The input tensor
        Returns:
            (torch.Tensor): Clampled tensor
        """
        return torch.clamp(x, min=self.min, max=self.max)
        
def build_activation_layer(cfg:dict)->nn.Module:
    """Build activation layer

    Args:
        cfg (dict): The activation layer config, which should contain
            - type (str): Activation type
            - layer args: Args needed to instantiate an activation layer
    Returns:
        (nn.Module): Created activation layer
    Reference: https://github.com/open-mmlab/mmcv/blob/main/mmcv/cnn/bricks/activation.py#L102
    """
    layer_type=cfg.pop('type')
    if hasattr(nn, layer_type): return getattr(nn, layer_type)(**cfg)
    elif layer_type=='Clamp' or layer_type=='Clip': return Clamp(**act_cfg_)
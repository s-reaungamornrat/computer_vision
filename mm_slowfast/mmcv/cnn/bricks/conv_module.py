from __future__ import annotations

from typing import Optional, Union
from functools import partial

import copy
import warnings
import inspect

import torch
import torch.nn as nn

from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm


from ..brick import build_activation_layer, build_conv_layer, build_norm_layer
from computer_vision.slowfast.mmengine.model.weight_init import kaiming_init, constant_init

def efficient_conv_bn_eval_forward(bn:_BatchNorm, conv:nn.modules.conv._ConvNd, 
                                  x:torch.Tensor):
    """Implementation based on https://arxiv.org/abs/2305.11624 
    Efficient ConvBN Blocks for Transfer Learning and Beyond and sometimes named
    Tune-Mode ConvBN Blocks For Efficient Transfer Learning 
    It leverages the associative law between convolution and affine transform, i.e., 
    normalize (weight conv feature)=(normalize weight) conv feature. It works for Eval
    mode of ConvBN blocks during validation, and can be used for training as well. It 
    reduces memory and computation cost

    The function call _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]) directly with fused weights 
    to speed up inference without changing the actual conv.weight stored in the layer.
    Args:
        bn (_BatchNorm): BatchNorm module
        conv (nn._ConvNd): Conv module
        x (torch.Tensor): Input feature map
    """
    # These lines of code are designed to deal with various cases like bn without affine transform, and conv 
    # without bias
    weight_on_the_fly=conv.weight
    if conv.bias is not None: bias_on_the_fly=conv.bias
    else: bias_on_the_fly=torch.zeros_like(bn.running_var)

    if bn.weight is not None: bn_weight=bn.weight
    else: bn_weight=torch.ones_like(bn.running_var)

    if bn.bias is not None: bn_bias=bn.bias
    else: bn_bias=torch.zeros_like(bn.running_var)

    # reshape to shape of [C_out, 1, 1, 1] for 2D and [C_out, 1, 1, 1, 1] for 3D
    weight_coeff=torch.rsqrt(bn.running_var+bn.eps).reshape([-1]+[1]*(len(conv.weight.shape)-1))
    # shape of [C_out, 1, 1, 1] in 2D and [C_out, 1, 1, 1, 1] in 3D
    coeff_on_the_fly=bn_weight.view_as(weight_coeff)*weight_coeff
    # shape of [C_out, C_in, k, k] in 2D and [C_out, C_in, k, k, k] in 3D
    weight_on_the_fly=weight_on_the_fly*coeff_on_the_fly
    # shape of [C_out] in 2D and 3D
    bias_on_the_fly=bn_bias+coeff_on_the_fly.flatten()*(bias_on_the_fly-bn.running_mean)
    return conv._conv_forward(x, weight_on_the_fly, bias_on_the_fly)
    
class ConvModule(nn.Module):
    """A conv block that bundles conv/norm/activation layers
    
    This block simplifies the usage of convolution layers whicch are commonly used with a norm layer (e.g., BatchNorm) and activation layer
    (e.g., ReLU). It is based on three build methods: `build_conv_layer()`, `build_norm_layer()`, and `build_activation_layer()`

    Args:
        in_channels (int): Number of channels in the input feature map
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int| tuple[int]): Size of the convolving kernel
        stride (int|tuple[int]): Stride of the convolution
        padding (int|tuple[int]): Zero-padding added to both sides of the input
        dilation (int|tuple[int]): Spacing between kernel elements
        groups (int): Number of blocked connections from input channels to output channels
        bias (bool|str): If specified as `auto`, it will be decided by the norm_cfg. Bias will be set as True if `norm_cfg` is None,
            otherwise, False. Default: `auto`
        conv_cfg (dict): Config dict for convolution layer. Default: None, which means using conv2d
        norm_cfg (dict): Config dict for normalization layer. Default: None
        act_cfg (dict): Config dict for activation layer. Default: dict(type='ReLU')
        inplace (bool): Whether to use inplace mode for activation. Default: True
        with_spectral_norm (bool): Whether use spectral norm in conv module. Default: False
        padding_mode (str): 'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'
        order (tuple[str]): The order of conv/norm/activation layers. It is a sequence of 'conv', 'norm', and 'act'. Common examples are
            ('conv', 'norm', 'act') and ('act', 'conv', 'norm'). Default: ('conv', 'norm', 'act') 
        efficient_conv_bn_eval (bool): Whether use efficient conv when the consecutive bn is in eval mode (either training or testing), as proposed
        in https://arxiv.org/abs/2305.11624. Default: False

    Reference: https://github.com/open-mmlab/mmcv/blob/main/mmcv/cnn/bricks/conv_module.py#L256
    """
    _abbr_='conv_block'
    
    def __init__(self, in_channels:int, out_channels:int, kernel_size:Union[int, tuple[int]], stride:Union[int, tuple[int]]=1,
                padding:Union[int, tuple[int]]=0, dilation:Union[int, tuple[int]]=1,groups:int=1, bias:Union[bool,str]='auto',
                conv_cfg:Optional[dict]=None, norm_cfg:Optional[dict]=None, act_cfg:Optional[dict]=dict(type='ReLU'), inplace:bool=True,
                with_spectral_norm:bool=False, padding_mode:str='replicate', order:tuple=('conv', 'norm', 'act'), efficient_conv_bn_eval:bool=False):
        super().__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        self.conv_cfg=conv_cfg
        self.norm_cfg=norm_cfg
        self.act_cfg=act_cfg
        self.inplace=inplace
        self.with_spectral_norm=with_spectral_norm
        self.order=order
        assert isinstance(self.order, tuple) and len(self.order)==3
        assert set(self.order)=={'conv', 'norm', 'act'}

        self.with_norm=norm_cfg is not None
        self.with_activation=act_cfg is not None
        # if the conv layer is before a norm layer, bias is unnecessary
        if bias=='auto': bias=not self.with_norm
        self.with_bias=bias

        # build convolution layer
        self.conv=build_conv_layer(conv_cfg, in_channels, out_channels, kernel_size, stride=stride, padding=padding, 
                                   dilation=dilation, groups=groups, bias=bias)
        # export the attributes of self.conv to a higher level for convenience
        self.in_channels=self.conv.in_channels
        self.out_channels=self.conv.out_channels
        self.kernel_size=self.conv.kernel_size
        self.stride=self.conv.stride
        self.padding=padding
        self.dilation=self.conv.dilation
        self.transposed=self.conv.transposed # boolean flag telling whether the convolution is a normal convolution or a transposed convolution
        self.output_padding=self.conv.output_padding
        self.groups=self.conv.groups

        if self.with_spectral_norm: self.conv=nn.utils.spectral_norm(self.conv)

        # build normalization layer
        if self.with_norm:
            # norm layer is after conv layer
            if order.index('norm')>order.index('conv'): norm_channels=out_channels
            else: norm_channels=in_channels
            self.norm_name, norm=build_norm_layer(norm_cfg, norm_channels)
            self.add_module(self.norm_name, norm) # add_module is torch.nn.Module member function
            if self.with_bias:
                if isinstance(norm, (_BatchNorm, _InstanceNorm)):
                    warnings.warn("Unnecessary conv bias before batch/instance norm")
        else: self.norm_name=None
            

        self.turn_on_efficient_conv_bn_eval(efficient_conv_bn_eval)

        # build activation layer
        if self.with_activation:
            act_cfg_=copy.deepcopy(act_cfg)
            # activation without 'inplace' argument
            if act_cfg_['type'] not in ['Tanh', 'PReLU', 'Sigmoid', 'HSigmoid', 'Swish', 'GELU']:
                act_cfg_.setdefault('inplace', inplace)
            self.activation=build_activation_layer(act_cfg_)

        self.init_weights()

    @property
    def norm(self):
        if self.norm_name: return getattr(self, self.norm_name)
        return None

    def turn_on_efficient_conv_bn_eval(self, efficient_conv_bn_eval=True):
        """Set a function to fuse Conv + BatchNorm into a single equivalent convolution  during evaluation
       to speed up inference. The operation is reversable and only affect the evaluation mode
        """
        # efficient_conv_bn_eval works for conv+bn with `track_running_stats` option
        if efficient_conv_bn_eval and self.norm and isinstance(self.norm, _BatchNorm) and \
            self.norm.track_running_stats:
            self.efficient_conv_bn_eval_forward=efficient_conv_bn_eval_forward
        else: self.efficient_conv_bn_eval_forward=None

    def init_weights(self):
        # 1. It is mainly for customized conv layers with their own initialization manners by calling their own
        #    ``init_weight()`` and we do not want ConvModule to override the initialization.
        # 2. For customized conv layers without their own initialization manner (that is, they do not have their own
        #    ``init_weight()``) and PyTorch's conv layers, they will be initialized by this method with default
        #    ``kaiming_init``
        # Note: For PyTorch's conv layers, they will be overwritten by our initialization implementation using default
        # ``kaiming_init``
        if not hasattr(self.conv, 'init_weights'):
            if self.with_activation and self.act_cfg['type']=='LeakyReLU':
                nonlinearity='leaky_relu'
                a=self.act_cfg.get('negative_slope', 0.01)
            else:
                nonlinearity='relu'
                a=0
            kaiming_init(self.conv, a=a, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm,1,bias=0)

    @staticmethod
    def create_from_conv_bn(conv:nn.modules.conv._ConvNd, 
                            bn:nn.modules.batchnorm._BatchNorm,
                            efficient_conv_bn_eval=True)->"ConvModule":
        """Create a ConvModule from a conv and a bn module"""
        self=ConvModule.__new__(ConvModule)
        super(ConvModule, self).__init__()

        self.conv_cfg=None
        self.norm_cfg=None
        self.act_cfg=None
        self.inplace=False
        self.with_spectral_norm=False
        self.order=('conv', 'norm', 'act')
        self.with_norm=True
        self.with_activation=False
        self.with_bias=conv.bias is not None

        # build convolution layer
        self.conv=conv
        # export the attributes of self.conv to a higher level for conveniece
        self.in_channels=self.conv.in_channels
        self.out_channels=self.conv.out_channels
        self.kernel_size=self.conv.kernel_size
        self.stride=self.conv.stride
        self.padding=self.conv.padding
        self.dilation=self.conv.dilation
        self.transposed=self.conv.transposed
        self.output_padding=self.conv.output_padding
        self.groups=self.conv.groups
        
        # build normalization layer
        self.norm_name, norm='bn', bn
        self.add_module(self.norm_name, norm)
        
        self.turn_on_efficient_conv_bn_eval(efficient_conv_bn_eval)
        
        return self

    def forward(self, x:torch.Tensor, activate:bool=True, norm:bool=True)->torch.Tensor:
        """ Do not know what the benefit of having `activate` and `norm` here as argument since there are
        `self.with_norm` and `self.with_activation`, but since mmcv implemented this way, we keep it
        """
        layer_index=0
        while layer_index<len(self.order):
            
            layer=self.order[layer_index]
            
            if layer=='conv':
                # if the next operation is norm and we have a norm layer in eval mode and we have enable 
                # `efficient_conv_bn_eval` for the conv operator, then use the optimized forward and skip
                # the next norm operator since it has been fused
                if layer_index+1<len(self.order) and self.order[layer_index+1]=='norm' and norm and self.with_norm and \
                not self.norm.training and self.efficient_conv_bn_eval_forward is not None:
                    self.conv.forward=partial(self.efficient_conv_bn_eval_forward, self.norm, self.conv)
                    layer_index+=1 # merged with `layer_index+=1` at the end of the loop to skip the next norm layer
                    x=self.conv(x)
                    del self.conv.forward
                else: x=self.conv(x)
                    
            elif layer=='norm' and norm and self.with_norm: x=self.norm(x)
                
            elif layer=='act' and activate and self.with_activation: x=self.activation(x)
                
            layer_index+=1
    
        return x

if __name__ == '__main__':
    
    conv=ConvModule(in_channels=3, out_channels=16, kernel_size=(5, 7, 7), stride=2, padding=3, dilation=1,groups=1, bias='auto',
                    conv_cfg=dict(type='Conv3d'), norm_cfg=dict(type='BN3d'), act_cfg=dict(type='ReLU'), inplace=True,
                    with_spectral_norm=True, padding_mode='replicate', order=('conv', 'norm', 'act'), efficient_conv_bn_eval=True)
    
    x=torch.rand(2,3,16,120,120)
    out=conv(x)
    nn.MSELoss()(out, torch.rand_like(out)).backward()
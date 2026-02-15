from __future__ import annotations

import copy
import warnings
from typing import Union, Optional, Sequence

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from torch.nn.modules.utils import _ntuple, _triple

from computer_vision.slowfast.mmcv.cnn.bricks.conv_module import ConvModule
from computer_vision.slowfast.mmcv.cnn.brick import build_activation_layer

class BasicBlock3d(nn.Module):
    """BasicBlock 3d block for ResNet3D
    Args:
        inplanes (int): Number of channels for the input in first conv3d layer
        planes (int): Number of channels produced by some norm/conv3d layer
        spatial_stride (int): Spatial stride in the conv3d layer. Default to 1
        temporal_stride (int): Temporal stride in the conv3d layer. Default to 1
        dilation (int): Spacing between kernel elements. Default to 1
        downsample (nn.Module | None): Downsample layer. Default to None
        style (str): 'pytorch' or 'caffe'. If set to 'pytorch', the stride-two layer is rge 3x3 conv layer;
            otherwise, the stride-two layer is the first 1x1 conv layer. Default to 'pytorch'
        inflate (bool): Whether to inflate kernel. Default to True. Note inflation is the process of turning a 2D kernel into a 3D kernel
        non_local (bool): Determine whether to apply non-local module in this block. Default to False
        non_local_cfg (dict): Config for non-local module. Default to ``dict()``
        conf_cfg (dict): Config for convolution layer. Default to ``dict(type='Conv3d')``
        norm_cfg (dict): Config for norm layer.  Required keys are ``type``. Default to ``dict(type='BN3d')``
        act_cfg (dict): Config dict for activation layer. Default to ``dict(type='ReLU')``
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some memory while slowing down the 
            training speed. Defaul to False
        init_cfg (dict | list[dict], optional): Initialization config dict. Default to None
    """
    expansion=1
    def __init__(self, inplanes:int, planes:int, spatial_stride:int=1, temporal_stride:int=1, dilation:int=1,
                downsample:Optional[nn.Module]=None, style:str='pytorch', inflate:bool=True, non_local:bool=False,
                non_local_cfg:dict=dict(), conv_cfg:dict=dict(type='Conv3d'), norm_cfg:dict=dict(type='BN3d'), 
                 act_cfg:dict=dict(type='ReLU'), with_cp:bool=False, init_cfg:Optional[Union[dict, list[dict]]]=None, 
                **kwargs)->None:
        super().__init__()
        assert style in ['pytorch', 'caffe']
        # Note "inflation" is the process of turning a 2D kernel into a 3D kernel and inflate_style controls how that inflation is done
        # Make sure that the only additionally allowed keys inside kwargs is 'inflate_style'
        assert set(kwargs).issubset(['inflate_style'])

        self.inplanes=inplanes
        self.planes=planes
        self.spatial_stride=spatial_stride
        self.temporal_stride=temporal_stride
        self.dilation=dilation
        self.style=style
        self.inflate=inflate
        self.conv_cfg=conv_cfg
        self.norm_cfg=norm_cfg
        self.act_cfg=act_cfg
        self.with_cp=with_cp
        self.non_local=non_local
        self.non_local_cfg=non_local_cfg

        self.conv1_stride_s=spatial_stride
        self.conv2_stride_s=1
        self.conv1_stride_t=temporal_stride
        self.conv2_stride_t=1

        if self.inflate:
            conv1_kernel_size=(3,3,3)
            conv1_padding=(1, dilation, dilation)
            conv2_kernel_size=(3,3,3)
            conv2_padding=(1,1,1)
        else:
            conv1_kernel_size=(1,3,3)
            conv1_padding=(0, dilation, dilation)
            conv2_kernel_size=(1,3,3)
            conv2_padding=(0,1,1)
        self.conv1=ConvModule(inplanes, planes, conv1_kernel_size, stride=(self.conv1_stride_t, self.conv1_stride_s,
                                                                          self.conv1_stride_s),
                              padding=conv1_padding, dilation=(1, dilation, dilation), bias=False, conv_cfg=self.conv_cfg,
                              norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.conv2=ConvModule(planes, planes*self.expansion, conv2_kernel_size, stride=(self.conv2_stride_t, self.conv2_stride_s,
                                                                                       self.conv2_stride_s),
                             padding=conv2_padding, bias=False, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=None)
        self.downsample=downsample
        act_cfg_=copy.deepcopy(self.act_cfg)
        self.relu=build_activation_layer(act_cfg_)

        if self.non_local:
            #self.non_local_block=NonLocal3d(self.conv2.norm.num_features, **self.non_local_cfg)
            raise NotImplementedError("Please see https://github.com/open-mmlab/mmaction2/blob/main/mmaction/models/backbones/resnet3d.py")


class Bottleneck3d(nn.Module):
    """Bottleneck 3d block for ResNet3d
    Args:
        inplanes (int): Number of channels for the input in first conv3d layer.
        planes (int): Number of channels produced by some norm/conv3d layers
        spatial_stride (int): Spatial stride in the conv3d layer. Default to 1
        temporal_stride (int): Temporal stride in the conv3d layer. Default to 1
        dilation (int): Spacing between kernel elements. Default to 1
        downsample (nn.Module, optional): Downsample layer. Default to None
        style (str): 'pytorch' or 'caffe'. If set to 'pytorch', the stride-two layer is the 3x3 conv layer;
            otherwise, the stride-two layer is the first 1x1 conv layer. Default to 'pytorch'
        inflate (bool): Whether to inflate kernel. Default to True
        inflate_style (str): '3x1x1' or '3x3x3', determining the kernel sizes and padding strides for conv1 
            and conv2 in each block. Default to '3x1x1'
        non_local (bool): Whether to apply non-local module in this block. Default to False
        non_local_cfg (dict): Config for non-local modul. Default to dict()
        conv_cfg (dict): Config dict for convolution layer. Default to dict(type='Conv3d')
        norm_cfg (dict): Config for norm layers, required keys are 'type'. Default to dict(type='BN3d')
        act_cfg (dict): Config dict for activation layer. Default to dict(type='ReLU')
        with_cp (bool): Whether to use checkpoint or not. Using checkpoint will save some memory while slowing down the training
            speed. Default to False
        init_cfg (dict|list[dict], optional): Initialization config dict. Default to None
    """
    expansion=4
    def __init__(self, inplanes:int, planes:int, spatial_stride:int=1, temporal_stride:int=1, dilation:int=1,
                downsample:Optional[nn.Module]=None, style:str='pytorch', inflate:bool=True, inflate_style:str='3x1x1',
                non_local:bool=False, non_local_cfg:dict=dict(), conv_cfg:dict=dict(type='Conv3d'), 
                 norm_cfg:dict=dict(type='BN3d'), act_cfg:dict=dict(type='ReLU'), with_cp:bool=False,
                 init_cfg:Optional[dict,list[dict]]=None)->None:
        super().__init__()
        assert style in ['pytorch', 'caffe']
        assert inflate_style in ['3x1x1', '3x3x3']

        self.inplanes=inplanes
        self.planes=planes
        self.spatial_stride=spatial_stride
        self.temporal_stride=temporal_stride
        self.dilation=dilation
        self.style=style
        self.inflate=inflate
        self.inflate_style=inflate_style
        self.norm_cfg=norm_cfg
        self.conv_cfg=conv_cfg
        self.act_cfg=act_cfg
        self.with_cp=with_cp
        self.non_local=non_local
        self.non_local_cfg=non_local_cfg

        if self.style=='pytorch':
            self.conv1_stride_s=1
            self.conv2_stride_s=spatial_stride
            self.conv1_stride_t=1
            self.conv2_stride_t=temporal_stride
        else:
            self.conv1_stride_s=spatial_stride
            self.conv2_stride_s=1
            self.conv1_stride_t=temporal_stride
            self.conv2_stride_t=1

        if self.inflate:
            if inflate_style=='3x1x1':
                conv1_kernel_size=(3,1,1)
                conv1_padding=(1,0,0)
                conv2_kernel_size=(1,3,3)
                conv2_padding=(0, dilation, dilation)
            else:
                conv1_kernel_size=(1,1,1)
                conv1_padding=(0,0,0)
                conv2_kernel_size=(3,3,3)
                conv2_padding=(1, dilation, dilation)
        else:
            conv1_kernel_size=(1,1,1)
            conv1_padding=(0,0,0)
            conv2_kernel_size=(1,3,3)
            conv2_padding=(0, dilation, dilation)

        self.conv1=ConvModule(inplanes, planes, conv1_kernel_size, stride=(self.conv1_stride_t, self.conv1_stride_s,
                                                                          self.conv1_stride_s),
                             padding=conv1_padding, bias=False, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.conv2=ConvModule(planes, planes, conv2_kernel_size, stride=(self.conv2_stride_t, self.conv2_stride_s, 
                                                                        self.conv2_stride_s),
                             padding=conv2_padding, dilation=(1, dilation, dilation), bias=False, conv_cfg=self.conv_cfg,
                             norm_cfg=self.norm_cfgm, act_cfg=self.act_cfg)
        self.conv3=ConvModule(planes, planes*self.expansion, 1, bias=False, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg,
                             act_cfg=None) # No activation in the third ConvModule for bottleneck
        self.downsample=downsample
        act_cfg_=copy.deepcopy(self.act_cfg)
        self.relu=build_activation_layer(act_cfg_)

        if self.non_local:
            raise NotImplementedError('Plase see https://github.com/open-mmlab/mmaction2/blob/main/mmaction/models/backbones/resnet3d.py')
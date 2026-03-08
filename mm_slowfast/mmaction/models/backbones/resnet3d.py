from __future__ import annotations

import copy
import warnings
from typing import Union, Optional, Sequence

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from torch.nn.modules.utils import _ntuple, _triple
from torch.nn.modules.batchnorm import _BatchNorm

from computer_vision.slowfast.mmcv.cnn.bricks.conv_module import ConvModule
from computer_vision.slowfast.mmcv.cnn.brick import build_activation_layer
from computer_vision.slowfast.mmengine.model.weight_init import kaiming_init, constant_init

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
            
    def forward(self, x:torch.Tensor)->torch.Tensor:
    
        def _inner_forward(x):
            """Forward wrapper for utilizing checkpoint"""
            identity=x
            out=self.conv1(x)
            out=self.conv2(out)
            if self.downsample is not None: identity=self.downsample(x)
            out=out+identity
            return out
            
        if self.with_cp and x.requires_grad: out=cp.checkpoint(_inner_forward, x)
        else: out=_inner_forward(x)
        out=self.relu(out)
        
        if self.non_local: 
            raise NotImplementedError("Please see https://github.com/open-mmlab/mmaction2/blob/main/mmaction/models/backbones/resnet3d.py#L137")
        return out

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
                             norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.conv3=ConvModule(planes, planes*self.expansion, 1, bias=False, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg,
                             act_cfg=None) # No activation in the third ConvModule for bottleneck
        self.downsample=downsample
        act_cfg_=copy.deepcopy(self.act_cfg)
        self.relu=build_activation_layer(act_cfg_)

        if self.non_local:
            raise NotImplementedError('Plase see https://github.com/open-mmlab/mmaction2/blob/main/mmaction/models/backbones/resnet3d.py')

    def forward(self, x:torch.Tensor)->torch.Tensor:
    
        def _inner_forward(x):
            """Forward wrapper for utilizing checkpoint"""
            identity=x
            
            out=self.conv1(x)
            out=self.conv2(out)
            out=self.conv3(out)
            if self.downsample is not None: identity=self.downsample(x)
            out=out+identity
            return out
    
        if self.with_cp and x.requires_grad: out=cp.checkpoint(_inner_forward, x)
        else: out=_inner_forward(x)
        out=self.relu(out)
    
        if self.non_local: raise NotImplementedError("Please see https://github.com/open-mmlab/mmaction2/blob/main/mmaction/models/backbones/resnet3d.py#L137")
        return out
        
class ResNet3d(nn.Module):
    """ResNet 3d backbone
    Args:
        depth (int): Depth of resnet, from (18,34,50,101,152}. Default to 50
        pretrained (str, optional): Name of pretrained model. Default to None
        stage_blocks (tuple, optional): Number of residual blocks in each stage. Default to None
        pretrained2d (bool): Whether to load pretrained 2D model. Default to True
        in_channels (int): Channel number of input features. Default to 3
        num_stages (int): Number of stages. Default to 4
        base_channels (int): Channel number of stem output features. Default to 64
        out_indices (sequence[int]): Indices of output features. Default to ``(3,)``
        spatial_strides (sequence[int]): Spatial strides of residual blocks of each stage. Default to ``(1,2,2,2)``
        temporal_strides (sequence[int]): Temporal strides of residual blocks of each stage. Default to ``(1,1,1,1)``
        dilations (sequence[int]): Dilation of each stage. Default to ``(1,1,1,1)``
        conv1_kernel (sequence[int]): Kernel size of the first conv layer. Default to ``(3,7,7)``
        conv1_stride_s (int): Spatial stride of the first conv layer. Default to 2
        conv1_stride_t (int): Temporal stride of the first conv layer. Defaul to 1
        pool1_stride_s (int): Spatial stride of the first pooling layer. Default to 2
        pool1_stride_t (int): Temporal stride of the first pooling layer. Default to 1
        with_pool2 (bool): Whether to use pool2. Default to True
        style (str): 'pytorch' or 'caffe'. If set to 'pytorch', the stride-tow layer is the 3x3 conv layer; otherwise, 
            the stride-two layer is the first 1x1 conv layer. Default to 'pytorch'
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means not freezing any parameters. Default to -1
        inflate (sequence[int]): Inflate dimensions of each block. Default to ``(1,1,1,1)``
        inflate_style (str): '3x1x1' or '3x3x3' which determines the kernel sizes and padding strides for conv1 and conv2 in each
            block. Default to '3x1x1'
        conv_cfg (dict): Config for conv layer. Required keys is ``type``. Default to dict(type='Conv3d')
        norm_cfg (dict): Config for norm layer. Required keys are ``type`` and ``requires_grad``. Defaul to dict(type='BN3d', requires_grad=True)
        act_cfg (dict): Config dict for activation layer. Default to dict(type='ReLU', inplace=True)
        norm_eval (bool): Whether to set BN layers to eval mode, namely, freeze running stats (``mean`` and ``var``). Default to False
        with_cp (bool): Whether to use checkpoint or not. Using checkpoint will save some memory while slowing down the training speed.
            Default to False
        non_local (sequence[int]): Determine whether to apply non-local module in the corresponding block of each stage. Default to ``(0,0,0,0)``
        zero_init_residual (bool): Whether to use zero initialization for residual block. Defaul to True
        init_cfg (dict|list[dict],optional): Initialization config dict. Default to None
    """
    arch_settings={18:(BasicBlock3d, (2,2,2,2)),
                  34:(BasicBlock3d, (3,4,6,3)),
                  50:(Bottleneck3d, (3,4,6,3)),
                  101:(Bottleneck3d, (3,4,23,3)),
                  152:(Bottleneck3d, (3,8,36,3))}
    def __init__(self, depth:int=50, pretrained:Optional[str]=None, stage_blocks:Optional[tuple]=None,
                 pretrained2d:bool=True, in_channels:int=3, num_stages:int=4, base_channels:int=64,
                 out_indices:Sequence[int]=(3,), spatial_strides:Sequence[int]=(1,2,2,2), 
                 temporal_strides:Sequence[int]=(1,1,1,1), dilations:Sequence[int]=(1,1,1,1), 
                 conv1_kernel:Sequence[int]=(3,7,7), conv1_stride_s:int=2, conv1_stride_t:int=1,
                 pool1_stride_s:int=2, pool1_stride_t:int=1, with_pool1:bool=True, with_pool2:bool=True,
                 style:str='pytorch', frozen_stages:int=-1, inflate:Sequence[int]=(1,1,1,1), inflate_style:str='3x1x1',
                 conv_cfg:dict=dict(type='Conv3d'), norm_cfg:dict=dict(type='BN3d', requires_grad=True),
                 act_cfg:dict=dict(type='ReLU', inplace=True), norm_eval:bool=False, with_cp:bool=False, 
                 non_local:Sequence[int]=(0,0,0,0), non_local_cfg:dict=dict(), zero_init_residual:bool=True, 
                 init_cfg:Optional[Union[dict, list[dict]]]=None,verbose=False, **kwargs)->None:
        super().__init__()
        assert depth in self.arch_settings, f"Invalid depth {depth} for resnet"
        self.verbose=verbose # print details on model construction 
        self.depth=depth
        self.pretrained=pretrained
        self.pretrained2d=pretrained2d
        self.in_channels=in_channels
        self.base_channels=base_channels
        self.num_stages=num_stages
        assert 1<=num_stages<=4, f"num_stages must be in range [1,4], but got {num_stages}"
        self.stage_blocks=stage_blocks
        self.out_indices=out_indices
        assert max(out_indices)<num_stages, f"{max(out_indices)=} must be less than {num_stages=}"
        self.spatial_strides=spatial_strides
        self.temporal_strides=temporal_strides
        self.dilations=dilations
        assert len(spatial_strides)==len(temporal_strides)==len(dilations)==num_stages
        if self.stage_blocks is not None: assert len(self.stage_blocks)==num_stages

        self.conv1_kernel=conv1_kernel
        self.conv1_stride_s=conv1_stride_s
        self.conv1_stride_t=conv1_stride_t
        self.pool1_stride_s=pool1_stride_s
        self.pool1_stride_t=pool1_stride_t
        self.with_pool1=with_pool1
        self.with_pool2=with_pool2
        self.style=style
        self.frozen_stages=frozen_stages
        self.stage_inflations=_ntuple(num_stages)(inflate) # repeat `inflate` for `num_stages` times
        self.non_local_stages=_ntuple(num_stages)(non_local)
        self.inflate_style=inflate_style
        self.conv_cfg=conv_cfg
        self.norm_cfg=norm_cfg
        self.act_cfg=act_cfg
        self.norm_eval=norm_eval
        self.with_cp=with_cp
        self.zero_init_residual=zero_init_residual
        
        self.block, stage_blocks=self.arch_settings[depth]
        if self.stage_blocks is None: self.stage_blocks=stage_blocks[:num_stages]
        self.inplanes=self.base_channels
        self.non_local_cfg=non_local_cfg

        self._make_stem_layer()

        self.res_layers=[]
        lateral_inplanes=getattr(self, 'lateral_inplanes', [0,0,0,0])
        for i, num_blocks in enumerate(self.stage_blocks):
            if self.verbose: print(f"stage {i}", '-'*20)
            spatial_stride=spatial_strides[i]
            temporal_stride=temporal_strides[i]
            dilation=dilations[i]
            planes=self.base_channels * 2**i
            if self.verbose:print(f"{spatial_stride=}, {temporal_stride=}, {temporal_stride=}, {self.inplanes=}, {planes=}")
            res_layer=self.make_res_layer(self.block, self.inplanes+lateral_inplanes[i], planes, num_blocks, 
                                          spatial_stride=spatial_stride, temporal_stride=temporal_stride,
                                          dilation=dilation, style=self.style, norm_cfg=self.norm_cfg,
                                          conv_cfg=self.conv_cfg, act_cfg=self.act_cfg, non_local=self.non_local_stages[i],
                                          non_local_cfg=self.non_local_cfg, inflate=self.stage_inflations[i],
                                          inflate_style=self.inflate_style, with_cp=with_cp, verbose=self.verbose, **kwargs)
            self.inplanes=planes*self.block.expansion
            layer_name=f'layer{i+1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.feat_dim=self.block.expansion * self.base_channels * 2**(len(self.stage_blocks)-1)

    def _make_stem_layer(self)->None:
        """Construct the stem layers consists of a conv+norm+act module and a pooling layer"""
        self.conv1=ConvModule(self.in_channels, self.base_channels, kernel_size=self.conv1_kernel,
                             stride=(self.conv1_stride_t, self.conv1_stride_s, self.conv1_stride_s),
                             padding=tuple([(k-1)//2 for k in _triple(self.conv1_kernel)]),
                             bias=False, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.max_pool=nn.MaxPool3d(kernel_size=(1,3,3), stride=(self.pool1_stride_t, self.pool1_stride_s,
                                                               self.pool1_stride_s), padding=(0,1,1))
        self.pool2=nn.MaxPool3d(kernel_size=(2,1,1), stride=(2,1,1))

    @staticmethod
    def make_res_layer(block:nn.Module, inplanes:int, planes:int, blocks:int, spatial_stride:Union[int, Sequence[int]]=1,
                      temporal_stride:Union[int, Sequence[int]]=1, dilation:int=1, style:str='pytorch',
                      inflate:Union[int, Sequence[int]]=1, inflate_style:str='3x1x1', non_local:Union[int, Sequence[int]]=0,
                      non_local_cfg:dict=dict(), norm_cfg:Optional[dict]=None, act_cfg:Optional[dict]=None,
                      conv_cfg:Optional[dict]=None, with_cp:bool=False, verbose=False, **kwargs)->nn.Module:
        """Build residual layer for ResNet3d
        Args:
            block (nn.Module): Residual module to be built.
            inplanes (int): Number of channels for the input feature in each block
            planes (int): Number of channels for the output feature in each block
            blocks (int): Number of residual blocks
            spatial_stride (int|Sequence[int]): Spatial strides in residual and conv layers. Default to 1
            temporal_stride (int|Sequence[int]): Temporal stride in residual and conv layers. Default to 1
            dilation (int): Spacing between kernel elements. Default to 1
            style (str): 'pytorch' or 'caffe'. If set to 'pytorch', the stride-twi layer is the 3x3 conv layer;
                otherwise, the stride-two layer is the first 1x1 conv layer. Default to 'pytorch'
            inflate (int|Sequence[int]): Whether to inflate each block. Default to 1
            inflate_style (str): '3x1x1' or '3x3x3' determining the kernel sizes and padding strides for conv1 and conv2
                in each block. Default to '3x1x1'
            non_local (int|Sequence[int]): Whether to apply non-local module in the corresponding block of each stages. 
                Default to 0
            non_local_cfg (dict): Config for non-local module. Default to dict()
            conv_cfg (dict, optional): Config for conv layers. Default to None
            norm_cfg (dict, optional): Config for norm layers. Default to None
            act_cfg (dict, optional): Config for activate layers. Default to None.
            with_cp (bool, optional): Whether to use checkpoint. Using checkpoint will save some memory while slowing down 
                the training speed. Default to False
            verbose (bool, optional): Whether to print model construction details
        Returns:
            (nn.Module): A residual layer for the given config
        """
        inflate=inflate if not isinstance(inflate, int) else (inflate,)*blocks
        non_local=non_local if not isinstance(non_local, int) else (non_local,)*blocks
        assert len(inflate)==len(non_local)==blocks
        downsample=None
        if spatial_stride!=1 or inplanes!=planes*block.expansion:
            downsample=ConvModule(inplanes, planes*block.expansion, kernel_size=1, 
                                  stride=(temporal_stride, spatial_stride, spatial_stride),
                                 bias=False, conv_cfg=conv_cfg, norm_cfg=norm_cfg,act_cfg=None)
            if verbose: print(f'downsample module: {(spatial_stride!=1)=}, {(inplanes!=planes*block.expansion)=}')
        layers=[]
        layers.append(
            block(inplanes, planes, spatial_stride=spatial_stride, temporal_stride=temporal_stride, dilation=dilation,
                 downsample=downsample, style=style, inflate=(inflate[0]==1), inflate_style=inflate_style, non_local=(non_local[0]==1),
                 non_local_cfg=non_local_cfg, norm_cfg=norm_cfg, conv_cfg=conv_cfg, act_cfg=act_cfg, with_cp=with_cp, **kwargs)
        )
        if verbose: print(f"resblock 0: {inplanes=}, {planes=} {(inflate[0]==1)=}, {(non_local[0]==1)=}, {inflate_style=}")
        inplanes=planes*block.expansion
        for i in range(1, blocks):
            if verbose: print(f"resblock {i}: {inplanes=}, {planes=} {(inflate[0]==1)=}, {(non_local[0]==1)=}, {inflate_style=}")
            layers.append(
                block(inplanes, planes, spatial_stride=1, temporal_stride=1, dilation=dilation, style=style, inflate=(inflate[i]==1),
                     inflate_style=inflate_style, non_local=(non_local[i]==1), non_local_cfg=non_local_cfg, norm_cfg=norm_cfg,
                     conv_cfg=conv_cfg, act_cfg=act_cfg, with_cp=with_cp, **kwargs)
            )
        return nn.Sequential(*layers)

    def inflate_weights(self)->None:
        """Inflate weights"""
        self._inflate_weights(self)
        
    @staticmethod
    def _inflate_weights(self)->None:
        """Inflate the resnet2d parameters to resnet3d

        The differences between resnet3d and resnet2d mainly lie in an extra axis of conv kernel. To utilize the pretrained parameters in 2d models,
        the weight of conv2d models should be inflated to fit the shapes of the 3d counterpart.
        """
        state_dict_r2d=torch.load(self.pretrained, map_location='cpu', weights_only=False)
        assert 'model' in state_dict_r2d, "Make sure that model state_dict was saved with 'model' as a key"
        
        state_dict_r2d=state_dict_r2d['model']

        inflated_param_names=[]
        for name, module in self.named_modules():
            if isinstance(module, ConvModule):
                # we use a ConvModule to wrap conv+bn+relu layers, thus the name mapping is needed
                # i.e., convert the name of 3d module to the name of its corresponding 2 module in the state_dict
                if 'downsample' in name:
                    # layer{X}.{Y}.downsample.conv->layer{X}.{Y}.downsample.0
                    original_conv_name=name+'.0'
                    # layer{X}.{Y}.downsample.bn->layer{X}.{Y}.downsample.1
                    original_bn_name=name+'.1'
                else:
                    # layer{X}.{Y}.conv{n}.conv -> layer{X}.{Y}.conv{n}
                    original_conv_name=name
                    # layer{X}.{Y}.conv{n}.bn -> layer{X}.{Y}.bn{n}
                    original_bn_name=name.replace('conv', 'bn')
                if original_conv_name+".weight" not in state_dict_r2d:
                    warnings.warn(f"Module not exist in state_dict_r2d: {original_conv_name}")
                else:
                    shape_2d=state_dict_r2d[original_conv_name+".weight"].shape
                    shape_3d=module.conv.weight.data.shape
                    if shape2d!=shape3d[:2]+shape3d[3:]: warnings.warn(f"Weight shape mismatch for {original_conv_name} "
                                                                       f"with 3d weight shape {shape_3d} and 2d weight shape {shape_2d}")
                    else: self._inflate_conv_params(module.conv, state_dict_r2d, original_conv_name, inflated_param_names)
                if original_bn_name+".weight" not in state_dict_r2d: warnings.warn(f"Module not exist on the state_dict_r2d: {original_bn_name}")
                else: self._inflate_bn_params(module.bn, state_dict_r2d, original_bn_name, inflated_param_names)
        # check if any parameters in the 2d checkpoint are not loaded
        remaining_names=set(state_dict_r2d.keys())-set(inflated_param_names)
        if remain_names: warning.warn(f"These parameters in the 2d checkpoint are not loaded: {remain_names}")

    def _freeze_stages(self)->None:
        """Prevent all the parameters before `self.frozen_stages` from being optimized"""
        if self.frozen_stages>=0:
            self.conv1.eval()
            for param in self.conv1.parameters(): param.requires_grad=False
        for i in range(1, self.frozen_stages+1):
            m=getattr(self,f'layer{i}')
            m.eval()
            for param in m.parameters(): param.requires_grad=False

    @staticmethod
    def _init_weights(self, pretrained:Optional[str]=None)->None:
        """Initialize the parameters either from existing checkpoint or from scratch
        Args:
            pretrained (str|None): The path of the pretrained weight. Will override the original `pretrained` if set.
        """
        if pretrained: self.pretrained=pretrained
        if isinstance(self.pretrained, str):
            print(f"Load model from {self.pretrained}")

            if self.pretrained2d:
                # Inflate 2d model into 3D 
                self.inflate_weights()
            else: 
                # Load 3d model
                checkpoint=torch.load(self.pretrained, map_location='cpu', weights_only=False)
                if 'model' in checkpoint: self.load_state_dict(checkpoint['model'])
        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv3d): kaiming_init(m)
                elif isinstance(m, _BatchNorm): constant_init(m, 1)
            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck3d): constant_init(m.conv3.bn, 0)
                    elif isinstance(m, BasicBlock3d): constant_init(m.conv2.bn, 0)
        else: raise TypeError("pretrained must be str or None")

    def init_weights(self,pretrained:Optional[str]=None)->None:
        self._init_weights(self, pretrained)
        
    def forward(self, x:torch.Tensor)->Union[torch.Tensor, tuple[torch.Tensor]]:
        """
        Returns:
            (torch.Tensor | tuple[torch.Tensor]): The feature extracted by the backbone
        """
        x=self.conv1(x)
        if self.with_pool1: x=self.max_pool(x)
    
        outs=[]
        for i, layer_name in enumerate(self.res_layers):
            res_layer=getattr(self, layer_name)
            x=res_layer(x)
            if i==0 and self.with_pool2: x=self.pool2(x)
            if i in self.out_indices: outs.append(x)
        if len(outs)==1: return outs[0]
            
        return tuple(outs)

    def train(self, mode:bool=True)->None:
        """Set the optimization status when training"""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm): m.eval()
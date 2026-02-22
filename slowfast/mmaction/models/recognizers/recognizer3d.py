from __future__ import annotations
from typing import Iterable, Optional, Union
from collections import OrderedDict

import copy
import inspect
import warnings

import torch.nn as nn

from computer_vision.slowfast.mmaction.models.data_preprocessors.data_preprocessor import ActionDataPreprocessor
from computer_vision.slowfast.mmaction.models.heads.slowfast_head import SlowFastHead
from computer_vision.slowfast.mmaction.models.losses.cross_entropy_loss import CrossEntropyLoss
from computer_vision.slowfast.mmaction.models.backbones.resnet3d_slowfast import ResNet3dSlowFast
from computer_vision.slowfast.mmengine.model.utils import merge_dict
from computer_vision.slowfast.mmaction.utils import ForwardResults
from computer_vision.slowfast.mmengine.optim.optimizer.optimizer_wrapper import OptimWrapper
from computer_vision.slowfast.mmengine.utils.misc import is_list_of
from computer_vision.slowfast.mmengine.model.base_model.data_preprocessor import BaseDataPreprocessor
from computer_vision.slowfast.mmaction.utils import SampleList

class Recognizer3D(nn.Module):
    """Recognizers
    Args:
        backbone (dict): Backbone modules to extract feature
        cls_head (dict): Classification head to process feature. 
        train_cfg (dict, optional): Config for training. Default to None
        test_cfg (dict, optional): Config for testing. Default to None
        data_preprocessor (dict, optional): The pre-process config of class `ActionDataPreprocessor`, usually
            including ``mean``, ``std``, and ``format_shape``. Default to None.
    References:
        https://github.com/open-mmlab/mmaction2/blob/main/mmaction/models/recognizers/recognizer3d.py
        https://github.com/open-mmlab/mmaction2/blob/main/mmaction/models/recognizers/base.py
        https://github.com/open-mmlab/mmengine/blob/main/mmengine/model/base_model/base_model.py
    """
    def __init__(self, backbone:dict, cls_head:dict|None=None, train_cfg:dict|None=None, test_cfg:dict|None=None,
                data_preprocessor:dict|None=None)->None:

        super().__init__()

        data_preprocessor.pop('type')
        self.data_preprocessor=ActionDataPreprocessor(**data_preprocessor)


        # record the source of the backbone
        self.backbone_from='mmaction2'

        # backbone
        backbone_=backbone
        if 'type' in backbone:
            backbone_=copy.deepcopy(backbone)
            backbone_.pop('type')
        self.backbone=ResNet3dSlowFast(**backbone_)
        
        # head
        cls_head_=cls_head
        if 'type' in cls_head:
            cls_head_=copy.deepcopy(cls_head)
            cls_head_.pop('type')
        self.cls_head=SlowFastHead(**cls_head_)
        
        self.train_cfg=train_cfg
        self.test_cfg=test_cfg
        
    @property
    def with_neck(self)->bool:
        """Whethere recognizer has a neck"""
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_cls_head(self)->bool:
        """Whether the recognizer has a cls_head"""
        return hasattr(self, 'cls_head') and self.cls_head is not None

    def loss(self, inputs:torch.Tensor, data_samples:SampleList, **kwargs)->dict:
        """Calculate losses from a batch of inputs and data samples
        Args:
            inputs (torch.Tensor): Raw inputs of recognizer, usually mean centered and std scaled
            data_samples (list[ActionDataSample]): Batch of data samples, usually includes information such as `gt_label`
        Returns:
            (dict): A dict of loss components
        """
        feats, loss_kwargs=self.extract_feat(inputs, data_samples=data_samples)

        # loss_aux will be an empty dict if `self.with_neck` is False
        loss_aux =loss_kwargs.get('loss_aux', dict())
        loss_cls=self.cls_head.loss(feats, data_samples, **loss_kwargs)
        losses=merge_dict(loss_cls, loss_aux)
        return losses

    def predict(self, inputs:torch.Tensor, data_samples:SampleList, **kwargs)->SampleList:
        """Predict results from a batch of inputs and data samples with post-processing
        Args:
            inputs (torch.Tensor): Raw inputs of the recognizer, usually mean centered and std scaled
            data_samples (list[ActionDataSample]): Data sample batch, typically included information such as `gt_label`
        Returns:
            (list[ActionDataSample]): Recognition results. The returns value is ActionDataSample which usually contains
                `pred_scores` and the `pred_scores` usually contains following keys
            - 'item' (torch.Tensor): Classification scores, with shape (num_classes, )????
        """
        feats, predict_kwargs=self.extract_feat(inputs, test_mode=True)
        predictions=self.cls_head.predict(feats, data_samples, **predict_kwargs)
        return predictions

    def _forward(self,inputs:torch.Tensor, stage:str='backbone', **kwargs)->ForwardResults:
        """Network forward process. Usually includes backbone, neck and head forward without any post-processing
        Args:
            inputs (torch.Tensor): Raw inputs of the recognizer
            stage (str): Which stage to output the features
        Returns:
            (Union[dict[str, torch.Tensor], list[ActionDataSample], tuple[torch.Tensor], torch.Tensor]): Features from backbone,
                neck or head
        """
        feats,_=self.extract_feat(inputs, stage=stage)

    def forward(self, inputs:torch.Tensor, data_samples:SampleList|None=None, mode:str='tensor', **kwargs)->ForwardResults:
        """Unify the forward process in both training and testing
        
        The method accepts 3 modes
        - 'tensor': Forward the whole network and return tensor or tuple of tensor without post-processing
        - 'predict': Forward and return the prediction, which are fully processed to a list of ActionDataSample
        - 'loss': Forward and return a dict of losses according to the given inputs and data samples

        Note that this method does not handle neither backpropagation nor optimizer update, which are done in `train_step`

        Args:
            inputs (torch.Tensor): The input tensor with shape (N, C, ...)
            data_samples (list[ActionDataSample], optional): Annotation data of every samples. Default to None
            mode (str): Return what kind of value. Default to 'tensor'
        Returns:
            The return type depends on `mode`
            - If `mode='tensor'`, return a tensor or a tuple of tensors
            - If `mode='predict'`, return a list of ActionDataSample
            - If `mode='loss'`, return a dict of tensor
        """
        if mode=='tensor': return self._forward(inputs, **kwargs)
        elif mode=='predict': return self.predict(inputs, data_samples, **kwargs)
        elif mode=='loss': return self.loss(inputs, data_samples, **kwargs)
        raise RuntimeError(f"Invalid mode {mode}. Only support loss, predict, and tensor mode")

    def _run_forward(self, data:Union[dict, tuple,list], mode:str)->Union[dict[str, torch.Tensor], list]:
        """Unpack data for forward
        Args:
            data (dict |tuple | list): Data sampled from dataset
            mode (str): Mode for forward
        Returns:
            (dict | list): Results of training or testing 
        """
        if isinstance(data, dict): results=self(**data, mode=mode)
        elif isinstance(data, (list, tuple)): results=self(*data, mode=mode)
        else: raise TypeError(f"Output of `data_preprocessor should be list, tuple or dict but got {type(data)}")
        return results

    def parse_losses(self, losses:dict[str, torch.Tensor])->tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Parse the raw output losses of the network
        Args:
            losses (dict): Raw output of the network, which usually contain losses and other necessary information
        Returns:
            (tuple[torch.Tensor, dict[str, torch.Tensor]]): The first is the loss tensor passed to optim_wrapper which may be a weighted sum
                of all losses and the second is log_vars which will be sent to logger
        """
        log_vars=[]
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor): log_vars.append([loss_name,loss_value.mean()])
            elif is_list_of(loss_value, torch.Tensor):
                log_vars.append([loss_name, sum(_loss.mean() for _loss in loss_value)])
            else: raise TypeError(f"{loss_name} is not a tensor or list of tensor")
        loss=sum(value for key, value in log_vars if 'loss' in key)
        log_vars.insert(0, ['loss', loss])
        log_vars=OrderedDict(log_vars)

        return loss, log_vars
        
    def train_step(self, data:Union[dict, tuple, list], optim_wrapper:OptimWrapper)->dict[str, torch.Tensor]:
        """Implement the default model training process including preprocessing, model forward propagation, loss calculation, optimization
        and back propagation
    
        During non-distributed training, if subclasses do not override the `train_step`, `EpochBasedTrainLoop` or `IterBasedTrainLoop` will
        call this method to update model parameters. 
        
        Args:
            data (dict | tuple |list): Data sampled from the dataset
            optim_wrapper (OptimWrapper): OptimWrapper instance
        Returns:
            (dict[str, torch.Tensor]): A dict of tensor for logging
        """
        # 1. Call self.data_preprocessor(data, training=True) to collect batch_inputs and corresponding data_samples(labels)
        with optim_wrapper.optim_context(self):
            data=self.data_preprocessor(data, True)
            # 2. Call self(batch_inputs, data_samples, mode='loss') to get raw loss
            losses=self._run_forward(data, mode='loss')
        # 3. Call self.parse_losses to get parsed_losses tensor used in backward and provide dict of loss tensor for logging 
        parsed_losses, log_vars=self.parse_losses(loss)
        # 4. Call optim_wrapper.update_params(loss) to update mode
        optim_wrapper.update_params(parsed_losses)
        return log_vars

    def val_step(self, data:Union[tuple, dict, list])->list:
        """Get the predictions of given data

        Calls `self.data_preprocessor(data, False)` and `self(inputs, data_sample, mode='predict')` in order. Return the predictions
        which will be passed to evaluator

        Args:
            data (dict|list|tuple): Data sampled from dataset
        Returns:
            (list): The predictions of given data
        """
        data=self.data_preprocessor(data, False)
        return self._run_forward(data, mode='predict')

    def test_step(self, data:Union[dict, tuple, list])->list:
        """The same as val_step
        Args:
            data (dict|list|tuple): Data sampled from dataset
        Returns:
            (list): The predictions of given data
        """
        data=self.data_preprocessor(data, False)
        return self._run_forward(data, mode='predict')

    def _set_device(self, device:torch.device)->None:
        """Recursively set device for `BaseDataPreprocessor` instance

        Args:
            device (torch.device): The desired device of the parameters and buffers in this module
        """
        def apply_fn(module):
            if not isinstance(module, BaseDataPreprocessor): return 
            if device is not None: module._device=device
        self.apply(apply_fn)

    def _extract_feat_in_test(self, inputs:torch.Tensor, stage:str='neck', num_crops:int=1):
        """Extract features of different stages
        
        Args:
            inputs (torch.Tensor): The input tensor of shape (N,C,T,H,W) where N is the batch size or N*num_crops, C is the input channel, 
                and T is the number of frames
            stage (str): Which stage to output the feature. Default to 'head'
            data_samples (list[ActionDataSample], optional): Action data samples, which are only needed in training. Default to None
            num_crops (int): Number of crops
        Returns:
            (torch.Tensor): The extracted features
            (dict): A dict recording the kwargs for downstream pipeline. These keys are usually included: 'loss_aux'
        """
        # Record the kwargs required by `loss` and `predict`
        loss_predict_kwargs=dict()
    
        if self.test_cfg is not None:
            loss_predict_kwargs['fcn_test']=self.test_cfg.get('fcn_test', False)
        if self.test_cfg is not None and self.test_cfg.get('max_testing_views', False):
            assert isinstance(max_testing_views, int)
            total_views=inputs.shape[0]
            assert num_crops==total_views, 'max_testing_views is onlt compatible with batch_size==1'
    
            view_ptr=0
            feats=[]
            while view_ptr<total_views:
                batch_imgs=inputs[view_ptr:view_ptr+max_testing_views]
                feat=self.backbone(batch_imgs)
                if self.with_neck: feat,_=self.neck(feat)
                feats.append(feat)
                view_ptr+=max_testing_views
    
            def recursively_cat(feats):
                # recursively traverse feats until it's a tensor then concat
                out_feats=[]
                for e_idx, elem in enumerate(feats[0]):
                    batch_elem=[feat[e_idx] for feat in feats]
                    if not isinstance(elem, torch.Tensor): batch_elem=recursively_cat(batch_elem)
                    else: batch_elem=torch.cat(batch_elem)
                    out_feats.append(batch_elem)
                return tuple(out_feats)
            if isinstance(feats[0], tuple): x=recursively_cat(feats)
            else: x=torch.cat(feats)
        else: # self.test_cfg is None
            x=self.backbone(inputs)
            if self.with_neck: x, _=self.neck(x)
        return x, loss_predict_kwargs

    def _extract_feat_in_train(self, inputs:torch.Tensor, stage: str='neck', data_samples:Optional[SampleList]=None)->tuple:
        """Extract features of different stages
        
        Args:
            inputs (torch.Tensor): The input tensor of shape  (N, C,T,H,W) or (N, num_crops, C,T,H,W) where N is the batch size, C is the input channel, 
                and T is the number of frames
            stage (str): Which stage to output the feature. Default to 'head'
            data_samples (list[ActionDataSample], optional): Action data samples, which are only needed in training. Default to None
        Returns:
            (torch.Tensor): The extracted features
            (dict): A dict recording the kwargs for downstream pipeline. These keys are usually included: 'loss_aux'
        """
        # Record the kwargs required by `loss` and `predict`
        loss_predict_kwargs=dict()
    
        # Return features extracted through backbone
        x=self.backbone(inputs)
        if stage=='backbone': return x, loss_predict_kwargs
    
        loss_aux=dict()
        if self.with_neck: x, loss_aux=self.neck(x, data_samples=data_samples)
    
        # Return features extracted through neck
        loss_predict_kwargs['loss_aux']=loss_aux
        if stage=='neck': return x, loss_predict_kwargs
    
        # Return raw logits through head
        if self.with_cls_head and stage=='head':
            x=self.cls_head(x, **loss_predict_kwargs)
            return x, loss_predict_kwargs

    def extract_feat(self, inputs:torch.Tensor, stage: str='neck', data_samples:Optional[SampleList]=None, test_mode:bool=False)->tuple:
        """Extract features of different stages
        
        Args:
            inputs (torch.Tensor): The input tensor of shape  (N, C,T,H,W) or (N, num_crops, C,T,H,W) where N is the batch size, C is the input channel,
                and T is the number of  frames
            stage (str): Which stage to output the feature. Default to 'head'
            data_samples (list[ActionDataSample], optional): Action data samples, which are only needed in training. Default to None
            test_mode (bool): Whether running in test mode. Default to False
        Returns:
            (torch.Tensor): The extracted features
            (dict): A dict recording the kwargs for downstream pipeline. These keys are usually included: 'loss_aux'
        """
        num_segs=1 # num_crops
        if inputs.ndim==6: num_segs=inputs.shape[1] # num_crops
        # (N, num_crops, C, T, H, W) -> (N*num_crops, C, T, H, W)
        # num_crops is calculated by 
        # 1) `twice_sample` in `SampleFrames`
        # 2) `num_sample_positions` in `DenseSameFrames`
        # 3) `ThreeCrop/TenCrop` in `test_pipeline`
        # 4) `num_clips` in `SampleFrames` or its subclass if `clip_len!=1`
        inputs=inputs.view((-1,)+inputs.shape[-4:])
        print(f'{inputs.shape=}')
        
        # Check settings of test
        if test_mode:
            return self._extract_feat_in_test(inputs, stage, num_crops=num_segs)
        return self._extract_feat_in_train(inputs, stage, data_samples)
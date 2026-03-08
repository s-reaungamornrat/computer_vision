from __future__ import annotations
from typing import Optional, Sequence, Union

import torch

from computer_vision.slowfast.mmengine.model.base_model.data_preprocessor import BaseDataPreprocessor
from computer_vision.slowfast.mmengine.model.utils import stack_batch
from computer_vision.slowfast.mmaction.utils import SampleList

class ActionDataPreprocessor(BaseDataPreprocessor):
    """Data pre-processor for action recognition tasks
    Args:
        mean (Sequence[float|int], optional): The pixel mean of channels of images or stacked optical flow. Default to None
        std (Sequence[float|int],optional): The pixel standard deviation of channels of images or stacked optical flow. Default to None
        to_rgb (bool): Whether to convert image from BGR to RGB. Default to False
        to_float32 (bool): Whether to convert data to float32. Default to True
        blending (dict, optional): Config for batch blending augmentation such as CutMix and MixUp. Default to None
            see https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/mvit/mvit-small-p244_k400-maskfeat-pre_8xb32-16x4x1-100e_kinetics400-rgb.py
            Example
                blending=dict(
                type='RandomBatchAugment',
                augments=[
                    dict(type='MixupBlending', alpha=0.8, num_classes=400),
                    dict(type='CutmixBlending', alpha=1, num_classes=400)
                ]),
        format_shape (str): Format shape of input data. Default to 'NCHW'
    Reference: https://github.com/open-mmlab/mmaction2/blob/main/mmaction/models/data_preprocessors/data_preprocessor.py
    """
    def __init__(self,mean:Optional[Sequence[Union[float|int]]]=None, std:Optional[Sequence[Union[float|int]]]=None, to_rgb:bool=False, to_float32:bool=True,
                blending:Optional[dict]=None, format_shape:str='NCHW')->None:
        super().__init__()
        self.to_rgb=to_rgb
        self.to_float32=to_float32
        self.format_shape=format_shape

        self._enable_normalize=False
        if mean is not None:
            assert std is not None, "To enable the normalization in preprocessing, please specify both mean and std"
            # Enable the normalization in preprocessing
            self._enable_normalize=True
            if self.format_shape=='NCHW': normalizer_shape=(-1,1,1)
            elif self.format_shape in ['NCTHW', 'MIX2d3d']:normalizer_shape=(-1,1,1,1)
            else: raise ValueError(f'Invalid format shape: {format_shape}')

            self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32).view(normalizer_shape), False)
            self.register_buffer('std', torch.tensor(std, dtype=torch.float32).view(normalizer_shape), False)

        # this is blending augmentation, see example in
        # https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/mvit/mvit-small-p244_k400-maskfeat-pre_8xb32-16x4x1-100e_kinetics400-rgb.py
        self.blending=None
        if blending is not None: raise NotImplementedError("See https://github.com/open-mmlab/mmaction2/blob/main/mmaction/models/utils/blending_utils.py")

    def forward(self,data:Union[dict,Tuple[dict]], training:bool=False)->Union[dict, tuple[dict]]:
        """Perform normalization, padding, bgr2rgb conversion, and batch augmentation
        Args:
            data (dict|tuple[dict]): Data sampled from dataloader
            training (bool): Whether to enable training time augmentation
        Returns:
            (dict | tuple[dict]): Data in the same format as the model input
        """
        data=self.cast_data(data)
        print(f"In mmaction.models.data_preprocessors.data_preprocessor.ActionDataPreprocessor.forward {type(data)=}\n{data.keys()}")
        if isinstance(data, dict): return self.forward_onesample(data, training=training)
        elif isinstance(data, (tuple, list)):
            outputs=[]
            for data_sample in data:
                outputs.append( self.forward_onesample(data_sample, training=training) )
            return tuple(outputs)
        raise TypeError(f"Unsupported data type: {type(data)}")

    def forward_onesample(self, data:dict, training:bool=False)->dict:
        """Perform normalization, padding, bgr2rbg conversion and batch augmentation on one data sample
        Args:
            data (dict): Data sampled from dataloader
            training (bool): Whether to enable training time augmentation
        Returns:
            (dict): Data in the same format as the model input
        """
        inputs, data_samples=data['inputs'], data['data_samples']
        print(f"In mmaction.models.data_preprocessors.data_preprocessor.ActionDataPreprocessor.forward_onesample {[x.shape for x in inputs]=}\n{data_samples}")
        inputs, data_samples=self.preprocess(inputs, data_samples, training)
        data['inputs']=inputs
        data['data_samples']=data_samples
        return data

    def preprocess(self, inputs:list[torch.Tensor], data_samples:SampleList, training:bool=False)->tuple:
        #--- Pad and stack ---
        batch_inputs=stack_batch(inputs)
        print(f"In mmaction.models.data_preprocessors.data_preprocessor.ActionDataPreprocessor.preprocess {type(batch_inputs)=}, {batch_inputs.shape=}")
        if self.format_shape=='MIX2d3d':
            if batch_inputs.ndim==4: format_shape, view_shape='NCHW', (-1,1,1)
            else:format_shape, view_shape='NCTHW', None
        else: format_shape, view_shape=self.format_shape, None
        print(f"In mmaction.models.data_preprocessors.data_preprocessor.ActionDataPreprocessor.preprocess {format_shape=}, {view_shape=}")
        print(f"In mmaction.models.data_preprocessors.data_preprocessor.ActionDataPreprocessor.preprocess {self.to_rgb=}, {self._enable_normalize=}, {self.to_float32=}, {self.blending=}")
        # -----  To RGB -----
        if self.to_rgb:
            if format_shape=='NCHW': batch_inputs=batch_inputs[...,[2,1,0],:,:]
            elif format_shape=='NCTHW': batch_inputs=batch_inputs[...,[2,1,0],:,:,:]
            else: raise ValueError(f"Invalid format shape: {format_shape}")

        # ---- Normalization ----
        if self._enable_normalize:
            if view_shape is None: 
                batch_inputs=(batch_inputs-self.mean)/self.std
                print(f"In mmaction.models.data_preprocessors.data_preprocessor.ActionDataPreprocessor.preprocess normalized batch {type(batch_inputs)=}, {batch_inputs.shape=}")
            else:
                mean=self.mean.view(view_shape)
                std=self.std.view(view_shape)
                batch_inputs=(batch_inputs-mean)/std
        elif self.to_float32:
            batch_inputs=batch_inputs.to(torch.float32)

        # ---- Blending -----
        if training and self.blending is not None:
            batch_inuts, data_samples=self.blending(batch_inputs, data_samples)

        return batch_inputs, data_samples
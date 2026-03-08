from __future__ import annotations
from typing import Mapping, Optional, Sequence, Union

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from computer_vision.slowfast.mmengine.model.utils import stack_batch
from computer_vision.slowfast.mmengine.utils.misc import is_seq_of
from computer_vision.slowfast.mmengine.structures.base_data_element import BaseDataElement

CastData=Union[tuple, dict, BaseDataElement, torch.Tensor, list, bytes, str, None]

class BaseDataPreprocessor(nn.Module):
    """Base data pre-processor used for copying data to the target device

    Subclasses inherit from `BaseDataPreprocessor` could override the forward method to implement custom data pre-processing such as batch-resize, 
    MixUp or CutMix

    Args:
        non_blocking (bool): Whether block current process when transferring data to device.
    Note: 
        Data dict returned by dataloader must be a dict and at least contain the `inputs` key
    Reference: https://github.com/open-mmlab/mmengine/blob/main/mmengine/model/base_model/data_preprocessor.py
    """
    def __init__(self,non_blocking:Optional[bool]=False):
        super().__init__()
        self._non_blocking=non_blocking
        self._device=torch.device('cpu')

    def cast_data(self,data:CastData)->CastData:
        """Copying data to the target device
        Args:
            data (dict): Data returned by ``Dataloader``
        Returns:
            CollatedResult: Inputs and data sample at target device
        """
        if isinstance(data, Mapping): return {key:self.cast_data(data[key]) for key in data}
        elif isinstance(data, (str, bytes)) or data is None: return data
        elif isinstance(data, tuple) and hasattr(data, '_fields'):
            # namedtuple
            return type(data)(*(self.cast_data(sample) for sample in data))
        elif isinstance(data, Sequence):
            return type(data)(self.cast_data(sample) for sample in data)
        elif isinstance(data, (torch.Tensor, BaseDataElement)):
            return data.to(self.device, non_blocking=self._non_blocking)
        return data

    def forward(self, data:dict, training:bool=False)->Union[dict, list]:
        """Preprocess the data into the model input format

        After the data pre-processing of `cast_data`, `forward` will stack the input tensor list to a natch tensor at the first dimension
        Args:
            data (dict): Data returned by dataloader
            training (bool): Whether to enable training time augmentation
        Returns:
            (dict|list): Data in the format of the model input
        """
        return self.cast_data(data)

    @property
    def device(self): return self._device

    def to(self, *args, **kwargs)->nn.Module:
        """Override this method to set `device`
        Returns:
            (nn.Module): Model itself
        """
        # Since torch has not officially merged the npu-related fields, using the _parse_to function directly will cause the NPU to not be found
        # Here the input parameters are processed to avoid error
        if args and isinstance(args[0], str) and 'npu' in args[0]:
            args=tuple([list(args)[0].replace('npu', torch.npu.native_device)])
        if kwargs and 'npu' in str(kwargs.get('device','')):
            kwargs['device']=kwargs['device'].replace('npu', torch.npu.native_device)

        device=torch._C._nn._parse_to(*args, **kwargs)[0]
        if device is not None: self._device=torch.device(device)
        return super().to(*args, **kwargs)

    def cuda(self, *args, **kwargs)->nn.Module:
        """Override to set device
        Returns:
            (nn.Module): Model itself
        """
        self._device=torch.device(torch.cuda.current_device())
        return super().cuda()

    def cpu(self, *args, **kwargs)->nn.Module:
        """Override to set device
        Returns:
            (nn.Module): Model itself
        """
        self._device=torch.device('cpu')
        return super().cpu()
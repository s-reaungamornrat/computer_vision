from __future__ import annotations

import itertools
from collections.abc import Sized
from typing import Any, Union

import numpy as np
import torch

from .base_data_element import BaseDataElement

BoolTypeTensor=Union[torch.BoolTensor, torch.cuda.BoolTensor]
LongTypeTensor=Union[torch.LongTensor, torch.cuda.LongTensor]

IndexType:Union[Any]=Union[str, slice, int, list, LongTypeTensor, BoolTypeTensor, np.ndarray]

# Modified from https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/data_structures/instance_data.py # noqa
class InstanceData(BaseDataElement):
    """Data structure for instance-level annotations or predictions

    Subclass of `BaseDataElement`. All value in `data_fields` should have the same length. This design refer to 
    https://github.com/facebookresearch/detectron2/blob/master/detectron2/structures/instances.py # noqa E501
    InstanceData also support extra functions: `index`, `slice`, and `cat` for data field. The type of value in data field can be
    base data structure such as `torch.Tensor`, `numpy.ndarray`, `list`, `str`, `tuple`, and can be customized data structure that has `__len__`,
    `__getitem__`, and `cat` attributes.
    Reference: https://github.com/open-mmlab/mmengine/blob/main/mmengine/structures/instance_data.py
    """
    def __setattr__(self,name:str,value:Sized):
        """Setattr is only used to set data

        The value must have the attribute of `__len__` and have the same length as `InstanceData`
        """
        if name in ('_metainfo_fields', '_data_fields'):
            if not hasattr(self, name): super().__setattr__(name, value)
            else: raise AttributeError(f"{name} has been used as a private attribute, which is immutable")
        else:
            assert isinstance(value, Sized), "value must contain `__len__` attribute"
            if len(self)>0:
                assert len(value)==len(self), (f"The length of values {len(value)} is not consistent with the length of this InstanceData obj "
                                               f"{len(self)}")
            super().__setattr__(name, value)
            
    __setitem__=__setattr__

    def __getitem__(self, item:IndexType)->"InstanceData":
        """
        Args:
            item (str, int, list, slice, np.ndarray, torch.LongTensor, torch.BoolTensor): indices
        Returns:
            (InstanceData): Corresponding values
        """
        # IndexType.__args__ returns a tuple listing all types making up IndexType
        assert isinstance(item, IndexType.__args__)
        if isinstance(item, list): item=np.array(item)
        if isinstance(item, np.ndarray): 
            # The default int type of numpy is platform dependent, int32 for Windows and int64 for linux. torch.Tensor requires the int64 index,
            # therefore we simply convert it to int64 here. More details in https://github.com/numpy/numpy/issues/9464
            item=item.astype(np.int64) if item.dtype==np.int32 else item
            item=torch.from_numpy(item)
        if isinstance(item, str): return getattr(self, item)
        if isinstance(item, int):
            if item>=len(self) or item<-len(self): raise IndexError(f"Index {item} out of range!")
            else:
                # keep the dimension
                item=slice(item, None, len(self))
        new_data=self.__class__(metainfo=self.metainfo)
        if isinstance(item, torch.Tensor):
            assert item.dim()==1, 'Only support to get the values along the first dimension'
            if isinstance(item, BoolTypeTensor.__args__):
                assert len(item)==len(self), (f"The shape of the input(BoolTensor) {len(item)} does not match the shape of the tensor "
                                              f"{len(self)} at the first dimension")
            for k, v in self.items():
                if isinstance(v, torch.Tensor): new_data[k]=v[item]
                elif isinstance(v, np.ndarray): new_data[k]=v[item.cpu().numpy()]
                elif isinstance(v, (str, list, tuple)) or (hasattr(v, '__getitem__') and hasattr(v, 'cat')):
                    # convert to indices from BoolTensor
                    if isinstance(item, BoolTypeTensor.__args__): indices=torch.nonzeros(item).view(-1).cpu().numpy().tolist()
                    else: indices=item.cpu().numpy().tolist()
                    slice_list=[]
                    if indices:
                        for index in indices: slice_list.append(slice(index,None,len(v)))
                    else: slice_list.append(slice(None,0,None))
                    r_list=[v[s] for s in slice_list]
                    if isinstance(v, (str, list, tuple)): 
                        new_value=r_list[0]
                        for r in r_list[1:]: new_value=new_value+r
                    else: new_value=v.cat(r_list)
                    new_data[k]=new_value
                else: raise ValueError(f"The type of `{k}` is `{type(v)}`, which has no attribute of `cat` so it doe snot support slcie with `bool`")
        else:
            # item is a slice
            for k, v in self.items(): new_data[k]=v[item]

        return new_data

    @staticmethod
    def cat(instances_list:list['InstanceData'])->'InstanceData':
        """Concatenate the instances of all InstanceData obj in the list

        Note: to ensure that cat returns as expected, make sure that all elements in the list must have exactly the same keys

        Args:
            instances_list (list['InstanceData']): A list of 'InstanceData'
        Returns:
            (InstanceData):
        """
        assert all(isinstance(results, InstanceData) for results in instances_list) 
        assert len(instances_list)>0
        if len(instances_list)==1: return instances_list[0]
        # metainfor and data_fields must be exactly the same for each element to avoid exception
        field_keys_list=[instances.all_keys() for instances in instances_list]
        assert len({len(field_keys) for field_keys in field_keys_list})==1 and len(set(itertools.chain(*field_keys_list)))==len(field_keys_list[0]),\
        ("There are different keys in `instances_list`, which may cause the cat operation to fail. "
         "Please make sure all elements in `instances_list` have the exact same key")

        new_data=instances_list[0].__class__(metainfo=metainfo_list[0].metainfo)
        for k in instances_list[0].keys():
            values=[results[k] for results in instances_list]
            v0=values[0]
            if isinstance(v0, torch.Tensor): new_values=torch.cat(values, dim=0)
            elif isinstance(v0, np.ndarray): new_values=np.concatenate(values, axis=0)
            elif isinstance(v0, (str, list, tuple)):
                new_values=v0[:]
                for v in values[1:]: new_values+=v
            elif hasattr(v0, 'cat'): new_values=v0.cat(values)
            else: raise ValueError(f"The type of `{k}` is `{type(v0)}` which has no attribute of `cat`")
            new_data[k]=new_values
        return new_data
        
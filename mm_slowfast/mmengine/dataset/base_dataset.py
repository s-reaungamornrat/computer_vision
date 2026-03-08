from __future__ import annotations
from typing import Optional, Union, Sequence, Callable

import os
import gc
import copy
import pickle
import warnings

import numpy as np

from torch.utils.data import Dataset

from mmengine import Config

class Compose:
    """Compose multiple transform sequentially
    Args:
        transforms (Sequence[Callable], optional): Sequence of transform objects to be composed
    """
    def __init__(self, transforms:Optional[Sequence[Callable]]):
        
        self.transforms:list[Callable]=[]
        
        if transforms is None: transforms=[]
        
        for transform in transforms:
            assert callable(transform), f"Transform must be callable but got {type(transform)}"
            self.transforms.append(transform)
            
    def __call__(self, data:dict)->Optional[dict]:
        """Call function to apply transforms sequentially
        Args:
            data (dict): A result dict contains the data to be transformed
        Returns:
            (dict): The transformed data
        """
        for t in self.transforms:
            data=t(data)
            # The transform will return None when it failed to load images or cannot find suitable augmentation parameters to augment the data
            # Here we simply return None if the transform returns None and the dataset will handle it by randomly selecting another data sample
            if data is None: return None
        return data

    def __repr__(self)->str:
        """Print `self.transforms` in sequence"""
        format_string=self._class__.__name__+"("
        for t in self.transforms:
            format_string+='\n'
            format_string+=f'    {t}'
        format_string+="\n)"
        return format_string

class BaseDataset(Dataset):
    """BaseDataset
    Args:
        ann_file (str, optional): Annotation file path. Default to ''
        metainfo (dict|Config, optional): Meta information for dataset such as class information. Default to None
        data_root (str, optional): The root directory for `data_prefix` and `ann_file`. Default to ''
        data_prefix (dict): Prefix for training data. Default to dict(img_path='')
        filter_cfg (dict, optional): Config for filter data. Default to None.
        indices (int|Sequence[int], optional): Support using first few data in annotation file to faciliatet training/testing on a small set
        serialize_data (bool, optional): Whether to hold memory using serialized objects, when enabled, data loader workers can use shared RAM from master 
            process instead of making a copy. Default to True
        pipeline (list[Callable], optional): List of data processing module, i.e., transforms. Default to []
        test_mode (bool, optional): Whether to run in test mode phase. Default to None
        lazy_init (bool, optional): Whether to load annotation during instatiation. In some cases, such as visualization, only the meta information of the 
            dataset is needed, which is not necessary to load annotation file. `Basedataset` can skip load annotations to save time by set `lazy_init=True`.
            Default to False
        max_refetch (int, optional): If `Basedataset.prepare_data` get a None image. The maximum extra number of cycles to get a valid image. 
            Default to 1000
    Note:
        BaseDataset collects meta information from `annotation_file` (lowest priority), BaseDataset.METAINFO (medium) and `metainfo` (highest) passed to 
        constructors. The lower priority meta information will be overwritten by higher one.
    Reference:
        https://github.com/open-mmlab/mmengine/blob/main/mmengine/dataset/base_dataset.py#L277
    """
    METAINFO:dict=dict()
    _fully_initialized:bool=False
    def __init__(self, ann_file:Optional[str]='', metainfo:Union[dict, Config,None]=None, data_root:Optional[str]='', data_prefix:dict=dict(img_path=''),
                 filter_cfg:Optional[dict]=None, indices:Optional[Union[int, Sequence[int]]]=None, serialize_data:bool=True, 
                 pipeline:list[Union[dict, Callable]]=[], test_mode:bool=False, lazy_init:bool=False, max_refetch:int=1000):
        
        self._metainfo=self._load_metainfo(copy.deepcopy(metainfo))
        self.data_root=data_root
        self.data_prefix=copy.copy(data_prefix)
        self.filter_cfg=copy.deepcopy(filter_cfg)
        self._indices=indices
        self.serialize_data=serialize_data
        self.test_mode=test_mode
        self.max_refetch=max_refetch
        self.data_list:list[dict]=[]
        self.data_bytes:np.ndarray

        if os.path.isdir(self.data_root):
            self.data_prefix={k:os.path.join(self.data_root, v) for k, v in self.data_prefix.items()}
            self.ann_file=os.path.join(self.data_root, ann_file)
            assert all(os.path.exists(v) for v in self.data_prefix.values())
            assert os.path.exists(self.ann_file)
        else: self.ann_file=ann_file

        self.pipeline=Compose(pipeline)

        # Full initialization of data
        if not lazy_init: self.full_init()

    def get_data_info(self, idx:int)->dict:
        """Get annotation by index and automically call `full_init` if the dataset has not been fully initialized
        Args:
            idx (int): The index of data
        Returns:
            (dict): The idx-th annotation of dataset
        """
        if not self._fully_initialized: self.full_init()
        if self.serialize_data:
            start_addr=0 if idx==0 else self.data_address[idx-1].item()
            end_addr=self.data_address[idx].item()
            bytes=memoryview(self.data_bytes[start_addr:end_addr])
            data_info=pickle.loads(bytes)
        else: data_info=copy.deepcopy(self.data_list[idx])
        # Some codebase need `sample_idx` of data information. Here we convert the idx to a positive number and save it in data information
        if idx>=0: data_info['sample_idx']=idx
        else: data_info['sample_idx']=len(self)+idx # negative

        return data_info
        
    def full_init(self):
        """Load annotation file and set `BaseDataset._fully_initialized` to True

        If `lazy_init=False`, `full_init` will be called during the instantiation and `self._fully_initialized` will be set True. 
        If `obj._fully_initialized=False`, the class method decorated by `force_full_init` will call `full_init` automatically

        Several steps to initialize annotation
            - load_data_list: load annotations from annotation file
            - filter data information: filter annotations according to filter_cfg
            - slice_data: slice dataset according to `self._indices`
            - serialize_data: serialize `self.data_list` if `self.serialize_daya` is True
        """
        if self._fully_initialized: return
        # load data information
        self.data_list=self.load_data_list()
        # filter illegal data, such as data that have no annotations
        #self.data_list=self.filter_data() # do not filter
        # Get subset data according to indices
        if self._indices is not None: self.data_list=self._get_unserialized_subset(self._indices)

        # serialize data_list
        if self.serialize_data: self.data_bytes, self.data_address=self._serialize_data()

        self._fully_initialized=True
    
    def _get_unserialized_subset(self, indices:Union[Sequence[int],int])->list:
        """Get subset of data information list
        Args:
            indices (int|Sequence[int]): If type of indices is int, indices represents the first or last few data of data information. If type of 
                indices is Sequence, indices represent the target data information index which consist of subset data information.
        Returns:
            (tuple[np.ndarray, np.ndarray]): Subset of data information
        """
        if isinstance(indices, int): 
            if indices>=0: sub_data_list=self.data_list[:indices] # return the first few data information
            else: sub_data_list=self.data_list[indices:] # return the last few data information
        elif isinstance(indices, Sequence):
            # return the data information according to the given indices
            sub_data_list=[self.data_list[idx] for idx in indices]
        else: raise TypeError(f"Indices should be int or a sequence of int, but got {type(indices)}")
        return sub_data_list

    def _serialize_data(self)->tuple[np.ndarray, np.ndarray]:
        """Serialize `self.data_list` to save memory when launching multiple workers in data loading. This function will be called in `full_init`

        Hold memory using serialized objects, and data loader workers can use shared RAM from master process instead of making a copy
        Returns:
            (tuple[np.ndarray, np.ndarray]): Serialized result and corresponding address
        """
        def _serialize(data):
            buffer=pickle.dumps(data, protocol=4)
            return np.frombuffer(buffer, dtype=np.uint8)
            
        # Serialize data information list avoid making multiple copies of `self.data_list` when iterate `torch.utils.data.dataloader` with multiple workers
        data_list=[_serialize(x) for x in self.data_list]
        address_list=np.asarray([len(x) for x in data_list], dtype=np.int64)
        data_address:np.ndarray=np.cumsum(address_list)
        # TODO check if np.concatenate is necessary
        data_bytes=np.concatenate(data_list)
        # Empty cache for preventing making multiple copies of `self.data_info` when loading data multi-processes
        self.data_list.clear()
        gc.collect()
        return data_bytes, data_address
        
    def load_data_list(self)->list[dict]:
        """Load annotations from an annotation file"""
        raise NotImplementedError('Please implement load_data_list')
        
    @classmethod
    def _load_metainfo(cls, metainfo:Union[dict, Config, None]=None)->dict:
        """Collect meta information from the dict of meta
        Args:
            metainfo (dict|Config, optional): Meta information dict. If `metainfo` contains existed filename, it will be parsed by `list_from_file`
        Returns:
            (dict): Parsed meta information
        """
        cls_metainfo=copy.deepcopy(cls.METAINFO) # avoid `cls.METAINFO` being overwritten by `metainfo`
        if metainfo is None: return cls_metainfo
        assert isinstance(metainfo, (dict, Config)), f"metainfor should be a Mapping or Config, but got {type(metainfo)}"

        for k, v in metainfo.items():
            if isinstance(v, str): # if type of value is str and can be loaded from corresponding backend, It means the file name of meta file
                # try: cls_metainfo[k]=list_from_file(v)
                # except (TypeError, FileNotFoundError):
                #     warnings.warn(f"{v} is not a meta file, simply parsed as meta information")
                #    cls_metainfo[k]=v
                cls_metainfo[k]=v
            else: cls_metainfo[k]=v
        return cls_metainfo

    @property
    def metainfo(self)->dict:
        """Get meta information of dataset
        Returns:
            (dict): meta information collected from `BaseDataset.METAINFO` annotation file and metainfo argument during instantiation
        """
        return copy.deepcopy(self._metainfo)
        
    def __getitem__(self, idx:int)->dict:
        """Get the idx-th image and data information of dataset after `self.pipeline`, and `full_init` will ve called if the dataset has not been fully 
        initialized. 
        During training phase, if `self.pipeline` return `None`, `self._rand_another` will be called until a valid image is fetched or the maximum limit 
        of refetched is reached
        Args:
            idx (int): The index of self.data_list
        Returns:
            (dict): The idx-th image and data information of dataset after `self.pipeline`
        """
        # Performing full initialization by calling `__getitem__` will consume extra memory. If a dataset is not fully initialized by setting 
        # `lazy_init=True` and then fed into the dataloader. Different workers will simultaneously read and parse the annotation. It will cost 
        # more time and memory, althought this may work. Therefore, it is recommended to manually call `full_init` before dataset fed into 
        # dataloader to ensure all workers used shared RAM from master process
        if not self._fully_initialized:
            warnings.warn("Please call `full_init()` manually to accelerate the dataloader process")
            self.full_init()
        if self.test_mode:
            data=self.prepare_data(idx)
            if data is None: raise Exception('Test time pipeline should not get `None` data sample')
            return data

        for _ in range(self.max_refetch+1):
            data=self.prepare_data(idx)
            # Broken images or random augmentation may cause the returned data to be None
            if data is None: 
                idx=self._rand_another()
                continue
            return data
            
    def __len__(self)->int:
        """Get the length of dataset and automatically call `full_init` if the dataset has not been fully initialized
        Returns:
            (int): The length of dataset
        """
        if not self._fully_initialized: self.full_init()
        if self.serialize_data: return len(self.data_address)
        return len(self.data_list)
        
    def _rand_another(self)->int:
        """Get random index
        Returns:
            (int): Random index from 0 to `len(self)-1`
        """
        return np.random.randint(0, len(self))
        
    def prepare_data(self, idx)->Any:
        """Get data processed by `self.pipeline`
        Args:
            idx (int): The index of `data_info`
        Returns:
            (Any): Depends on `self.pipeline`
        """
        data_info=self.get_data_info(idx)
        return self.pipeline(data_info)

    # def parse_data_info(self, raw_data_info:dict)->Union[dict, list[dict]]:
    #     """Parse raw annotation to target format

    #     This method should return dict or list of dict. Each dict or list contains the data information of a training sample. If the protocol of the 
    #     sample annotations is changed, this function can be overriden to update the parsing logic while keeping compatibility
    #     Args:
    #         raw_data_info (dict): Raw data info loaded from `ann_file`
    #     Returns:
    #         (dict | list[dict]): Parsed annotation
    #     """
    #     for prefix_key, prefix in self.data_prefix.items():
    #         assert prefix_key in raw_data_info, (f"raw_data_info: {raw_data_info} does not contain prefix key {prefix_key}, please check your data_prefix")
    #         raw_data_info[prefix_key]=os.path.join(prefix, raw_data_info[prefix_key])
    #     return raw_data_info


    # def _copy_without_annotation(self, memo=dict())->'BaseDataset':
    #     """Deepcopy of all attributes other than `data_list`, `data_address`, and `data_bytes`
    #     Args:
    #         memo (dict): Memory dict which used to reconstruct complex object correctly
    #     """
    #     cls=self.__class__
    #     other=cls.__new__(cls)
    #     memo[id(self)]=other
        
    #     for key, value in self.__dict__.items():
    #         if key in ['data_list', 'data_address', 'data_bytes']: continue
    #         super(BaseDataset, other).__setattr__(key, copy.deepcopy(value, memo))
    #     return other

    # def get_subset_(self, indices: Union[Sequence[int], int]) -> None:
    #     """The in-place version of `get_subset` to convert dataset to a subset of original dataset
        
    #      This method will convert the original dataset to a subset of dataset. If type of indices is int, `get_subset_` will return a subdataset
    #      which contains the first or last few data information according to whether indices is positive or negative. If type of indices is a sequence
    #      of int, the subdataset will extract the data information according to the index given in indices.
        
    #     Args:
    #         indices (int|Sequence[int]): If type of indices is int, indices represent the first or last few data of the dataset according to whether
    #             the indices is positive or negative. If type of indices is sequence, indices represent the target data information index of dataset.
    #     """
    #     if not self._fully_initialized: self.full_init()
    #     # Get subset of data from serialized data or data information sequence according to `self.serialize_data`
    #     if self.serialize_data: self.data_byte, self.data_address=self._get_serialized_subset(indices)
    #     else: self.data_list=self._get_unserialized_subset(indices)

    # def get_subset(self, indices:Union[Sequence[int], int]) -> 'BaseDataset':
    #     """Return a subset of dataset"""
    #     # Get subset of data from serialized data or data information list according to `self.serialize_data`. Since `_get_serialized_subset` will 
    #     # recalculate the subset data information, `_copy_without_annotation` will copy all attributes except data information.
    #     sub_dataset=self._copy_without_annotation()
    #     # Get subset of dataset with serialize and unserialized data
    #     if self.serialize_data:
    #         data_bytes, data_address=self._get_serialized_subset(indices)
    #         sub_dataset.data_byte=data_byte.copy()
    #         sub_dataset.data_address=data_address.copy()
    #     else:
    #         data_list=self._get_unserialized_subset(indices)
    #         sub_dataset.data_list=copy.deepcopy(data_list)
    #     return sub_dataset

    # def _get_serialized_subset(self, indices:Union[Sequence[int], int])->tuple[np.ndarray, np.ndarray]:
    #     """Get subset of serialized data information list"""
    #     sub_data_bytes:Union[list, np.ndarray]
    #     sub_data_address:Union[list, np.ndarray]
    #     # here indices represent the count/number of elements in sub_data_bytes and sub_data_address. It does not represent indices
    #     if isinstance(indices, int): 
    #         if indices>=0: 
    #             assert indices <= len(self.data_address), f"{indices} is out of dataset length {len(self)}"
    #             # Return the first few data information
    #             end_addr=self.data_address[indices-1].item() if indices>0 else 0
    #             # Slicing operatioon of `np.ndarray` does not trigger a memory copy
    #             sub_data_bytes=self.data_bytes[:end_addr]
    #             # Since the buffer size of first few data information is not changed
    #             sub_data_address=self.data_address[:indices]
    #         else:
    #             assert -indices<=len(self.data_address), f"{indices} is out of dataset length {len(self)}"
    #             # Return the last few data information
    #             ignored_bytes_size=self.data_address[indices-1]
    #             start_addr=self.data_address[indices-1].item()
    #             sub_data_bytes=self.data_bytes[start_addr:]
    #             sub_data_address=self.data_address[indices:]
    #             sub_data_address=sub_data_address-ignored_bytes_size
    #     elif isinstance(indices, Sequence):
    #         sub_data_bytes, sub_data_address=[],[]
    #         for idx in indices:
    #             assert len(self)>idx>=-len(self)
    #             start_addr=0 if idx==0 else self.data_address[idx-1].item()
    #             end_addr=self.data_address[idx].item()
    #             # Get data information by address
    #             sub_data_bytes.append(self.data_bytes[start_addr:end_addr])
    #             # Get data information size
    #             sub_data_address.append(end_addr-start_addr)
    #         # Handle indices 
    #         if sub_data_bytes:
    #             sub_data_bytes=np.concatenate(sub_data_bytes)
    #             sub_data_address=np.cumsum(sub_data_address)
    #         else: sub_data_bytes, sub_data_address=np.array([]), np.array([])
    #     else: raise TypeError(f'Indices should be a int or sequence of int, but got {type(indices)}')

    #     return sub_data_bytes, sub_data_address

    # def _get_unserialized_subset(self, indices:Union[Sequence[int], int])->list:
    #     """Get subset of data information list
    #     Args:
    #         indices (int | Sequence[int]): If type of indices is int, indices represents the frst or last few data of data information. If type of indices
    #             is Sequence, indices represents the target data information index which consist of subset data information
    #     Returns:
    #         (tuple[np.ndarray, np.ndarray]): Subset of data information
    #     """
    #     if isinstance(indices, int):
    #         if indices>=0: sub_data_list=self.data_list[:indices] # Return the first few data information
    #         else: sub_data_list=self.data_list[indices:] # Return the last few data information
    #     elif isinstance(indices, Sequence):
    #         # Return the data information according to given indices
    #         sub_data_list=[self.data_list[idx] for idx in indices]
    #     else: raise TypeError(f"Indices should be an int or a sequence of int, but got {type(indices)}")

    #     return sub_data_list
        
        
         
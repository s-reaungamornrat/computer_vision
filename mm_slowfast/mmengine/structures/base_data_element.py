from __future__ import annotations
from typing import Any, Iterator, Optional, Union, Type

import copy
import numpy as np

import torch

class BaseDataElement:
    """A base data interface that supports Tensor-like and dict-like operations
    
    A typical data elements refer to predicted results or ground truth labels on a task, such as predicted bboxes, instance masks, semantic
    segmentation masks, etc. Because ground truth labels and predicted results often have similar properties (for example, the predicted bboxes
    and the ground truth bboxes), MMEngine uses the same abstarct data interface to encapsulate predicted results and ground truth labels, and it 
    is recommended to use different name conventions to distinguish them, such as using `gt_instances` and `pred_instances` to distinguish between 
    labels and predicted results. Additionally, we distinguish data elements at instance level, pixel level, and label level. Each of these types has
    its own characteristics. Therefore, MMEngine defines the base class `BaseDataElement` and implement `InstanceData`, `PixelData`, and `LabelData`
    inheriting from `BaseDataElement` to represent different types of ground truth labels or predictions.

    Another common data element is sample data. A sample data consists of input data (such as an image) and its annotations and predictions. In general,
    an image can have multiple types of annotations and/or predictions at the same time (for example, both pixel-level semantic segmentation annotations
    and instance-level detection bboxes annotations). All labels and predictions of a training sample are often passed between Dataset, Model, Visualizer,
    and Evaluator components. In oder to simplify the interface between components, we can treate them as a large data element and encapsulate them. Such
    data elements are generally called XXDataSample in OpenMMLab. Therefore, similar to `nn.Module`, the `BaseDataElement` allows `BaseDataElement` as 
    its attribute. Such a class generally encapsulates all the data of a sample in the algorithm library, and its attributes generally are various types of 
    data elements. For example, MMDetection is assigned by the BaseDataElement to encapsulate all the data elements of the sample labeling and prediction of 
    a sample in the algorithm library.

    The attributes in `BaseDataElement` are divided into two parts: `metainfo` and `data` respectively.

        - `metainfo`: contains the information about the image such as filena,e, image_shape, pad_shape, etc. The attributes can be accessed or modified 
           by dict-like or object-like operations, such as `.`, `in`, `del`, `pop(str)`, `get(str)`, `metainfo_keys()`, metainfo_values()`, metainfo_items()`
           and `set_metainfo()` (for set or change key-value pairs in metainfo)
        - `data`: stores annotations or model predictions. The attributes can be accessed or modified by dict-like or object-like operations, such as `.`,
           `in`, `del`, `pop(str)`, `get(str)`, `keys()`, `values()`, `items()`. Users can also apply tensor-like methods to all to `obj:torch.Tensor` in the
           `data_field`, such as `.cuda()`, `.cpu()`, `.numpy()`, `.to()`, `.to_tensor()`, `.detach()`
           
    Args:
        metainfo (dict, optional): A dict containing the meta information of single image, such as `dict(img_shape=(512,512,3), scale_factor=(1,1,1,1))`
            Default to None
        kwargs (dict, optional): A dict contains annotations of single image or model prediction. Default to None.
    Examples:
        >>> gt_instances=BaseDataElement()
        >>> bboxes=torch.rand(5,4)
        >>> scores=torch.rand(5)
        >>> img_id=0
        >>> img_shape=(800, 1333)
        >>> gt_instances=BaseDataElement(metainfo=dict(img_id=img_id, img_shape=img_shape), bboxes=bboxes, scores=scores)
        >>> gt_instances=BaseDataElement(metainfo=dict(img_id=img_id, img_shape=(640,640)))
        >>> gt_instances1=gt_instances.new(metainfo=dict(img_id=img_id, img_shape=(640,640)), bboxes=torch.rand(5,4), scores=torch.rand(5))
        >>> gt_instances2=gt_instances1.new()
        >>> # add and process property
        >>> gt_instances=BaseDataElement()
        >>> gt_instances.set_metainfo(dict(img_id=9, img_shape=(100,100)))
        >>> assert 'img_shape' in gt_instances.metainfo_keys()
        >>> assert 'img_shape' in gt_instances
        >>> assert 'img_shape' in gt_instances.all_keys()
        >>> print(gt_instances.img_shape)
        (100,100)
        >>> gt_instances.scores=torch.rand(5,)
        >>> assert 'scores' in gt_instances.keys()
        >>> assert 'scores' in gt_instances
        >>> assert 'scores' in gt_instances.all_keys()
        >>> assert 'scores' not in gt_instances.metainfo_keys()
        >>> print(gt_instances.scores)
        tensor([0.5230, 0.7885, 0.2426, 0.3911, 0.4876])
    Reference: https://github.com/open-mmlab/mmengine/blob/main/mmengine/structures/base_data_element.py
    """
    def __init__(self, *, metainfo:Optional[dict]=None, **kwargs)->None: # All parameters after `*` must be passed as keyword arguments.
        self._metainfo_fields:set=set() # immutable (cannot be changed or removed) datastructure used to keep track of its own attributes 
        self._data_fields:set=set() # immutable (cannot be changed or removed) datastructure used to keep track of its own attributes

        if metainfo is not None: self.set_metainfo(metainfo=metainfo)
        if kwargs: self.set_data(kwargs)

    def set_metainfo(self, metainfo:dict)->None:
        """Set or change ky-value pairs in `metainfo_field` by parameter `metainfo`
        Args:
            metainfo (dict): A dict contains the meta information of image, such as `img_shape`, `scale_factor` etc
        """
        assert isinstance(metainfo, dict), f"metainfo should be a dict, but got {type(metainfo)}"
        meta=copy.deepcopy(metainfo)
        for k, v in meta.items(): self.set_field(name=k, value=v, field_type='metainfo', dtype=None)

    def set_data(self, data:dict)->None:
        """Set or change key-value pairs in `data_field` by parameter `data`
        Args:
            data (dict): A dict contains annotations of image or model prediction
        """
        assert isinstance(data, dict), f"data should be a `dict` but got {type(data)}"
        for k, v in data.items(): 
            # Use `setattr()` rather than `self.set_field` to allow `set_data` to set property method
            setattr(self, k, v)
            
    def update(self, instance:'BaseDataElement')->None:
        """The method updates the BaseDataElement with the elements from another BaseDataElement object
        Args:
            instance (BaseDataElement): Another BaseDataElement object for update the current object
        """
        assert isinstance(instance, BaseDataElement), f"instance should be a `BaseDataElement` but got {type(instance)}"
        self.set_metainfo(dict(instance.metainfo_items()))
        self.set_data(dict(instances.items()))

    def new(self, *, metainfo:Optional[dict]=None, **kwargs)->"BaseDataElement":
        """Return a new data element with same type. If `metainfo` and `data` are None, the new data element will have the same metainfo and data.
        If metainfo and data are not None, the new result will overwrite it with the input value
        Args:
            metainfo (dict, optional): A dict containing the meta information of image such as `img_shape`, `scale_factor`, etc. Default to None
            kwargs (dict, optional): A dict containing annotations of image or model predictions
        Returns:
            (BaseDataElement): A new data element with the same type
        """
        new_data=self.__class__()
        if metainfo is not None: new_data.set_metainfo(metainfo)
        else: new_data.set_metainfo(dict(self.metainfo_items()))
        if kwargs: new_data.set_data(kwargs)
        else: new_data.set_data(dict(self.items()))
        return new_data

    def clone(self):
        """Deep copy the current data element
        Returns:
            (BaseDataElement): The copy of current data element
        """
        clone_data=self.__class__()
        clone_data.set_metainfo(dict(self.metainfo_items()))
        clone_data.set_data(dict(self.items()))
        return clone_data

    def keys(self)->list:
        """
        Returns:
            (list): All keys in the data_fields
        """
        # we assume that the name of the attribute related to property is '_' + the name of the property. We use this rule to filter out private 
        # keys. 
        # TODO: use more robust way to solve this problem
        private_keys={'_'+key for key in self._data_fields if isinstance(getattr(type(self), key, None), property)}
        return list(self._data_fields-private_keys)

    def metainfo_keys(self)->list:
        """
        Returns:
            (list): All keys in metainfo_fields
        """
        return list(self._metainfo_fields)

    def values(self)->list:
        """
        Returns:
            (list): All values in data
        """
        return [getattr(self, k) for k in self.keys()] 

    def metainfo_values(self)->list:
        """
        Returns:
            (list): All values in metainfo
        """
        return [getattr(self, k) for k in self.metainfo_keys()]

    def all_keys(self)->list:
        """
        Returns:
            (list): All keys in metainfo and data
        """
        return self.metainfo_keys()+self.keys()

    def all_values(self)->list:
        """
        Returns:
            (list): All values in metainfo and data
        """
        return self.metainfo_values()+self.values()
        
    def all_items(self)->Iterator[tuple[str, Any]]:
        """
        Returns:
            (Iterator): An iterator object whose element is (key, value) tuple pairs for `metainfo` and `data`
        """
        for k in self.all_keys(): yield (k, getattr(self, k))

    def items(self)->Iterator[tuple[str, Any]]:
        """
        Returns:
            (Iterator): An iterator object whose element is (key, value) tuple pairs for `data`
        """
        for k in self.keys(): yield (k, getattr(self, k))

    def metainfo_items(self)->Iterator[tuple[str, Any]]:
        """
        Returns:
            (Iterator): An iterator object whose element is (key, value) tuple pairs for `metainfo`
        """
        for k in self.metainfo_keys(): yield (k, getattr(self, k))
            
    @property
    def metainfo(self)->dict:
        """dict: A dict contains metainfo of current data element"""
        return dict(self.metainfo_items())

    def __setattr__(self, name:str, value:Any):
        """Set data"""
        if name in ('_metainfo_fields', '_data_fields'): 
            if not hasattr(self, name): super().__setattr__(name, value)
            else: raise AttributeError(f"{name} has been used as a private attribute, which is immutable")
        else:
            self.set_field(name=name, value=value, field_type='data', dtype=None)

    def __delattr__(self, item:str):
        """Delete the item in data element
        Args:
            item (str): The key to delete
        """
        if item in ('_metainfo_fields', '_data_fields'): raise AttributeError(f"{item} has been used as a private attribute, which is immutable")
        super().__delattr__(item)
        if item in self._metainfo_fields: self._metainfo_fields.remove(item)
        elif item in self._data_fields: self._data_fields.remove(item)

    # dict-like methods
    __delitem__=__delattr__

    def get(self, key, default=None)->Any:
        """Get property in data and metainfo as the same as python"""
        # use `getattr()` rather than `self.__dict__.get()` to allow getting properties
        return getattr(self, key, default)

    def pop(self, *args)->Any:
        """Pop property in data and metainfo as the same as python"""
        assert len(args)<3, '`pop` get more than 2 arguments'
        name=args[0]
        if name in self._metainfo_fields:
            self._metainfo_fields.remove(args[0])
            return self.__dict__.pop(*args)
        elif name in self._data_fields:
            self._data_fields.remove(args[0])
            return self.__dict__.pop(*args)
        elif len(args)==2: # with default value
            return args[1]
        else: raise KeyError(f"{args[0]} is not contained in metainfo or data")

    def __contains__(self, item:str)->bool:
        """Whether the item is in data element
        Args:
            item (str): The key to inquire
        """
        return item in self._data_fields or item in self._metainfo_fields

    def set_field(self, value:Any, name:str, dtype:Optional[Union[Type, tuple[Type,...]]]=None, field_type:str='data')->None:
        """Special method for set union field, used as property.setter functions"""
        assert field_type in ['metainfo', 'data']
        if dtype is not None: assert isinstance(value, dtype), f"{value} should be a {dtype} but got {type(value)}"
        if field_type=='metainfo':
            if name in self._data_fields: raise AttributeError(f"Cannot set {name} to be a field of metainfo because {name} is already a data field")
            self._metainfo_fields.add(name)
        else:
            if name in self._metainfo_fields: raise AttributeError(f"Cannot set {name} to be a field of data because {name} is already a metainfo field")
            self._data_fields.add(name)
        super().__setattr__(name, value)

    # Tensor-like method
    def to(self, *args, **kwargs)->"BaseDataElement":
        """Apply same name function to all tensors in data_fields"""
        new_data=self.new()
        for k, v in self.items():
            if hasattr(v, 'to'):
                v=v.to(*args, **kwargs)
                data={k:v}
                new_data.set_data(data)
        return new_data

    # Tensor-like method
    def cpu(self)->"BaseDataElement":
        """Convert all tensors to CPU in data"""
        new_data=self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v=v.cpu()
                data={k:v}
                new_data.set_data(data)
        return new_data

    # Tensor-like method
    def cuda(self)->"BaseDataElement":
        """Convert all tensors to GPU in data"""
        new_data=self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v=v.cuda()
                data={k:v}
                new_data.set_data(data)
        return new_data

    # Tensor-like method
    def musa(self)->"BaseDataElement":
        """Convert all data to musa in data"""
        new_data=self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v=v.musa()
                data={k:v}
                new_data.set_data(data)
        return new_data

    # Tensor-like methods
    def npu(self)->"BaseDataElement":
        """Convert all tensors to NPU in data"""
        new_data=self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v=v.npu()
                data={k:v}
                new_data.set_data(data)
        return new_data

    def mlu(self)->"BaseDataElement":
        """Convert all tensors to MLU in data"""
        new_data=self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v=v.mlu()
                data={k:v}
                new_data.set_data(data)
        return new_data

    # Tensor-like methods
    def detach(self)->"BaseDataElement":
        """Detach all tensors in data"""
        new_data=self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v=v.detach()
                data={k:v}
                new_data.set_data(data)
        return new_data

    # Tensor-like method
    def numpy(self)->"BaseDataElement":
        new_data=self.new()
        for k, v in self.items():
            if isinstance(v,(torch.Tensor, BaseDataElement)):
                v=v.detach().cpu().numpy()
                data={k:v}
                new_data.set_data(data)
        return new_data

    def to_tensor(self)->"BaseDataElement":
        """Convert all np.ndarray to tensor in data"""
        new_data=self.new()
        for k, v in self.items():
            data={}
            if isinstance(v, np.ndarray):
                v=torch.from_numpy(v)
                data[k]=v
            elif isinstance(v, BaseDataElement):
                v=v.to_tensor()
                data[k]=v
            new_data.set_data(data)
        return new_data

    def to_dict(self)->dict:
        """Convert BaseDataElement to dict"""
        return {k:v.to_dict() if isinstance(v, BaseDataElement) else v for k, v in self.all_items()}

    def __repr__(self)->str:
        """Represent the object"""
        def _addindent(s_:str, num_spaces:int)->str:
            """This function is modified from `pytorch` https://github.com/pytorch/
            pytorch/blob/b17b2b1cc7b017c3daaeff8cc7ec0f514d42ec37/torch/nn/modu
            les/module.py#L29. It adds `num_spaces` spaces in front of each line, except the first line.
            Args:
                s_ (str): The string to add spaces
                num_spaces (int): The number of spaces to add
            Returns:
                (str): The string after adding indent
            """
            s=s_.split('\n')
            # do not do anything foro single-line stuff
            if len(s)==1: return s_
            first=s.pop(0)
            s=[(num_spaces*" ")+line for line in s] # add `num_spaces` spaces in front of each line
            s='\n'.join(s)
            s=first+"\n"+s
            return s
            
        def dump(obj:Any)->str:
            """Represent the object
            Args:
                obj (Any): The obj to represent
            Returns:
                (str): The represented string
            """
            _repr=''
            if isinstance(obj, dict):
                for k, v in obj.items(): _repr+=f"\n{k}:{_addindent(dump(v), 4)}"
            elif isinstance(obj, BaseDataElement):
                _repr+="\n\n    META INFORMATION"
                metainfo_items=dict(obj.metainfo_items())
                _repr+=_addindent(dump(metainfo_items), 4)
                _repr+="\n\n    DATA FIELDS"
                items=dict(obj.items())
                _repr+=_addindent(dump(items), 4)
                classname=obj.__class__.__name__
                _repr=f"<{classname}({_repr}\n) at {hex(id(obj))}>"
            else: _repr+=repr(obj)

            return _repr
        # def _addindent(s_: str, num_spaces: int) -> str:
        #     """This func is modified from `pytorch` https://github.com/pytorch/
        #     pytorch/blob/b17b2b1cc7b017c3daaeff8cc7ec0f514d42ec37/torch/nn/modu
        #     les/module.py#L29.

        #     Args:
        #         s_ (str): The string to add spaces.
        #         num_spaces (int): The num of space to add.

        #     Returns:
        #         str: The string after add indent.
        #     """
        #     s = s_.split('\n')
        #     # don't do anything for single-line stuff
        #     if len(s) == 1:
        #         return s_
        #     first = s.pop(0)
        #     s = [(num_spaces * ' ') + line for line in s]
        #     s = '\n'.join(s)  # type: ignore
        #     s = first + '\n' + s  # type: ignore
        #     return s  # type: ignore
            
        # def dump(obj: Any) -> str:
        #     """Represent the object.

        #     Args:
        #         obj (Any): The obj to represent.

        #     Returns:
        #         str: The represented str.
        #     """
        #     _repr = ''
        #     if isinstance(obj, dict):
        #         for k, v in obj.items():
        #             _repr += f'\n{k}: {_addindent(dump(v), 4)}'
        #     elif isinstance(obj, BaseDataElement):
        #         _repr += '\n\n    META INFORMATION'
        #         metainfo_items = dict(obj.metainfo_items())
        #         _repr += _addindent(dump(metainfo_items), 4)
        #         _repr += '\n\n    DATA FIELDS'
        #         items = dict(obj.items())
        #         _repr += _addindent(dump(items), 4)
        #         classname = obj.__class__.__name__
        #         _repr = f'<{classname}({_repr}\n) at {hex(id(obj))}>'
        #     else:
        #         _repr += repr(obj)
        #     return _repr

        return dump(self)
        
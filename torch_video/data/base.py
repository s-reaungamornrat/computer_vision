import os
from pathlib import Path
from typing import Any, Callable, Optional, Union, cast

import torch.utils.data as data

class VisionDataset(data.Dataset):
    """Base class for making datasets which are compatible with torchvision.

    Args:
        root (str|Path, optional): Root directory of dataset
        transforms (callable, optional): A function/transforms that takes in an image and annotation and returns the transformed versions of both
    Reference:
        https://github.com/pytorch/vision/blob/main/torchvision/datasets/vision.py
    """
    def __init__(self, root:Union[str,Path]=None, transforms:Optional[Callable]=None)->None:
        
        if isinstance(root, str): root=os.path.expanduser(root)
            
        self.root=root
        self.transforms=transforms

    def __getitem__(self, index:int)->Any:
        """
        Args:
            index (int): Index
        Returns:
            (Any): Sample and meta data, optionally transformed by the respective transforms
        """
        raise NotImplementedError

    def __len__(self)->int: raise NotImplementedError

    def __repr__(self)->str:
        head="Dataset"+self.__class__.__name__
        body=[f"Number of datapoints: {self.__len__()}"]
        if self.root is not None: body.append(f'Root location: {self.root}')
        body+=self.extra_repr().splitlines()
        if hasattr(self, 'transforms') and self.transforms is not None: body+=[repr(self.transforms)]
        lines=[head]+[" "*self._repr_indent+line for line in body]
        return '\n'.join(lines)

    def _format_transform_repr(self,transform:Callable, head:str)->list[str]:
        lines=transform.__repr__().splitlines()
        return [f'{head}{lines[0]}']+['{}{}'.format(" "*len(head), line) for line in lines[1:]]
        
    def extra_repr(self)->str: return ""
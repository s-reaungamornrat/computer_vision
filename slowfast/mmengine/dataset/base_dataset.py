from __future__ import annotations
from typing import Optional, Union, Sequence, Callable

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
        
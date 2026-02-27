from __future__ import annotations
from typing import Union

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

def stack_batch(tensor_list:list[torch.Tensor], pad_size_divisor:int=1, pad_value:Union[int, float]=114)->torch.Tensor:
    """Stack multiple tensors to form a batch and pad the tensor to the max shape use the right bottom padding mode. If `pad_size_divisor>0`, add padding
    to ensure the shape of each dim is divisible by `pad_size_divisor`

    Note: this function stacks tensors along the batch dimension (i.e., dim=0)
    Args:
        tensor_list (list[torch.Tensor]): A list of tensors with the same dimension, e.g.,[(C1,T1,H1,W1),(C2,T2,H2,W2),...,(Cn,Tn,Hn,Wn)] or 
            [(C1,H1,W1),(C2,H2,W2),...,(Cn,Hn,Wn)]
        pad_size_divisor (int): If `pad_size_divisor>0`, add padding to ensure the shape of each dimension is divisible by `pad_size_divisor`. This depends
            on the model, and many models need to be divisible by 32. Default to 1
        pad_value (int, float): The padding value. Default to 0
    Returns:
        (torch.Tensor): 
    """
    assert isinstance(tensor_list, list), f"Expected input type to be list, but got {type(tensor_list)}"
    assert tensor_list, "`tensor_list` could not be an empty list"
    assert len({tensor.ndim for tensor in tensor_list})==1, ("Expected the dimension of all tensors to be the same, but got "
                                                              f"{[tensor.ndim for tensor in tensor_list]}")
    dim=tensor_list[0].ndim
    num_img=len(tensor_list)
    
    # 2d matrix each row represent size of each tensor, e.g., 
    # [[C1,T1,H1,W1],
    # ...
    # [Cn,Tn,H1,W1]]
    all_sizes=torch.Tensor([tensor.shape for tensor in tensor_list]) # of type float32 by default
    # max size along each dimension, i.e., [Cmax, Tmax, Hmax, Wmax]
    max_sizes=torch.ceil(torch.max(all_sizes, dim=0).values/pad_size_divisor)*pad_size_divisor
    
    # padding size for C,T,H,W. In other words, padC, padT, padH, padW
    padded_sizes=max_sizes-all_sizes # 2d matrix with padding size for each tensor along each dimension
    # The first dimension normally means channel, which should not be padded
    padded_sizes[:,0]=0
    
    if padded_sizes.sum()==0: return torch.stack(tensor_list)

    # The order `pad` the second argument of `F.pad` is the reverse of `padded_sizes` 
    pad=torch.zeros(num_img, 2*dim, dtype=torch.int)

    # pad is [0, padW, 0, padH, 0, padT, 0, 0] where the last 2 for channel.
    pad[:,1::2]=padded_sizes[:, range(dim-1,-1,-1)] 
    batch_tensor=[]
    for idx, tensor in enumerate(tensor_list):
        batch_tensor.append(
            F.pad(tensor, tuple(pad[idx].tolist()), value=pad_value)
        )
    return torch.stack(batch_tensor)

def merge_dict(*args):
    """Merge all dicts into one dict
    Args:
        *args (list[dict]): List of dicts needs to be merged
    Returns:
        (dict): Merged dict from args
    """
    output=dict()
    for item in args:
        assert isinstance(item, dict), f"all arguments of merge_dict should be a dict, but got {type(item)}"
        output.update(item)
    return item
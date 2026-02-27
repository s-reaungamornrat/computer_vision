from __future__ import annotations
from typing import Optional, Mapping, Sequence, Any

def pseudo_collate(data_batch:Sequence)->Any:
    """Convert list of data sampled from dataset into a batch of data, of which type consistent with the type of each data_element in `data_batch`
    
    The default behavior of dataloader is to merge a list of samples to form a mini-batch of Tensor(s). However, this function will not stack tensors 
    to batch tensors, and convert int, float, ndarray to tensors.
    
    The code is referenced from
    `Pytorch default_collate <https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py>`_.
    
    Args:
        data_batch (Sequence): Batch of data from dataloader
    Returns:
        (Any): Transversed data in the same format as the data_element of `data_batch`
    """
    data_item=data_batch[0]
    data_item_type=type(data_item)
    
    if isinstance(data_item, (str, bytes)): return data_batch
    elif isinstance(data_item, tuple) and hasattr(data_item, '_fields'): 
        return data_item_type(*(pseudo_collate(samples) for samples in zip(*data_batch)))
    elif isinstance(data_item, Sequence):
        # check to make sure that data_item in batch have consistent size
        it=iter(data_batch)
        data_item_size=len(next(it))
        assert all(len(data_item)==data_item_size for data_item in it), "Each data_item in the list of batch should be of equal size"
        # convert a list of rows into a list of columns, for example from
        # data_batch=[
        #        ('img1', 'cat'),
        #        ('img2', 'dog'),
        #        ('img3', 'bird')
        #] to [('img1', 'img2', 'img3'), ('cat', 'dog', 'bird')]
        transposed=list(zip(*data_batch))
    
        if isinstance(data_item, tuple): return [pseudo_collate(samples) for samples in transposed]
        else:
            try: return data_item_type([pseudo_collate(samples) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)` e.g., range
                return [pseudo_collate(samples) for samples in transposed]
    elif isinstance(data_item, Mapping):
        return data_item_type({key:pseudo_collate([d[key] for d in data_batch]) for key in data_item})
    
    return data_batch
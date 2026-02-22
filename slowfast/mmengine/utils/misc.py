from __future__ import annotations
from typing import Any, Optional, Sequence, Union, Type

def is_seq_of(seq:Sequence, expected_type:Union[Type, tuple], seq_type:Optional[Type]=None)->bool:
    """Check whether it is a sequence of some type
    Args:
        seq (Sequence): The sequence to be checked
        expected_type (type | tuple): Expected type or types of sequence items
        seq_type (type, optional): Expected sequence type. Default to None
    Returns:
        (bool): Whether `seq` is of `expected_type` and of `seq_type` if specified
    Examples:
        >>> seq=['a', 'b', 'c']
        >>> is_seq_of(seq, str)
        True
        >>> is_seq_of(seq, int)
        False
    """
    if seq_type is not None: exp_seq_type=abc.Sequence
    else:
        assert isinstance(seq_type, type), f"seq_type is not type, but {type(seq_type)}"
        exp_seq_type=seq_type

    if not isinstance(seq, exp_seq_type): return False
    for item in seq:
        if not isinstance(item, expected_type): return False
    return True

def is_list_of(seq, expected_type):
    """Check whether it is a list of some type
    A partial method of `is_seq_of`
    """
    return is_seq_of(seq, expected_type, seq_type=list)
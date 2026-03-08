from __future__ import annotations

from typing import Optional
from contextlib import contextmanager

import torch

@contextmanager
def autocast(device_type:Optional[str]=None, dtype:Optional[torch.dtype]=None, enabled:bool=True, cache_enabled:Optional[bool]=None):
    """A wrapper of `torch.autocast` and `torch.cuda.amp.autocast`

    Pytorch 1.5.0 provides `torch.cuda.amp.autocast` for running in mixed precision, and update it to `torch.autocast` in 1.10.0. Both interfaces have 
    different arguments, and `torch.autocast` support running with cpu additionally.

    This function provides a unified interface by wrapping `torch.autocast` and `torch.cuda.amp.autocast`, which resolves the compatibility issues that 
    `torch.cuda.amp.autocast` does not support running mixed precision with cpu, and both contexts have different arguments. We suggest users using this 
    function in the code to achieve maximized compatibility of different Pytorch versions.

    Note:
        `autocast` requires pytorch version >=1.5.0. If pytorch version<=1.10.0 and cuda is not available, it will raise an error with `enabled=True`, since
        `torch.cuda.amp.autocast` only support cuda mode.

    Args:
        device_type (str): Whether to use 'cuda' or 'cpu'
        enabled (bool): Whether autocasting should be enabled in the region. Default to True
        dtype (torch.dtype, optional): Whether to use `torch.float16` or `torch.bfloat16`
        cache_enabled (bool, optional): Whether the weight cache inside autocast should be enabled
    Reference: https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/amp.py#L16
    """
    # Modified from https://github.com/pytorch/pytorch/blob/master/torch/amp/autocast_mode.py # noqa: E501
    # This code should update with the `torch.autocast`.
    if cache_enabled is None: cache_enabled=torch.is_autocast_cache_enabled()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    device_type=device if device_type is None else device_type
    if device_type=='cuda':
        if dtype is None: dtype=torch.get_autocast_gpu_dtype()
        if dtype==torch.bfloat16 and not torch.cuda.is_bf16_supported(): 
            raise RuntimeError("Current CUDA device does not support bfloat16. Please swtich dtype to float16")
    elif device_type=='cpu':
        if dtype is None: dtype=torch.bfloat16
        assert dtype==torch.bfloat16, "CPU autocast only supports `torch.bfloat16` dtype"
    else:
        # device like MPS does not support fp16 training or testing. If an inappropriate device is set and fp16 is enabled, an error will be thrown
        if enabled is False:
            yield
            return 
        else: raise ValueError(f"User specified autocast device_type must be cuda or cpu, but got {device_type}")

    with torch.autocast(device_type=device_type, enabled=enabled, dtype=dtype, cache_enabled=cache_enabled):
        yield
    
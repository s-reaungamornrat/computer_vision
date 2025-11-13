from __future__ import annotations

from pathlib import Path
from typing import Any
from argparse import Namespace
from types import SimpleNamespace

import yaml
from computer_vision.yolov11_pose.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace

def cfg2dict(cfg: str | Path | dict | SimpleNamespace | Namespace)->dict:
    """
    Convert a configuration object to a dict
    Args:
        cfg (str | Path | dict | SimpleNamespace): Configuration object to be converted
    Returns:
        (dict): Configuration object in a dict format
    """
    if isinstance(cfg, (str, Path)): 
        with open(DEFAULT_CFG_PATH) as f: cfg=yaml.load(f, Loader=yaml.SafeLoader)
    elif isinstance(cfg, (SimpleNamespace,Namespace)): cfg=vars(cfg)
    return cfg

def get_cfg(cfg: str|Path|dict|SimpleNamespace=DEFAULT_CFG_DICT,overrides:dict|SimpleNamespace|Namespace|None=None)->SimpleNamespace:
    """
    Load and merge configuration data from a file or dict, with optional overrides
    Args:
        cfg (str | Path | dict | SimpleNamespace): Configuration object 
        overrides (dict|None): Dict containing key-value pairs to override the base configuration
    Returns:
        (SimpleNamespace): Namespace containing the merged configuration arguments
    """
    cfg=cfg2dict(cfg)
    if overrides:
        overrides=cfg2dict(overrides)
        print('overrides ', type(overrides), ' cfg ', type(cfg))
        cfg={**cfg,**overrides} # merge cfg and overrides (preferring overrides)
    return IterableSimpleNamespace(**cfg)
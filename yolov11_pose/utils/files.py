from __future__ import annotations

import os
import glob
from pathlib import Path

def get_latest_run(search_dir:str='.')->str:
    """Return the path to the most recent 'last.pt' file in the specified directory for resuming training"""
    last_list=glob.glob(f'{search_dir}/**/last*pt', recursive=True)
    return max(last_list, key=os.path.getctime) if last_list else ""
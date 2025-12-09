from __future__ import annotations
from typing import Any

import os
import random

import torch
import numpy as np

from computer_vision.yolov11_pose.data.dataset import YOLODataset
from computer_vision.yolov11_pose.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace


def seed_worker(worker_id: int) -> None:
    """Set dataloader worker seed for reproducibility across worker processes."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
def build_yolo_dataset(args:Namespace, cfg:IterableSimpleNamespace, task:str, img_path:str, batch:int, data:dict[str,Any]|str,
                      mode:str='train', rect:bool=False, stride:int=32, channels=3)->torch.utils.data.Dataset:
    """Build and return a YOLO dataset based on configuration parameters
    Args:
        args (Namespace): Additional hyperparameters
        cfg (IterableSimpleNamespace): Hyperparameters
        task (str): Task to perform
        img_path (str): Path to image folder, e.g., data/coco/images/train2017
        batch (int): Batch size 
        data (dict[str,Any]|str): Data configuration file or dict
        mode (str): Data usage mode, including 'train' and 'validation' 
        rect (bool): Whether to use rectangle image pre-processing
        stride (int): Model stride
        channels (int): Number of image color channels
    Returns:
        (torch.utils.data.Dataset): Dataset
    """
    assert task in ['pose','detect','segment'],f'task must be "pose", "detect", or "segment" but got {task}'
    assert mode in ['train', 'validation'], f'mode must be "train" or "validation", but got {mode}'
    return YOLODataset(args=args, data=data, task=task, img_path=img_path, imgsz=cfg.imgsz, 
                       cache=cfg.cache or None, augment=mode=='train', hyp=cfg, prefix='',
                       rect=cfg.rect or rect, batch_size=args.batch_size or hyp.batch, stride=stride,
                       pad=0. if mode=='train' else 0.5, single_cls=cfg.single_cls, classes=cfg.classes,
                       fraction=cfg.fraction if mode=='train' else 1.,channels=channels)
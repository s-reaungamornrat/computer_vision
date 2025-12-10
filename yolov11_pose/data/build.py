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

def build_dataloader(dataset, batch:int, workers:int, shuffle:bool=True, drop_last:bool=False,
                     pin_memory:bool=True)->torch.utils.data.DataLoader:
    """Create and return a DataLoader for training and validation
    Args:
        dataset (torch.utils.data.Dataset): Dataset to load data from
        batch (int): Batch size for the dataloader
        workers (int): Number of worker threads for loading data
        shuffle (bool, optional): Whether to shuffle the dataset
        drop_last (bool, optional): Whether to drop the last incomplete batch
        pin_memory (bool, optional): Whether to use pinned memory for dataloader
    Returns:
        (torch.utils.data.DataLoader): A dataloader that can be used for training and validation
    """
    batch=min(batch, len(dataset))
    nd=torch.cuda.device_count() # number of CUDA devices
    nw=min(os.cpu_count()//max(nd, 1), workers) # number of workers
    return torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=shuffle, num_workers=nw,
                               collate_fn=getattr(dataset, 'collate_fn', None),
                               pin_memory=nd>0 and pin_memory, 
                               drop_last=drop_last and len(dataset)%batch!=0,
                               worker_init_fn=seed_worker, 
                                prefetch_factor=4 if nw>0 else None) # increase prefetch_factor over the default of 2
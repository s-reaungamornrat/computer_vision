from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from computer_vision.yolov11_pose.cfg import get_cfg
from computer_vision.yolov11_pose.utils.checks import check_imgsz
from computer_vision.yolov11_pose.utils.metrics import PoseMetrics

class DectectionValidator:
    """
    This class implements validation functionality specific to objet detection tasks, including metrics calculation, prediection,
    processing, and visualization of results

    Examples:
        >>> args=dict(model='yolo11n.pt', data='coco8.yaml')
        >>> validator=DetectionValidator(args=args)
        >>> validator()
    """
    def __init__(self, dataloader=None, save_dir=None, args=None)->None:
        """Initialize detection validator with necessary variables and settings
        Args:
            dataloader (torch.utils.data.DataLoader, optional): Dataloader to use for validation
            save_dir (Path, optional): Directory to save results
            args (dict[str, Any], optional): Arguments for validator
        """
        self.args=get_cfg(overrides=args)
        self.dataloader=dataloader
        self.stride=None
        self.data=None
        self.device=None
        self.batch_i=None # current batch index
        self.training=True # whether the model is in training mode
        self.names=None # class name mapping
        self.seen=None # number of images seen so far during validation
        self.stats=None # statistics collected during validation
        self.confusion_matrix=None
        self.jdict=None # list to store JSON validation results
        self.speed={'preprocess':0., 'inference':0., 'postprocess':0} # storing respective batch processing time in milliseconds
        self.save_dir=self.args.save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        if self.args.conf is None: self.args.conf=0.01 if self.args.task=='obb' else 0.001 # reduce OBB val memory usage
        self.args.imgsz=check_imgsz(self.args.imgsz, max_dim=1)

        self.plots={}
        self.is_coco=False
        self.is_lvis=False
        self.class_map=None
        self.args.task='detect'
        self.iouv=torch.linspace(0.5, 0.95, 10) # IoU thresholds from .5 to .95 in spaces of .05, i.e.,  mAP@0.5:0.95
        self.niou=self.iouv.numel()
        self.metrics=DetMetrics()
        

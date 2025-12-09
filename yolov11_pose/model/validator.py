from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

from computer_vision.yolov11_pose.utils.metrics import PoseMetrics
from computer_vision.yolov11_pose.engine.validator import DectectionValidator

class PoseValidator(DectectionValidator):
    """A class extending the DetectionValidator class for validation based on a pose model

    This validator is specifically designed for pose estimation tasks, handling keypoints and implementing specialized metrics
    for pose evaluation
    
    Examples:
        >>> args=dict(model='yolo11n-pose.pt', data='coco8-pose.yaml')
        >>> validator=PoseValidator(args=args)
        >>> validator()
    """
    def __init__(self, dataloader=None, save_dir=None, args=None)->None:
        """Initialize a PoseValidator object for pose estimation validation
        
        The validator is specifically designed for pose estimation tasks, handling keypoints and implementing specialized metrics
        for pose evaluation

        Args:
            dataloader(torch.utils.data.DataLoader, optional): Dataloader to be used for validation
            save_dir (Path|str, optional): Directory to save results
            args (dict|Namespace, optional): Arguments for the validator including task set to `pose`
        Examples:
            >>> args=dict(model='yolo11n-pose.pt', data='coco8-pose.yaml')
            >>> validator=PoseValidator(args=args)
            >>> validator()
        Notes:
            This class extends DetectionValidator with pose-specific functionality. It initializes with sigma values
            for OKS calculation and sets up PoseMetrics for evaluation. A warning is displayed when using Apple MPS due 
            to a known bug with pose models 
        """
        super().__init__(dataloader, save_dir, args)
        self.sigma=None
        self.kpt_shape=None
        self.args.task='pose'
        self.metrics=PoseMetrics()
        if isinstance(self.args.device, str) and self.args.device.lower()=='mps':
            warnings.warn("Apple MPS known Pose bug. Recommend `device=cpu` for Pose Model"
                          "See https://github.com/ultralytics/ultralytics/issues/4031.")
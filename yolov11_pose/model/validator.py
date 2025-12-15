from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

from computer_vision.yolov11_pose.utils.metrics import PoseMetrics, OKS_SIGMA
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
            
    def init_metrics(self, model:torch.nn.Module)->None:
        """Initialize evaluation metrics for YOLO pose validation
        Args:
            model (torch.nn.Module): Model to validate
        """
        super().init_metrics(model)
        self.kpt_shape=self.data['kpt_shape']
        is_pose=self.kpt_shape==[17, 3]
        nkpt=self.kpt_shape[0]
        self.sigma=OKS_SIGMA if is_pose else np.ones(nkpt)/nkpt

    def preprocess(self, batch:dict[str, Any])->dict[str, Any]:
        """Preprocess batch by converting keypoints data to float and moving it to the device"""
        batch=super().preprocess(batch)
        batch['keypoints']=batch['keypoints'].float()
        return batch

    def postprocess(self, preds:torch.Tensor|tuple[torch.Tensor, tuple[list[torch.Tensor], torch.Tensor]])->list[dict[str, torch.Tensor]]:
        """Postprocess YOLO predictions to extract and reshape keypoints for pose estimation

        This method extends the parent class postprocessing by extracting keypoints from the 'extra' field of predictions and reshaping them 
        according to the keypoint shape configuration. The keypoints are reshaped from a flattened format to the proper dimension (typically 
        [N, 17, 3] for COCO human pose format)

        Args:
            preds (torch.Tensor|tuple[torch.Tensor, tuple[list[torch.Tensor], torch.Tensor]]): Raw prediction tensor from the YOLO model containing
                bounding boxes, confidence scores, class predictions, and keypoint data. If it is a tuple, the first element in the tuple is the 
                prediction result
        Returns:
            (list[dict[str, torch.Tensor]]): Processed predictions after NMS, wehre each dict contains 'bboxes', 'conf', 'cls', 
                and 'keypoints' tensors
        Notes:
            If no keypoints are present in a prediction (empty keypoints), that prediction is skipped and continues to the next one. The keypoints
            are extracted from 'extra' field which contains additionall task-specific data beyond basic detection
        """
        preds=super().postprocess(preds)
        for pred in preds:
            # Convert 'extra' of shape Nx(M*d) [e.g. (121,51) where N=121 and M*d=17*3=51] to NxMxd [e.g., to 121x17x3]
            pred['keypoints']=pred.pop('extra').view(-1, *self.kpt_shape) # remove 'extra' and add 'keypoints'
        return preds

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

from computer_vision.yolov11_pose.utils.metrics import PoseMetrics, OKS_SIGMA, kpt_iou
from computer_vision.yolov11_pose.engine.validator import DectectionValidator
from computer_vision.yolov11_pose.utils.ops import xyxy2xywh, scale_coords
from computer_vision.yolov11_pose.engine.results import Results

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

    def _prepare_batch(self, si:int, batch:dict[str, Any])->dict[str, Any]:
        """Prepare a batch for processing by converting keypoints to float and scaling to original dimensions
        
        Args:
            si (int): Batch index
            batch (dict[str, Any]):dict containing batch data with keys like 'keypoints', 'batch_idx', etc.
        Returns:
            (dict[str, Any]): Prepared batch with keypoints scaled to original image dimensions
        Notes:
            This method extends the parent class's _prepare_batch method by adding keypoints processing. Keypoints are
            scaled from normalized coordinates to original image dimensions
        """
        pbatch=super()._prepare_batch(si, batch)
        kpts=batch['keypoints'][batch['batch_idx']==si]
        h,w=pbatch['imgsz']
        kpts=kpts.clone()
        kpts[...,0]*=w
        kpts[...,1]*=h
        pbatch['keypoints']=kpts
        return pbatch

    def _process_batch(self, preds:dict[str, torch.Tensor], batch:dict[str, Any])->dict[str, np.ndarray]:
        """Return correct prediction matrix by computing Intersection over Union (IoU) between detections and ground truth
        
        Args:
            preds (dict[str, torch.Tensor]): Dict containing prediction data with keys 'cls' for class predictions and 
                'keypoints' for keypoint predictions
            batch (dict[str, Any]): Dict containing ground truth data with 'cls' for class labels, 'bboxes' for bounding 
                boxes, and 'keypoints' for keypoint annotations
        Returns:
            (dict[str, np.ndarray]): Dict containing the correct prediction matrix including 'tp_p' for pose true positive
                across 10 IoU levels
        Notes:
            `0.53` scale factor used in area computation is referenced from 
            https://github.com/jin-s13/xtcocoapi/blob/master/xtcocotools/cocoeval.py#L384
            It is used to convert an area of bounding box to an area of person segmentation 
        """
        tp=super()._process_batch(preds, batch)
        gt_cls=batch['cls']
        if gt_cls.shape[0]==0 or preds['cls'].shape[0]==0:
            tp_p=np.zeros((preds['cls'].shape[0], self.niou), dtype=bool)
        else:
            # `0.53` is from https://github.com/jin-s13/xtcocoapi/blob/master/xtcocotools/cocoeval.py#L384
            area=xyxy2xywh(batch['bboxes'])[:, 2:].prod(dim=1)*0.53 # width*height
            iou=kpt_iou(batch['keypoints'], preds['keypoints'], sigma=self.sigma, area=area)
            tp_p=self.match_predictions(preds['cls'], gt_cls, iou).cpu().numpy()
        tp.update({'tp_p':tp_p}) # update tp with kpts IoU
        return tp

    def scale_preds(self, predn:dict[str, torch.Tensor], pbatch:dict[str, Any])->dict[str, torch.Tensor]:
        """Scale predictions to the original image size"""
        return {
            **super().scale_preds(predn, pbatch),
            "kpts":scale_coords(pbatch['imgsz'], predn['keypoints'].clone(),
                                pbatch['ori_shape'], ratio_pad=pbatch['ratio_pad'])
        }
        
    def pred_to_json(self, predn:dict[str, torch.Tensor], pbatch:dict[str, Any])->None:
        """Convert YOLO predictions to COCO JSON format

        This method takes prediction tensors and a filename, converts the bounding boxes from YOLO format to COCO format,
        and appends the results to the internal JSON dict (self.jdict)

        Args:
            predn (dict[str, torch.Tensor]): Prediction dict containing 'bboxes', 'conf', 'cls', and 'keypoints' keys with 
                (N,4) bounding box coordinates, (N,) confidence scores, (N,) class predictions, and (N,17,3) keypoints
            pbatch (dict[str, Any]): Batch dict containing 'imgsz', 'ori_shape', 'ratio_pad', and 'im_file'
        Notes:
            This method extracts the image ID from the filename stem (either as an integer if numeric or as a string),
            converts bounding boxes from xyxy to xywh format, and adjust coordinates from center to top-left corner before
            saving to the JSON dict
        """
        super().pred_to_json(predn, pbatch)
        kpts=predn['kpts'] if 'kpts' in predn else predn['keypoints']
        for i, k in enumerate(kpts.flatten(1,2).tolist()): # (N,17,3) to (N,17*3)=(N,51)
            self.jdict[-len(kpts)+i]['keypoints']=k # iterate from -N, -N+1, -N+2, ..., -1 equal to 0, 1, ..., N

    def save_one_txt(self, predn:dict[str, torch.Tensor], save_conf:bool, shape:tuple[int, int], file:Path)->None:
        """Save YOLO pose detections to a text file in normalized coordinates

        Args:
            predn (dict[str, torch.Tensor]): Prediction dict with keys 'bboxes', 'conf', 'cls', and 'keypoints'
            save_conf (bool): Whether to save confidence scores
            shape (tuple[int,int]): Shape of the original image (height,width)
            file (Path): Output file path to save detections
        Notes:
            The output format is: class_id x_center y_center width height confidence keypoints where keypoints are normalized (x,y,visibility) 
            values for each point
        """
        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=torch.cat([predn['bboxes'], predn['conf'].unsqueeze(-1), predn['cls'].unsqueeze(-1)], dim=1),
            keypoints=predn['keypoints'],
        ).save_txt(file, save_conf=save_conf)
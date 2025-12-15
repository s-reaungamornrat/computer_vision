from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from computer_vision.yolov11_pose.cfg import get_cfg
from computer_vision.yolov11_pose.utils.checks import check_imgsz
from computer_vision.yolov11_pose.utils.metrics import DetMetrics, ConfusionMatrix
from computer_vision.yolov11_pose.data.utils import check_det_dataset
from computer_vision.yolov11_pose.data.converter import coco80_to_coco91_class
from computer_vision.yolov11_pose.utils.torch_utils import unwrap_model
from computer_vision.yolov11_pose.utils.nms import non_max_suppression # for post processing

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
        self.save_dir=self.args.save_dir if isinstance(self.args.save_dir, Path) else Path(self.args.save_dir)
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
        
    def init_metrics(self,model:torch.nn.Module)->None:
        """Initialize evaluation metrics for YOLO detection validation
        Args:
            model (torch.nn.Module): Model to evaluate
        """
        val=self.data.get(self.args.split, "") # validation path
        self.is_coco=(isinstance(val, str) and 'coco' in val and 
                      (val.endswith(f'{os.sep}val2017.txt') or val.endswith(f'{os.sep}test-dev2017.txt')))
        self.is_lvis=isinstance(val, str) and 'lvis' in val and not self.is_coco # LVIS
        self.class_map=coco80_to_coco91_class() if self.is_coco else list(range(1, len(model.names)+1))
        self.args.save_json|=self.args.val and (self.is_coco or self.is_lvis) and not self.training # run final val
        self.names=model.names
        self.nc=len(model.names)
        self.end2end=getattr(model, 'end2end', False)
        self.seen=0
        self.jdict=[] # (list[dict[str, Any]]): List fir storing JSON detection results
        self.metrics.names=model.names
        self.confusion_matrix=ConfusionMatrix(names=model.names, save_matches=self.args.plots and self.args.visualize)

    def preprocess(self, batch:dict[str, Any])->dict[str, Any]:
        """Preprocess batch of images for YOLO validation

        Args:
            batch (dict[str, Any]): Batch containing images and annotations
        Returns:
            (dict[str, Any]): Preprocess batch
        """
        for k, v in batch.items():
            if isinstance(v, torch.Tensor): batch[k]=v.to(device=self.device, non_blocking=self.device.type=='cuda')
        batch['img']=(batch['img'].half() if self.args.half else batch['img'].float())/255
        return batch

    def postprocess(self, preds:torch.Tensor|tuple[torch.Tensor, tuple[list[torch.Tensor], torch.Tensor]])->list[dict[str, torch.Tensor]]:
        """Apply Non-maximum suppression to prediction outputs

        Args:
            preds (torch.Tensor|tuple[torch.Tensor, tuple[list[torch.Tensor], torch.Tensor]]): Raw predictions from the model. If it is a tuple,
                the first element in the tuple is the prediction result
        Returns:
            (list[dict[str, torch.Tensor]]): Processed predictions after NMS, wehre each dict contains 'bboxes', 'conf', 'cls', 
                and 'extra' tensors
        """
        outputs=non_max_suppression(preds, conf_thres=self.args.conf, iou_thres=self.args.iou, 
                                    agnostic=self.args.single_cls or self.args.agnostic_nms,
                                    multi_label=True, max_det=self.args.max_det, 
                                    nc=0 if self.args.task=='detect' else self.nc,
                                    rotated=self.args.task=='obb', end2end=self.end2end)
        return [{'bboxes':x[:,:4], 'conf':x[:,4], 'cls':x[:,5], 'extra':x[:,6:]} for x in outputs]
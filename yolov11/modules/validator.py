from __future__ import annotations

import os
import cv2
import math
import yaml
import copy
from pathlib import Path
from typing import Any
from argparse import Namespace

import torch
import numpy as np
from computer_vision.yolov11.utils.check import check_imgsz
from computer_vision.yolov11.data.converter import coco80_to_coco91_class
from computer_vision.yolov11.utils.metrics import ConfusionMatrix, box_iou, DetMetrics
from computer_vision.yolov11.utils import nms, ops
from computer_vision.yolov11.utils.plotting import plot_labels, plot_predictions

class DetectionValidator:
    def __init__(self, hyperparam:Path|str|dict, data_cfg:Path|str|dict, dataloader:torch.utils.data.DataLoader=None, 
                 save_dir:Path | str=None, args:Namespace=None):
        """
        Initialize a DetectionValidator instance
        Args:
            hyperparam (Path|str|dict): Hyperparameter configuration file in yaml format
            dataloader (torch.utils.data.DataLoader, optional): Dataloader to be used for validation
            save_dir (Path | str, optional): Directory to save results
            args (Namespace, optional): Configuration for the validator
        """
        if isinstance(hyperparam, str): hyperparam=Path(hyperparam) # Hyperparameters
        if isinstance(hyperparam, Path):
            assert hyperparam.is_file(), f'{cfg} does not exist'
            with open(hyperparam) as f: self.hyperparam=yaml.load(f, Loader=yaml.SafeLoader)
        elif not isinstance(hyperparam, dict): raise TypeError(f'hyperparam must be Path/dict/str but got {type(hyperparam)}')
        else: self.hyperparam=hyperparam

        # Data dict containing dataset information
        if isinstance(args.data_cfg, str): args.data_cfg=Path(args.data_cfg)
        if isinstance(args.data_cfg, Path): 
            assert args.data_cfg.is_file(), f'{args.data_cfg} does not exist'
            with open(args.data_cfg, encoding="utf8") as f: self.data=yaml.load(f, Loader=yaml.SafeLoader)
        elif not isinstance(args.data_cfg, dict): raise TypeError(f'args.data_cfg must be Path/dict/str but got {type(args.data_cfg)}')
        else: self.data=args.data_cfg
            
        # Merge the namespace without overriding the original args
        for k, v in vars(Namespace(**self.hyperparam)).items():
            if not hasattr(args, k): setattr(args, k, v)
        self.args=args
        
        self.dataloader=dataloader
        self.stride=None # Model stride for padding calculation
        self.batch_i=None # Current batch index
        self.training=True # Whether the model is in training mode
        self.names=None # Class name mapping
        self.seen=None # Number of images seen so far during validaton
        self.stats=None # Statistics collected during validation
        self.confusion_matrix=None # Confusion matrix for classification evaluation
        self.nc=None # Number of classes
        self.iouv=None # IoU thresholds from .5 to 0.95 in spaces of 0.05
        self.jdict=None # List to store JSON validation results
        # Dict storing the respective batch processing times for each key in milliseconds
        self.speed={'preprocess':0., 'inference':0., 'loss':0., 'postprocess':0.}

        self.save_dir=save_dir or Path(args.output_dirpath)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        if self.args.conf is None:
            self.args.conf=0.01 if self.args.task=='obb' else 0.001 # reduce OBB val memory usage
        self.args.imgsz=check_imgsz(self.args.imgsz, max_dim=1)

        self.plots={}

        self.is_coco=False
        self.is_lvis=False
        self.class_map=None
        self.args.task='detect'
        self.iouv=torch.linspace(0.5, 0.95, 10) # IoU vector for mAP@.5:.95
        self.niou=self.iouv.numel()
        self.metrics=DetMetrics()

    def init_metrics(self, model:torch.nn.Module)->None:
        """
        Initialize evaluation metrics for YOLO detection validation
        Args:
            model (torch.nn.Module): Model to validate
        """
        val=self.data.get(self.args.split, "")
        self.is_coco=(isinstance(val, str) and "coco" in val and
                      (val.endswith(f"{os.sep}val2017.txt") or val.endswith(f"{os.sep}test-dev2017.txt")))
        self.is_lvis=isinstance(val, str) and "lvis" in val and not self.is_coco # is LVIS
        self.class_map=coco80_to_coco91_class() if self.is_coco else list(range(1, len(model.names)+1))
        self.args.save_json |=self.args.val and (self.is_coco or self.is_lvis) and not self.training # run final val
        self.names=model.names
        self.nc=len(model.names)
        self.end2end=getattr(model, "end2end", False)
        self.seen=0
        self.metrics.names=model.names
        self.confusion_matrix=ConfusionMatrix(names=model.names, save_matches=self.args.plots and self.args.visualize)

    def preprocess(self, batch:dict[str, Any])->dict[str, Any]:
        """
        Preprocess batch of images for YOLO validation
        Args:
            batch (dict[str, Any]): Batch containing images and annotations
        Returns:
            (dict[str, Any]): Preprocessed batch
        """
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k]=v.to(self.device, non_blocking=self.device.type=='cuda')
        batch['img']=(batch['img'].float())/255.
        return batch

    def postprocess(self, preds:torch.Tensor)->list[dict[str, torch.Tenor]]:
        """
        Apply non-maximum suppression to prediction
        Args:
            preds (torch.Tensor): Raw predictions from the model.
        Returns:
            (list[dict[str, torch.Tensor]]): Processed predictions after NMS, where each dict containts
                `bboxes`, `conf`, `cls` and `extra` tensors.
        """
        outputs=nms.non_max_suppression(prediction=preds, conf_thres=self.args.conf, iou_thres=self.args.iou, 
                                        classes=None, agnostic=self.args.single_cls or self.args.agnostic_nms, 
                                        multi_label=True, max_det=self.args.max_det, 
                                        nc=0 if self.args.task=='detect' else self.nc,
                                        max_nms=30000, max_wh=7680, end2end=self.end2end, return_idxs= False)
        return [{'bboxes':x[:,:4], 'conf':x[:,4], 'cls':x[:,5], 'extra':x[:,6:]} for x in outputs]

    def _prepare_batch(self, si:int, batch:dict[str, Any])->dict[str, Any]:
        """
        Prepare a batch of images and annotations for validation
        Args:
            si (int): Batch index
            batch (dict[str, Any]): Batch data containing images and annotation
        Returns:
            (dict[str, Any]): Prepared batch with processed annotations
        """
        
        idx=batch['batch_idx']==si # N element of True and False
        cls=batch['cls'][idx].squeeze(-1) # Nx1 -> Mx1 where M<N
        bbox=batch['bboxes'][idx] # Mx4 normalized boxes
        ori_shape=batch['ori_shape'][si] # a single tuple of height and width
        imgsz=batch['img'].shape[2:] # HxW
        ratio_pad=batch['ratio_pad'][si] # a tuple of ratio-tuple and pad-tuple of height and width
        if cls.shape[0]:
            bbox=ops.xywh2xyxy(bbox)*torch.tensor(imgsz, device=self.device)[[1,0,1,0]] # target boxes in pixel units
        return {'cls':cls, 'bboxes':bbox, 'ori_shape':ori_shape, 
                'imgsz':imgsz, 'ratio_pad': ratio_pad, 'im_file':batch['im_file'][si]}

    def _prepare_pred(self, pred:dict[str, torch.Tensor])->dict[str, torch.Tensor]:
        """
        Prepare predictions for evaluation against ground truth
        Args:
            pred (dict[str, torch.Tensor]): Post-processed predictions frm the model
        Returns:
            (dict[str, torch.Tensor]): Prepared predictions in native space
        """
        if self.args.single_cls: pred['cls']*=0
        return pred


    def match_predictions(self, pred_classes:torch.Tensor, true_classes:torch.Tensor, iou:torch.Tensor)->torch.Tensor:
        """
        Match predictions to ground truth objects using IoU
        Args:
            pred_classes (torch.Tensor): Predicted class indices of shape (D,)
            true_classes (torch.Tensor): Target class indices of shape (L,)
            iou (torch.Tensor): An LxD tensor containing the pairwise IoU values for predictions and ground truth
        Returns:
            (torch.Tensor): Correct tensir of shape (D,10) for 10 IoU thresholds
        """
        # Dx10 matrix where D is detections and 10 is IoU threshold
        correct=np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
        # LxD matrix where L is labels (rows) and D is detections (columns)
        correct_class=true_classes[:,None]==pred_classes # (L->Lx1 , D) -> LxD
        iou=iou*correct_class.to(dtype=iou.dtype) # zero out wrong classes
        iou=iou.cpu().numpy()
        for i, threshold in enumerate(self.iouv.cpu().tolist()):
            # LxD IoU > threshold with classes matched
            # tuple of 2 array for 2D dimension: one for L and the others for D
            matches=np.nonzero(iou>=threshold) 
            # Nx2 where N is the number of matches and 2 for L index and D index respectively
            matches=np.array(matches).T 
            if matches.shape[0]:
                if matches.shape[0]>1:
                    # iou[matches[:,0], matches[:,1]] gets IoUs that yeild match. This will return IoU for each match
                    # iou[matches[:,0], matches[:,1]].argsort() gets match indices that sort iou from small IoU to large IoU
                    # iou[matches[:,0], matches[:,1]].argsort()[::-1] gets match indices that sort from large to small
                    matches=matches[iou[matches[:,0], matches[:,1]].argsort()[::-1]]
                    # get matches that associates with unique detections
                    matches=matches[np.unique(matches[:,1], return_index=True)[1]]
                    # get matches that associates with unique labels
                    matches=matches[np.unique(matches[:,0], return_index=True)[1]]
                # Put True in (detection-index, threshold-index) in correct
                correct[matches[:,1].astype(int),i]=True
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)

    def _process_batch(self, preds:dict[str, torch.Tensor], batch:dict[str, Any])->dict[str, np.ndarray]:
        """
        Return correct prediction matrix
        Args:
            preds (dict[str, torch.Tensor]): Dict containing prediction data with `bboxes` anc `cls` keys
            batch (dict[str, Any]): Batch dict containing ground truth data with `bboxes` and `cls` keys
        Returns:
            (dict[str, np.ndarray]): Dict containing `tp` key with correct prediction matrix of shape 
                (N, 10) for 10 IoU levels
        """
        # predn, pbatch
        if batch['cls'].shape[0]==0 or preds['cls'].shape[0]==0: 
            return {'tp':np.zeros((preds['cls'].shape[0], self.niou), dtype=bool)}
        iou=box_iou(batch['bboxes'], preds['bboxes'])
        return {'tp':self.match_predictions(preds['cls'], batch['cls'], iou).cpu().numpy()}

    def pred_to_json(self, predn:dict[str, torch.Tensor], pbatch:dict[str, Any])->None:
        """
        Serialize YOLO predictions to COCO json format
        Args:
            predn (dict[str, torch.Tensor]): Predictions dict containing `bboxes`, `conf`, and `cls` keys
                with bounding box coordinates (top-left corner and width, height in pixel units), confidence scores,
                and class predictions
            pbatch (dict[str, Any]): Batch dict containing `imgsz`, `ori_shape`, `ratio_pad` and `im_file`
        Examples:
            >>> result={
            ...          'image_id':42,
            ...          'file_name': '42.jpeg',
            ...          'category_id': 18,
            ...          'bbox':[258.15, 41.29, 348.26, 243.78],
            ...          'score':0.236,
            }
        """
        path=Path(pbatch['im_file'])
        stem=path.stem
        image_id=int(stem) if stem.isnumeric() else stem
        box=ops.xyxy2xywh(predn['bboxes']) # xywh in pixel units
        box[:,:2]-=box[:,2:]/2 # xy center to top-left corner
        for b, s, c in zip(box.tolist(), predn['conf'].tolist(), predn['cls'].tolist()):
            self.jdict.append({
                'image_id':image_id,
                'file_name': path.name,
                'category_id': self.class_map[int(c)],
                'bbox': [round(x, 3) for x in b], # round to 3 floating precision
                'score': round(s, 5)
            })

    def update_metrics(self, preds:list[dict[str, torch.Tensor]], batch:dict[str, Any])->None:
        """
        Update metrics with new predictions and ground truth
        Args:
            preds (list[dict[str, torch.Tensor]]): List of predictions from the model
            batch (dict[str, Any]): Batch data containing ground truth
        """
    
        for si, pred in enumerate(preds):
            self.seen+=1
            pbatch=self._prepare_batch(si, batch)
            predn=self._prepare_pred(pred)
            cls=pbatch['cls'].cpu().numpy() # size (N,)
            no_pred=predn['cls'].shape[0]==0
        
            self.metrics.update_stats({
                **self._process_batch(predn, pbatch), 'target_cls':cls, 'target_img':np.unique(cls),
                'conf':np.zeros(0) if no_pred else predn['conf'].cpu().numpy(), # size (M,)
                'pred_cls':np.zeros(0) if no_pred else predn['cls'].cpu().numpy() # size (M,)
            })
            # Evaluate
            if self.args.plots:
                self.confusion_matrix.process_batch(predn, pbatch, conf=self.args.conf)
                if self.args.visualize:
                    self.confusion_matrix.plot_matches(batch['img'][si], pbatch['im_file'], self.save_dir)
                    
            if no_pred: continue
                
            # Save
            if self.args.save_json:
                # Scale boxes
                predn_scaled={**predn, 'bboxes':ops.scale_boxes(pbatch['imgsz'], predn['bboxes'].clone(), pbatch['ori_shape'], 
                                                   ratio_pad=pbatch['ratio_pad'])}
                self.pred_to_json(predn_scaled, pbatch)

    
    def __call__(self, trainer=None,  model=None):
        """
        Execute validation process, running inference on dataloader and computing performance metrics
        Args:
            trainer (object, optional): Trainer object that contains the model to validate
            model (nn.Module, optional): Model to validate if not using a trainer
        Returns:
            (dict): Dictionary containing validation statistics
        """

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from computer_vision.yolov11_pose.cfg import get_cfg
from computer_vision.yolov11_pose.utils.plotting import plot_images
from computer_vision.yolov11_pose.utils.checks import check_imgsz
from computer_vision.yolov11_pose.utils.metrics import DetMetrics, ConfusionMatrix, box_iou
from computer_vision.yolov11_pose.data.utils import check_det_dataset
from computer_vision.yolov11_pose.data.converter import coco80_to_coco91_class
from computer_vision.yolov11_pose.utils.torch_utils import unwrap_model
from computer_vision.yolov11_pose.utils.nms import non_max_suppression # for post processing
from computer_vision.yolov11_pose.utils.ops import xywh2xyxy, xyxy2xywh, scale_boxes
from computer_vision.yolov11_pose.engine.results import Results

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

    def _prepare_batch(self, si:int, batch:dict[str, Any])->dict[str, Any]:
        """Prepare a batch of images and annotations for validation
        Args:
            si (int): Batch index
            batch (dict[str, Any]): Batch data containing images and annotations
        Returns:
            (dict[str, Any]): Prepared batch with processed annotations.
        """
        idx=batch['batch_idx']==si # (Q,)
        cls=batch['cls'][idx].squeeze(-1)  # (N,) where N<Q
        bbox=batch['bboxes'][idx]  # (N,4)
        ori_shape=batch['ori_shape'][si]
        imgsz=batch['img'].shape[2:]
        ratio_pad=batch['ratio_pad'][si] # tuple of [h-ratio, w-ratio],[h-pad, w-pad]
        if cls.shape[0]:
            bbox=xywh2xyxy(bbox)*torch.tensor(imgsz, device=self.device)[[1,0,1,0]] # (N,4)*(4,) target boxes
        return {'cls':cls, 'bboxes':bbox, 'ori_shape':ori_shape, 
                'imgsz':imgsz, 'ratio_pad':ratio_pad, 'im_file':batch['im_file'][si]}

    def _prepare_pred(self, pred:dict[str, torch.Tensor])->dict[str, torch.Tensor]:
        """Prepare predictions for evaluation against ground truth
        Args:
            pred (dict[str, torch.Tensor]): Post-processed predictions from the model
        Returns:
            (dict[str, torch.Tensor]): Prepared predictions in native space
        """
        if self.args.single_cls: pred['cls']*=0
        return pred

    def match_predictions(self, pred_classes:torch.Tensor, true_classes:torch.Tensor, iou:torch.Tensor,
                          use_scipy:bool=False)->torch.Tensor:
        """Match predictions to ground truth objects using IoU
        Args:
            pred_classes (torch.Tensor): Predicted class indices of shape (N,)
            true_classes (torch.Tensor): Target class indices of shape (M,)
            iou (torch.Tensor):An MxN tensor containing the pairwise IoU values for predictions and ground truth
            use_scipy (bool, optional): Whether to use scipy for matching (more precise)
        Returns:
            (torch.Tensor): Correct tensor of shape (N, 10) for 10 IoU thresholds
        """
        # Nx10 matrix, where N is the number of detections and 10 is for the 10 IoU thresholds
        correct=np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
        # MxN matrix where M is the number of labels
        correct_class=true_classes[:,None]==pred_classes
        iou=iou*correct_class # zero out the wrong classes
        iou=iou.cpu().numpy()
        for i, threshold in enumerate(self.iouv.cpu().tolist()):
            if use_scipy:
                # WARNING: known issue that reduces mAP in https://github.com/ultralytics/ultralytics/pull/4708
                import scipy # scope import to avoid importing for all commands
                cost_matrix=iou*(iou>=threshold)
                if cost_matrix.any():
                    labels_idx, detections_idx=scipy.optimize.linear_sum_assignment(cost_matrix)
                    valid=cost_matrix[labels_idx, detections_idx]>0
                if valid.any(): correct[detections_idx[valid],i]=True
            else:
                # IoU > threshold and classes match
                matches=np.nonzero(iou>=threshold) # index to ground truths, index to detections
                matches=np.array(matches).T # Qx2 where Q is the number of pairs
                if matches.shape[0]:
                    if matches.shape[0]>1:
                        # get indices listing IoU (from small to large then reverse the list to) from large to small IoU
                        matches=matches[iou[matches[:,0], matches[:,1]].argsort()[::-1]]
                        # maintain only match of unique predictions
                        matches=matches[np.unique(matches[:,1], return_index=True)[1]] 
                        # maintain only match corresponding to unique ground truth
                        matches=matches[np.unique(matches[:,0], return_index=True)[1]]
                    correct[matches[:,1].astype(int), i]=True
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)

    def _process_batch(self, preds:dict[str, torch.Tensor], batch:dict[str, Any])->dict[str, np.ndarray]:
        """Return correct prediction matrix
        
        Args:
            preds (dict[str, torch.Tensor]): Dict containing prediction data with 'bboxes' and 'cls' keys
            batch (dict[str, Any]): Batch dict containing ground truth data with 'bboxes' and 'cls' keys
        Returns:
            (dict[str, np.ndarray]): Dict containing 'tp' key with correct prediction matrix of shape (N, 10) for 
                10 IoU levels
        """
        if batch['cls'].shape[0]==0 or preds['cls'].shape[0]==0:
            return {'tp':np.zeros((preds['cls'].shape[0], self.niou), dtype=bool)}
        iou=box_iou(batch['bboxes'], preds['bboxes'])
        return {'tp':self.match_predictions(preds['cls'], batch['cls'], iou).cpu().numpy()}

    def scale_preds(self, predn:dict[str, torch.Tensor], pbatch:dict[str, Any])->dict[str, torch.Tensor]:
        """Scale predictions to the original image size"""
        return {**predn, 
                "bboxes":scale_boxes(pbatch['imgsz'], predn['bboxes'].clone(),
                                     pbatch['ori_shape'], ratio_pad=pbatch['ratio_pad'])}
        
    def pred_to_json(self, predn:dict[str, torch.Tensor], pbatch:dict[str, Any])->None:
        """Serial YOLO predictions to COCO json format
        Args:
            predn (dict[str, torch.Tensor]): Prediction dict containing 'bboxes', 'conf', and 'cls', keys with (N,4) bounding box coordinates,
                (N,) confidence scores, and (N,) class predictions
            pbatch (dict[str, Any]): Batch dict containing 'imgsz', 'ori_shape', 'ratio_pad', and 'im_file'
        Notes:
            This method extracts the image ID from the filename stem (either as an integer if numeric or as a string),
            converts bounding boxes from xyxy to xywh format, and adjust coordinates from center to top-left corner before
            saving to the JSON dict
        """
        path=Path(pbatch['im_file'])
        stem=path.stem
        image_id=int(stem) if stem.isnumeric() else stem
        box=xyxy2xywh(predn['bboxes']) # xywh
        box[:,:2]-=box[:,2:]/2 # xy center to top-left corner
        for b, s, c in zip(box.tolist(), predn['conf'].tolist(), predn['cls'].tolist()):
            self.jdict.append({
                'image_id': image_id,
                'file_name': path.name,
                'category_id':self.class_map[int(c)],
                'bboxes': [round(x,3) for x in b],
                'score': round(s, 5)
            })

    def save_one_txt(self, predn:dict[str, torch.Tensor], save_conf:bool, shape:tuple[int, int], file:Path)->None:
        """Save YOLO detections to a txt file in normalized coordinates in a specific format

        Args:
            predn (dict[str, torch.Tensor]): Dict containing predictions with keys 'bboxes', 'conf', and 'cls'
            save_conf (bool): Whether to save confidence scores
            shape (tuple[int, int]): Shape of the original image (height, weight)
            file (Path): File path to save the detections
        """
        Results(np.zeros((shape[0], shape[1]), dtype=np.uint8),
               path=None,
               names=self.names,
               boxes=torch.cat([predn['bboxes'], predn['conf'].unsqueeze(-1), predn['cls'].unsqueeze(-1)], dim=1),
               ).save_txt(file, save_conf=save_conf)

    def update_metrics(self, preds:list[dict[str, torch.Tensor]], batch:dict[str, Any])->None:
        """Update metrics with new predictions and ground truth
        
        Args:
            preds (list[dict[str, torch.Tensor]]): List of predictions from the model
            batch (dict[str, Any]): Batch data containing ground truth
        """
        for si, pred in enumerate(preds):
            self.seen+=1
            pbatch=self._prepare_batch(si, batch)
            predn=self._prepare_pred(pred)
        
            cls=pbatch['cls'].cpu().numpy()
            no_pred=predn['cls'].shape[0]==0
            # pass numpy array inputs
            self.metrics.update_stats({**self._process_batch(predn, pbatch), # tp:(M,10) where 10 is the number of IoU thresholds
                                       "target_cls":cls, # (N,)
                                       "target_img":np.unique(cls),
                                       "conf":np.zeros(0) if no_pred else predn["conf"].cpu().numpy(), # (M,)
                                       "pred_cls":np.zeros(0) if no_pred else predn["cls"].cpu().numpy(), # (M,)
                                       })
            # Evaluate
            if self.args.plots:
                self.confusion_matrix.process_batch(predn, pbatch, conf=self.args.conf)
                if self.args.visualize:
                    self.confusion_matrix.plot_matches(batch['img'][si], pbatch['im_file'], self.save_dir)
            if no_pred: continue
        
            # Save
            if self.args.save_json or self.args.save_txt:
                predn_scaled=self.scale_preds(predn, pbatch)
            if self.args.save_json:
                self.pred_to_json(predn_scaled, pbatch)
            if self.args.save_txt:
                self.save_one_txt(predn_scaled, self.args.save_conf, pbatch['ori_shape'],
                                  self.save_dir/'labels'/f"{Path(pbatch['im_file']).stem}.txt")

    def plot_val_samples(self, batch:dict[str, Any], ni:int)->None:
        """Plot validation image samples

        Args:
            batch (dict[str, Any]): Batch containing images and annotations
            ni (int): Batch index
        """
        plot_images(labels=batch, paths=batch['im_file'], fname=self.save_dir/f'val_batch{ni}_labels.jpg', names=self.names)

    def plot_predictions(self, batch:dict[str, Any], preds:list[dict[str, torch.Tensor]], ni:int, max_det:int|None=None)->None:
        """Plot predicted bounding boxes on input images and save the results

        Args:
            batch (dict[str, Any]): Batch containing images and annotations, with keys including 'batch_idx', 'bboxes', 'cls', 'im_file', 'img', 
                'keypoints', 'ori_shape', 'ratio_pad', 'resized_shape', etc.
            preds (list[dict[str, torch.Tensor]]): List of predictions from the model, where the length of preds is the number of detections,
                each dict is each detection containing keys: 
                - 'bboxes' of size (N,4)
                - 'conf' of size (N,)
                - 'cls' of size (N,)
                - [optionally]'keypoints' of size (N,17,3)
                - [optionally] 'masks'
            ni (int): Batch index
            max_det (Optional[int]): Maximum number of detections to plot
        """
        # TODO: optimize this
        for i, pred in enumerate(preds): pred['batch_idx']=torch.ones_like(pred['conf'])*i # add batch index to prediction
        keys=preds[0].keys()
        max_det=max_det or self.args.max_det
        batched_preds={k:torch.cat([x[k][:max_det] for x in preds], dim=0) for k in keys} 
        # TODO: fix this
        batched_preds['bboxes'][:,:4]=xyxy2xywh(batched_preds['bboxes'][:,:4]) # convert xyxy to xywh format
        plot_images(images=batch['img'], labels=batched_preds, paths=batch['im_file'], fname=self.save_dir/f'val_batch{ni}_pred.jpg',
                    names=self.names)

    def get_stats(self)->dict[str, Any]:
        """Calculate and return matrics statistics
        Returns:
            (dict[str, Any]): Dict containing metrics results
        """
        self.metrics.process(save_dir=self.save_dir, plot=self.args.plots)
        self.metrics.clear_stats() # if we clear_stats, print_results will not have stats to print, i.e., 
        # if ... len(self.metrics.stats) will be False
        return self.metrics.results_dict

    def finalize_metrics(self)->None:
        """Set final values for metrics speed and confusion matrix"""
        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(save_dir=self.save_dir, normalize=normalize)
        self.metrics.speed=None # we did not concern speed here
        self.metrics.confusion_matrix=self.confusion_matrix
        self.metrics.save_dir=self.save_dir

    def print_results(self)->None:
        """Print training/validation set metrics per class"""
        
        pf="%22s"+"%11i"*2+"%11.3g"*len(self.metrics.keys) # print format
        print(pf%("all", self.seen, self.metrics.nt_per_class.sum(), *self.metrics.mean_results()))
        if self.metrics.nt_per_class.sum()==0:
            warnings.warn(f'no labels found in {self.args.task} set, cannot compute metrics without labels')
            
        # Print results per class. If there is only 1 class, "all" has already printed the outcome so we ignore 1 class case
        if self.args.verbose and not self.training and self.nc>1: 
            for i, c in enumerate(self.metrics.ap_class_index):
                print(pf%(
                    self.names[c], self.metrics.nt_per_image[c], self.metrics.nt_per_class[c],
                    *self.metrics.class_result(i),
                ))


    def __call__(self, trainer=None, model=None):
        """Execute validation process, running inference on dataloader, and computing performance metrics
        Args:
            trainer (object, optional): Trainer object that contains the model to validate
            model (nn.Module, optional): Model to validate if not using a trainer
        Returns:
            (dict): Dict containing validation statistics
        """
        self.training=trainer is not None
        augment=self.args.augment and (not self.training)
        
        if self.training:
            self.device=trainer.device
            self.data=trainer.data
            model=trainer.model # or trainer.ema.ema
            if trainer.args.compile and hasattr(model, '_orig_mod'): model=model._orig_mod # validate non-compiled original model to avoid issues
            self.loss=torch.zeros_like(trainer.loss_items, device=trainer.device)
            self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch==trainer.epochs-1)
            model.eval()
        else:
            assert model is not None, f'Please provide model'
            assert self.dataloader is not None, f'Please provide dataloader when initialize validator'
            self.device=list(model.parameters())[0].device
            stride=model.stride
            imgsz=check_imgsz(self.args.imgsz, stride=stride)
            if str(self.args.data).rsplit('.', 1)[-1] in {'yaml', 'yml'}:
                self.data=check_det_dataset(self.args.data)
            # elif validator.args.task=='classify':
            #     validator.data=check_cls_dataset(validator.args.data, split=validator.args.split)
            else:raise FileNotFoundError(f'Dataset {self.args.data} for task={self.args.task} is not found')
        
            if self.device.type in {'cpu', 'mps'}: self.args.workers=0 # faster CPU val as time dominated by inference, not dataloading
            self.stride=model.stride # used in get_dataloader() for padding
            model.eval()
        self.init_metrics(unwrap_model(model))
        self.jdict=[] # empty before each val
        
        for batch_i, batch in enumerate(self.dataloader):
            self.batch_i=batch_i
        
            # Preprocessing
            batch=self.preprocess(batch)
        
            with torch.no_grad():
                # Inference
                preds=model(batch['img'], augment=augment)
            
                # Loss
                if self.training: self.loss+=model.loss(batch, preds)[1]
        
            # Postprocess
            preds=self.postprocess(preds)
        
            self.update_metrics(preds, batch)
            if self.args.plots and batch_i<3:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)
        
        stats=self.get_stats()
        self.finalize_metrics()
        self.print_results()
        
        if self.training:
            model.float()
            loss=self.loss.clone().detach()
            results={**stats, **trainer.label_loss_items(loss.cpu()/len(self.dataloader), prefix='val')}
            return {k:round(float(v), 5) for k, v in results.items()} # return results as 5 decimal place floating points
        else:
            if self.args.save_json and self.jdict:
                with open(str(self.save_dir/"predictions.json"), 'w', encoding='utf-8') as f:
                    print(f'Saving {f.name}...')
                    json.dump(self.jdict, f) # flatten and save
            if self.args.plots or self.args.save_json:
                print(f'Results saved to {self.save_dir}')
            return stats
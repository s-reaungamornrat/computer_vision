from __future__ import annotations

from copy import copy, deepcopy
from pathlib import Path
from typing import Any
import collections
import argparse

from computer_vision.yolov11_pose.engine.trainer import DetectionTrainer
from computer_vision.yolov11_pose.nn.tasks import PoseModel
from computer_vision.yolov11_pose.utils import DEFAULT_CFG_DICT
from computer_vision.yolov11_pose.model.validator import PoseValidator

class PoseTrainer(DetectionTrainer):
    """A class extends DetectionTrainer to train YOLO pose estimation models

    This trainer specializes in handling pose estimation tasks, managing model training, validation, and visualization of pose keypoints
    alongside bounding boxes

    Examples:
        >>> args=dict(model='yolo11n-pose.pt', data='coco8-pose.yaml', epochs=3)
        >>> trainer=PoseTrainer(overrides=args)
        >>> trainer.train()
    """
    def __init__(self, cfg=DEFAULT_CFG_DICT, overrides:dict[str, Any]|None=None):
        """Initialize a PoseTrainer object for training YOLO pose estimation models
        Args:
            cfg (dict, optional): Default configuration dict containing training parameters
            overrides (dict, optional): Dict of parameter overriding the default configuration
        Notes:
            This trainer will automatically set the task to pose regardless of what is provided in overrides.
            A warning is issued when using Apples MPS device due to known bugs with pose models
        """
        if isinstance(overrides, argparse.Namespace): overrides.task='pose'
        elif overrides is None: overrides={}
        if isinstance(overrides, dict): overrides['task']='pose'
        super().__init__(cfg, overrides)

    def get_model(self, cfg:str|Path|dict[str, Any]|None=None, weights: str|Path|None|collections.OrderedDict=None, verbose:bool=True)->PoseModel:
        """Get pose estimation model with specified configuration and weights
        Args:
            cfg (str|Path|dict[str, Any],optional): Model configuration file path or dict
            weights (str|Path|collections.OrderedDict, optional): Path to model weight file or model state-dict as value in collections.OrderDict
            verbose (bool): Whether to display model information
        Returns:
            (PoseModel): Initialized pose estimation model
        """
        model=PoseModel(cfg=cfg,nc=self.data['nc'],data_kpt_shape=self.data['kpt_shape'], verbose=verbose)
        if weights: model.load_state_dict(weights)
        return model

    def set_model_attributes(self):
        """"Set keypoints shape attribute of PoseModel"""
        super().set_model_attributes()
        self.model.kpt_shape=self.data['kpt_shape']
        kpt_names=self.data.get('kpt_names')
        if not kpt_names:
            names=list(map(str, range(self.model.kpt_shape[0])))
            kpt_names={i:names for i in range(self.model.nc)}
        self.model.kpt_names=kpt_names

    def get_validator(self):
        """Return an instance of the PoseValidator class for validation"""
        self.loss_names='box_loss', 'pose_loss', 'kobj_loss', 'cls_loss', 'dfl_loss'
        return PoseValidator(dataloader=self.test_loader, save_dir=self.save_dir, args=deepcopy(self.args))
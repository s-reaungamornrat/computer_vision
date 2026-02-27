from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Optional, Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from computer_vision.slowfast.mmaction.structures.action_data_sample import ActionDataSample
from computer_vision.slowfast.mmaction.models.losses.cross_entropy_loss import CrossEntropyLoss
from computer_vision.slowfast.mmaction.utils import SampleList


class BaseHead(nn.Module, metaclass=ABCMeta):
    """Base class for head
    All head should subclass it and all subclasses should overwrite `forward`

    Args:
        num_classes (int): Number of classes to be classified
        in_channels (int): Number of channels of input features
        loss_cls (dict): Config for building loss. Defaul to dict(type='CrossEntropyLoss', loss_weight=1.)
        multi_class (bool): Whether it is a multi-class recognition task. Defaul to False
        label_smooth_eps (float): Epsilon used in label smoothing. Reference: arxiv.org/abs/1906.02629. Defaults to 0.
        topk (int | tuple[int]): K for top-k accuracy, Default to (1,5)
        average_clips (dict, optional): Config for averaging class scores over multiple clips. Default to None
        init_cfg (dict, optional): Config to control initialization. Default to None
    Reference: https://github.com/open-mmlab/mmaction2/blob/main/mmaction/models/heads/base.py
    """
    def __init__(self, num_classes:int, in_channels:int, loss_cls:Callable=CrossEntropyLoss(loss_weight=1.),multi_class:bool=False,
                label_smooth_eps:float=0., topk:Union[int, tuple[int]]=(1,5), average_clips:Optional[dict]=None, init_cfg:Optional[dict]=None)->None:
        
        super(BaseHead, self).__init__()
        self.num_classes=num_classes
        self.in_channels=in_channels
        self.loss_cls=loss_cls
        self.multi_class=multi_class
        self.label_smooth_eps=label_smooth_eps
        self.average_clips=average_clips
        assert isinstance(topk, (int, tuple)), f"topk must be int or tuple[int] but got {type(topk)}"
        if isinstance(topk, int): topk=(topk,)
        assert all(k>0 for k in topk), f"Top-k must be > 0, but got {topk}"
        self.topk=topk

    @abstractmethod
    def forward(self, x, **kwargs)->Union[dict[str, torch.Tensor], list[Any], tuple[torch.Tensor], torch.Tensor]:
        # see https://github.com/open-mmlab/mmaction2/blob/main/mmaction/utils/typing_utils.py#L27 for return type
        # Any was originally ActionDataSample see https://github.com/open-mmlab/mmaction2/blob/main/mmaction/structures/action_data_sample.py
        raise NotImplementedError

    def loss(self, feats:Union[torch.Tensor, tuple[torch.Tensor]], data_samples:SampleList, **kwargs)->dict:
        """Perform forward propagation of head and loss calculation on the features of the upsteam network
        Args:
            feats (torch.Tensor|tuple[torch.Tensor]): Features from upsteam network
            data_samples (list[ActionDataSample]): The batch of data samples
        Returns:
            (dict): A dict of loss components
        """
        cls_scores=self(feats, **kwargs)
        return self.loss_by_feat(cls_scores, data_samples)

    def loss_by_feat(self, cls_scores:torch.Tensor, data_samples:SampleList)->dict:
        """Calculate the loss based on the features extracted by the head
        Args:
            cls_scores (torch.Tensor): Classification prediction results of all class, with shape (batch_size, num_classes)
            data_samples (list[ActionDataSample]): Data sample batch
        Returns:
            (dict): A dict of loss component
        """
        raise NotImplementedError("Please see https://github.com/open-mmlab/mmaction2/blob/main/mmaction/models/heads/base.py")
        

    def average_clip(self, cls_scores:torch.Tensor, num_segs:int=1)->torch.Tensor:
        """Averaging class scores over multiple clips

        Using different averaging types ('scores' or 'prob' or None, which defined in test_cfg) to computed the final averaged class score. Only 
        called in test mode

        Args:
            cls_scores (torch.Tensor): Class scores to be averaged, with shape (B*num_segs, num_classes)
            num_segs (int): Number of clips for each input sample
        Returns:
            (torch.Tensor): Averaged class scores
        """
        assert self.average_clips in ['score', 'prob', None], (f"{self.average_clips} is not supoorted. "
                                                               f"Currently supported ones are ['score', 'prob', None]")
        batch_size=cls_scores.shape[0]
        cls_scores=cls_scores.view((batch_size//num_segs, num_segs)+cls_scores.shape[1:])
        if self.average_clips is None: return cls_scores
        elif self.average_clips=='prob': cls_scores=F.softmax(cls_scores, dim=-1).mean(dim=1) # average along num_clips dimension
        elif self.average_clips=='score': cls_scores=cls_scores.mean(dim=1) # average along num_clips dimension
        return cls_scores

    def predict(self,feats:Union[torch.Tensor, tuple[torch.Tensor]], data_samples:SampleList, **kwargs)->SampleList:
        """Perform forward propagation of head and predict recognition results on the features of the upstream network
        Args:
            feats (torch.Tensor|tuple[torch.Tensor]): Features from upstream network
            data_samples (list[ActionDataSample]): The batch data samples
        Returns:
            (list[ActionDataSample]): Recognition results wrapped by ActionDataSample
        """
        cls_scores=self(feats, **kwargs)
        return self.predict_by_feat(cls_scores, data_samples)

    def predict_by_feat(self, cls_scores:torch.Tensor, data_samples:SampleList)->SampleList:
        """Transform a batch of output features extracted from the head into prediction results
        
        Args:
            cls_scores (torch.Tensor): Classification scores, has a shape (B*n_clips, num_classes) where n_clips=num_segs is the number of video
                segments/clips
            data_samples (list[ActionDataSample]): The annotation data of every samples. It usually includes information such as `gt_label`.
        Returns:
            (list[ActionDataSample]): Recognition results wrapped by ActionDataSample
        """
        num_segs=cls_scores.shape[0]//len(data_samples)
        cls_scores=self.average_clip(cls_scores, num_segs=num_segs) # from (B*n_clips, num_classes) to (B,num_classes)
        pred_labels=cls_scores.argmax(dim=-1, keepdim=True).detach() # (B, 1)
        for data_sample, score, pred_label in zip(data_samples, cls_scores, pred_labels):
            data_sample.set_pred_score(score) # (num_classes, )
            data_sample.set_pred_label(pred_label) # (1,)
        return data_samples
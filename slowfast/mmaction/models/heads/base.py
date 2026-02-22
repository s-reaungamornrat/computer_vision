from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Optional, Callable, List

import torch
import torch.nn as nn

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
        
        
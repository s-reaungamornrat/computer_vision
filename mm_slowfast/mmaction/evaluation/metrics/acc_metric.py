from __future__ import annotations

from typing import Optional, Union, Any, List, Sequence
from collections import OrderedDict
from itertools import product

import copy
import torch
import numpy as np

from computer_vision.slowfast.mmengine.evaluator.metric import BaseMetric
from computer_vision.slowfast.mmaction.evaluation.functional.accuracy import top_k_accuracy, mean_class_accuracy, mean_average_precision, mmit_mean_average_precision

def to_tensor(value):
    """Convert value to torch.Tensor"""
    if isinstance(value, np.ndarray): value=torch.from_numpy(value)
    elif isinstance(value, Sequence) and not isinstance(value, str): value=torch.tensor(value)
    assert isinstance(value, torch.Tensor), f"{type(value)} is a supported type"
    return value

class AccMetric(BaseMetric):
    """Accuracy evaluation metric
    Reference: https://github.com/open-mmlab/mmaction2/blob/main/mmaction/evaluation/metrics/acc_metric.py
    """
    default_prefix:Optional[str]='acc'

    def __init__(self, metric_list:Optional[Union[str, tuple[str]]]=('top_k_accuracy', 'mean_class_accuracy'), collect_device:str='cpu',
                 metric_options:Optional[dict]=dict(top_k_accuracy=dict(topk=(1,5))), prefix:Optional[str]=None)->None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        if not isinstance(metric_list, (str, tuple)):
            raise TypeError(f"metric_list must be str or tuple of str, but got {type(metric_list)}")
        if isinstance(metric_list, str): metrics=(metric_list,)
        else: metrics=metric_list

        # coco evaluation metrics
        for metric in metrics:
            assert metric in ['top_k_accuracy', 'mean_class_accuracy', 'mmit_mean_average_precision', 'mean_average_precision']

        self.metrics=metrics
        self.metric_options=metric_options

    def process(self, data_batch:Sequence[tuple[Any,dict]], data_samples:Sequence[dict])->None:
        """Process one batch of data samples and data_samples. The processed results should be stored in `self.results`, which will be used 
        to compute the metrics when all batches have been processed

        Args:
            data_batch (Sequence[dict]): A batch of data from the dataloader
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        data_samples=copy.deepcopy(data_samples)
        for data_sample in data_samples:
            result=dict()
            pred=data_sample['pred_score']
            label=data_sample['gt_label']

            # Ad-hoc for RGBPoseConv3d
            if isinstance(pred, dict): 
                for item_name, score in pred.items(): pred[item_name]=score.cpu().numpy()
            else: pred=pred.cpu().numpy()

            result['pred']=pred
            if label.size(0)==1: result['label']=label.item() # single label
            else: result['label']=label.cpu().numpy() # multi-label
            self.results.append(result)

    def compute_metrics(self, results:list)->dict:
        """Compute the metrics from processed results
        Args:
            results (list): The processed results of each batch
        Returns:
            (dict): The computed metrics. The keys are the names of the metrics, and the values are corresponding results
        """
        labels=[x['label'] for x in results]
        preds=[x['pred'] for x in results]
        return self.calculate(preds, labels)

    def calculate(self, preds:list[np.ndarray], labels:list[int|np.ndarray])->dict:
        """Compute the metrics from processed results
        Args:
            preds (list[np.ndarray]): List of prediction scores
            labels (list[int|np.ndarray]): List of the labels
        Returns:
            (dict): The computed metrics. The keys are the names of the metrics and the values are corresponding results
        """
        eval_results=OrderedDict()
        metric_options=copy.deepcopy(self.metric_options)
        for metric in self.metrics:
            if metric=='top_k_accuracy':
                topk=metric_options.setdefault('top_k_accuracy', {}).setdefault('topk', (1,5))
                assert isinstance(topk, (int, tuple)), f"topk must be int or tuple of int, but got {type(topk)}"
                if isinstance(topk, int): topk=(topk,)

                top_k_acc=top_k_accuracy(preds, labels, topk)
                for k, acc in zip(topk, top_k_acc): eval_results[f"top{k}"]=acc
            if metric=='mean_class_accuracy':
                mean1=mean_class_accuracy(preds, labels)
                eval_results['mean1']=mean1
            if metric in ['mean_average_precision', 'mmit_mean_average_precision']:
                if metric=='mean_average_precision':
                    mAP=mean_average_precision(preds, labels)
                    eval_results['mean_average_precision']=mAP
                elif metric=='mmit_mean_average_precision':
                    mAP=mmit_mean_average_precision(preds, labels)
                    eval_results['mmit_mean_average_precision']=mAP
        return eval_results
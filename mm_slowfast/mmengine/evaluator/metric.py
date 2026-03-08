from __future__ import annotations

from typing import Optional, Union, Any, List, Sequence
from abc import ABCMeta, abstractmethod

from torch import Tensor

import warnings

from computer_vision.slowfast.mmengine.structures.base_data_element import BaseDataElement

def _to_cpu(data:Any)->Any:
    """Transfer all tensors and BaseDataElement to cpu"""
    if isinstance(data, (Tensor, BaseDataElement)): return data.to('cpu')
    elif isinstance(data, (list, tuple)): return [_to_cpu(d) for d in data]
    elif isinstance(data, dict): return {k:_to_cpu(v) for k, v in data.items()}
    return data
    
class BaseMetric(metaclass=ABCMeta): 
    """Base class for a metric
    
    The metric first processes each batch of data_samples and predictions, and appends the prcessed result list. Then it collects all results 
    together from all ranks if distributed training is used. Finally, it computes the metrics of the entire dataset.

    A subclass should assign a meaningful value to the class attribute `default_prefix`. See argument `prefix` for details

    Args:
        collect_device (str): Device name used for collecting results from different ranks during distributed training. Must be 'cpu' or 'gpu'.
            Default to 'cpu'
        prefix (str, optional): The prefix that will be added in the metric names to disambiguate homonymous metrics of different evaluators. 
            If prefix is not provided, self.default_prefix wil be used instead. Default to None
        collect_dir (str, optional): Synchronize directory for collecting data from different ranks. This argument should only be configured when
            `collect_device` is 'cpu'. Default to None
    Reference: https://github.com/open-mmlab/mmengine/blob/main/mmengine/evaluator/metric.py
    """
    default_prefix: Optional[str]=None
    def __init__(self, collect_device:str='cpu', prefix:Optional[str]=None, collect_dir:Optional[str]=None)->None:
        
        if collect_dir is not None and collect_device!='cpu': raise ValueError('`colect_dir` could only be configured when `collect_device="cpu"`')
        
        self._dataset_meta:Union[None, dict]=None
        self.collect_device=collect_device
        self.results:list[Any]=[]
        self.prefix=prefix or self.default_prefix
        self.collect_dir=collect_dir

        if self.prefix is None: warnings.warn('The prefix is not set in metric class')

    @property
    def dataset_meta(self)->Optional[dict]:
        """Meta info of the dataset"""
        return self._dataset_meta
        
    @dataset_meta.setter
    def dataset_meta(self, dataset_meta:dict)->None:
        """Set the dataset meta info to the metric"""
        self._dataset_meta=dataset_meta
        
    @abstractmethod
    def process(self, data_batch:Any, data_samples:Sequence[dict])->None:
        """Process one batch of data samples and predictions. The processed results should be stored in `self.results`, which will be
        used to compute the metrics when all batches have been processed
        
        Args:
            data_batch (Any): A batch of data from the dataloader
            data_samples (Sequence[dict]): A batch of outputs from the model
        """
    @abstractmethod
    def compute_metrics(self, results:list)->dict:
        """Compute the metrics from processed results
        Args:
            results (list): The processed results of each batch
        Returns:
            (dict): The computed metrics. The keys are the names of the metrics, and the values are corresponding results
        """
        
    def evaluate(self, size:int)->dict:
        """Evaluate the model performance of the whole dataset after processing all batches
        Args:
            size (int): Length of the entire validation dataset. When batch size >1, the dataloader may pad some data samples to make sure all
                ranks have the samme length of dataset slice. The `collect_results` function will drop the padded data based on this size.
        Returns:
            (dict): Evaluation metrics dict on the val dataset. The keys are the names of the metrics, and the values are corresponding results.
        """
        if len(self.results)==0:
            warnings.warn(f"{self.__class__.__name___} got empty `self.results`. Please ensure that the processed "
                         "results are properly added into `self.results` in `process` method.")
        results=self.results[:size] # where self.results is list[object]
        results=_to_cpu(results)
        _metrics=self.compute_metrics(results)
        # Add prefix to metric names
        if self.prefix:  _metrics={"/".join((self.prefix, k)):v for k, v in _metrics.items()}
        # reset results list
        self.results.clear()
        return _metrics
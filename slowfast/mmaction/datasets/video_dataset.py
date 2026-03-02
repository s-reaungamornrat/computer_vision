from __future__ import annotations
from typing import Optional, Union, Sequence, Callable

import os

import torch

from computer_vision.slowfast.mmengine.dataset.base_dataset import BaseDataset

class VideoDataset(BaseDataset):
    """Video dataset for action recognition

    The dataset loads raw videos and apply specific transforms to return a dict containing the frame tensors and other information

    The ann_file is a text file with multiple lines, each line indicates a sample video which the filepath and label, which are split with a 
    whitespace. Example of annotation
    some/path/000.mp4 1
    some/path/001.mp4 1
    some/path/002.mp4 2
    some/path/003.mp4 3

    Args:
        ann_file (str): Path to annotation file
        pipeline (list[Callable]]): A sequence of data transforms
        data_prefix (dict | ConfigDict): Path to a directory where videos are held. Default to dict(video='')
        multi_class (bool_: Whether the dataset is a multi-class dataset. Default to False
        num_classes (int, optional): Number of classes of the dataset, used in multi-class datasets. Default to None
        start_index (int): A start index for frames in consideration of different filename format. However, when taking videos as input,
            it should be set to 0, since frames loaded from videos count from 0. Default to 0
        modality (str): Data modality, with options of 'RGB', 'Flow'. Default to 'RGB'
        test_mode (bool): Whether to run test/validatioon. Default to False
        delimiter (str): Delimiter for the annotation file. Default to ' '
    """
    def __init__(self, ann_file:str, pipeline:list[Callable], data_prefix:dict=dict(video=''), multi_class:bool=False, 
                 num_classes:Optional[int]=None, start_index:int=0, modality:str='RGB', test_mode:bool=False, delimiter:str=' ', 
                 **kwargs)->None:
        self.delimiter=delimiter
        self.multi_class=multi_class
        self.num_classes=num_classes
        self.start_index=start_index
        self.modality=modality
        super().__init__(ann_file, pipeline=pipeline, data_prefix=data_prefix, test_mode=test_mode, **kwargs)

    def get_data_info(self, idx:int)->dict:
        """Get annotation by index"""
        data_info=super().get_data_info(idx)
        data_info['modality']=self.modality
        data_info['start_index']=self.start_index

        if self.multi_class:
            onehot=torch.zeros(self.num_classes)
            onehot[data_info['label']]=1.
            data_info['label']=onehot
        return data_info

    def load_data_list(self)->list[dict]:
        """Load annotation file to get video information"""
        assert os.path.exists(self.ann_file), f"{self.ann_file} does not exist"
        
        # read annotation file
        with open(self.ann_file, 'r', encoding='utf-8') as f: lines=f.read().strip().split('\n')
        
        data_list=[]
        for line in lines:
            line_split=line.strip().split(self.delimiter)
            if self.multi_class:
                assert self.num_classes is not None
                filename, label=line_split[0], line_split[1:]
                label=list(map(int, label))
            # add fake label for inference data without label
            elif len(line_split)==1: filename, label=line_split[0], -1
            else: 
                filename, label=line_split
                label=int(label)
            if self.data_prefix['video'] is not None:
                filename=os.path.join(self.data_prefix['video'], filename)
            assert os.path.isfile(filename)
            data_list.append(dict(filename=filename, label=label))
        
        return data_list
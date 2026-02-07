import os
import random
import warnings
from pathlib import Path
from typing import Any, Callable, Optional, Union, cast

import av
import torch


def get_video_container(path_to_vid:str|Path, multi_thread_decode:bool=False, backend:str='pyav'):
    """Given the path to the video, return the video container
    Args:
        path_to_vid (str|Path): Path to the video
        multi_thread_decode (bool): Whether to perform multithreading decoding
        backend (str): Decoder backend, with options of 'pyav' and 'torchvision' 
    Returns:
        (container): Video container
    """
    assert backend in ('torchvision', 'pyav'), f"Unknow backend {backend}"
    
    if backend=='torchvision':
        with open(path_to_vid, 'rb') as fp: container=fp.read()
        
    else:# if backend=='pyav':
        container=av.open(path_to_vid)
        if multi_thread_decode: # enable multithreading for decoding
            container.steams.video[0].thread_type="AUTO"
    return container

class UCF101(torch.utils.data.Dataset):
    """For training, a clip is randomly sampled from every video with random cropping, scaling, and flipping. 
    For validation and testing, multiple clips are uniformly sampled from every video with uniform cropping.
    For uniform cropping, we take the left, center, and right crop if the width is larger than height, or take 
    top, center, and bottom crop if the height is larger than the width
    Args:
        root (str|Path): Root directory of the UCF101 dataset.
        annotation_path (str|Path): Path to the folder containing the split files
    """
    def __init__(self, cfg,root,annotation_path, mode, num_retries=100, fold=1):
        assert mode in ['train', 'val', 'test'], f"Split {mode} is not supported"
        self.mode=mode
        self.fold=fold
        self.root=root
        self.annotation_path=annotation_path
        self.cfg=cfg
        # random percentage for grayscale conversion
        self.p_convert_gray=cfg.DATA.COLOR_RND_GRAYSCALE
        # probability to convert raw decoded video to grayscale temporal difference
        self.p_convert_dt=cfg.DATA.TIME_DIFF_PROB
        self._video_meta={}
        self._num_retries=num_retries
        self._num_epochs=0.
        # tracks exactly how many video samples have been successfully returned (yielded) by the dataset instance during a single epoch
        self._num_yielded=0 
        self.skip_rows=self.cfg.DATA.SKIP_ROWS
        self.use_chunk_loading=(
            True if self.mode=='train' and self.cfg.DATA.LOADER_CHUNK_SIZE>0 else False
        )
        # For training or validation mode, one clip is sampled from every video. For testing
        # NUM_ENSEMBLE_VIEWS clips are sampled from every video. For every clip, NUM_SPATIAL_CROPS
        # is cropped spatially from the frames
        if self.mode in ["train", "val"]: self._num_clips=1
        else: self._num_clips=cfg.TEST.NUM_ENSEMBLE_VIEWS*cfg.TEST.SPATIAL_CROPS
        self._construct_loader()
        self.aug=False
        self.rand_erase=False
        self.use_temporal_gradient=False
        self.temporal_gradient_rate=0.
        self.cur_epoch=0

        if self.mode=='train' and self.cfg.AUG.ENABLE:
            self.aug=True
            if self.cfg.AUG.RE_PROB>0: self.rand_erase=True
                
    def _construct_loader(self):
        """Construct a list of absolute paths to all videos and their corresponding labels as well as 
        temporal index of clips within each video and the key of `_video_meta` as flatten index of the clip consider
        all clips and videos"""
        data_fpath=self.annotation_path/f"{self.mode}list{self.fold:02d}.txt"
        assert data_fpath.is_file(), f"{data_fpath} does not exist"
        with open(data_fpath, "r") as file: data_path=file.read().strip().split('\n')
        
        self._path_to_videos=[]
        self._labels=[]
        self._spatial_temporal_idx=[]
        # dataset.cur_iter=0 # I do not know what this for
        # dataset.epoch=0. # I do not know what this for
        for video_idx, path_label in enumerate(data_path):
            path, label=path_label.split(' ')
            assert (self.root/path).is_file(), f"Video file {self.root/path} does not exist"
            for clip_idx in range(self._num_clips):
                self._path_to_videos.append(self.root/path)
                self._labels.append(int(label))
                self._spatial_temporal_idx.append(clip_idx)
                self._video_meta[video_idx*self._num_clips+clip_idx]={}
        
        assert len(self._path_to_videos)>0, (
            f"Failed to load UCF101 {self.mode} split fold {self.fold} from {data_fpath} "
            f"for root at {self.root}"
        )
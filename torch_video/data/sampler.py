from typing import Any, Callable, Optional, Union, cast
from collections.abc import Iterator, Sized

import torch
from torch.utils.data import Sampler
from .dataset import VideoClipMetadata

class UniformClipSampler(Sampler):
    """Sample `num_clips_per_video` clips for each video, equally spaced. If 
    number of unique clips in the video is fewer than `num_clips_per_video`, only sample
    to the available number of clips
    
    Args:
        video_clip_metadata (VideoClipMetadata): Video clip metadat to sample from 
        num_clips_per_video (int): Number of clips to be sampled per video
    """
    def __init__(self, video_clip_metadata:VideoClipMetadata, num_clips_per_video:int)->None:
        
        if not isinstance(video_clip_metadata, VideoClipMetadata):
            raise TypeError(f"Expected video_clip_metadata to be an instance of VideoClipMetadata, but got {type(video_clip_metadata)}")
            
        self.video_clips=video_clip_metadata
        self.num_clips_per_video=num_clips_per_video

    def __iter__(self)->Iterator[int]:
        idxs=[]
        s=0
        # select num_clips_per_video for each video uniformy spaced
        for c in self.video_clips.clip_start_times:
            length=len(c)
            if length==0: continue # corner case where video decoding fails
            sampled=torch.linspace(s, s+length-1, steps=min(self.num_clips_per_video, length)).floor().to(torch.int64)
            s+=length
            idxs.append(sampled)
        
        idxs_=torch.cat(idxs).tolist()
        return iter(cast(list[int], idxs_))

    def __len__(self)->int:
        return sum(min(len(c), self.num_clips_per_video) for c in self.video_clips.clip_start_times)

class RandomClipSampler(Sampler):
    """Samples at most `max_clips_per_video` clips for each video randomly

    Args:
        video_clip_metadata (VideoClipMetadata): Video clip metadat to sample from 
        max_clips_per_video (int): Maximum number of clips to be sampled per video
    """
    def __init__(self, video_clip_metadata:VideoClipMetadata, max_clips_per_video:int)->None:
        
        if not isinstance(video_clip_metadata, VideoClipMetadata):
            raise TypeError(f"Expected video_clip_metadata to be an instance of VideoClipMetadata, but got {type(video_clip_metadata)}")
            
        self.video_clips=video_clip_metadata
        self.max_clips_per_video=max_clips_per_video

    def __iter__(self)->Iterator[int]:

        idxs=[]
        s=0
        # select at most max_video_clips_per_video for each video, randomly
        for c in self.video_clips.clip_start_times:
            length=len(c)
            size=min(length, self.max_clips_per_video)
            sampled=torch.randperm(length)[:size]+s
            s+=length
            idxs.append(sampled)
        idxs_=torch.cat(idxs)
        # shuffle all clips randomly
        perm=torch.randperm(len(idxs_))

        return iter(idxs_[perm].tolist())
        
    def __len__(self)->int:
        return sum(min(len(c), self.max_clips_per_video) for c in self.video_clips.clip_start_times)
        
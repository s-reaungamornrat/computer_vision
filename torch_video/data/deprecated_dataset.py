import os
import math
import bisect
import warnings

from pathlib import Path
from typing import Any, Callable, Optional, Union, cast, TypeVar

T = TypeVar("T")

import torch
from torchvision.io import _probe_video_from_file, _read_video_from_file, read_video, read_video_timestamps

from .utils import unfold, find_classes, has_file_allowed_extension, make_dataset
from .base import VisionDataset

class _VideoTimestampsDataset:
    def __init__(self, video_paths:list[str])->None: self.video_paths=video_paths
    def __len__(self)->int: return len(self.video_paths)
    # read_video_timestamps returns
    # - pts (List[int] if pts_unit = 'pts', List[Fraction] if pts_unit = 'sec'): presentation timestamps for each one of the frames in the video.
    # - video_fps (float, optional): the frame rate for the video
    def __getitem__(self, idx:int)->tuple[list[int], Optional[float]]: return read_video_timestamps(self.video_paths[idx])

def _collate_fn(x: T) -> T:
    """
    Dummy collate function to be used with _VideoTimestampsDataset
    """
    return x

class VideoClip:
    """
    https://github.com/pytorch/vision/blob/main/torchvision/datasets/video_utils.py#L137
    https://github.com/pytorch/vision/blob/main/torchvision/datasets/ucf101.py#L8
    https://github.com/pytorch/vision/blob/main/torchvision/io/video.py#L412
    Args:
        video_paths (list[str],optional): list of path to video files
        clip_length_in_frames (int, optional): desired/output number of frames per clip
        frames_between_clips (int, optional): distance between each clip
        frame_rate (int, optional): desired/output frame rate
        metadata_path (str, optional): path to .pt file storing video data metadata as dict[str, Any], must containing 
            - 'video_paths': list of path to video files
            - 'video_pts': list of PTS tensors, each PTS tensor is for each video whose length is equal the number of frames in each video 
            - 'video_fps': list of FPS of each video, thus whose length equal the number of videos
            if provided, this overwrite `video_paths`
    """
    def __init__(self, video_paths, clip_length_in_frames=16, frames_between_clips=1, frame_rate=None,
                num_workers=0, metadata_path=None, metadata=None):
        
        if metadata is None: assert not all(x is None for x in [video_paths, metadata_path]), 'Please provide either video_paths or metadata_path'
        self.video_paths=video_paths
        self.num_workers=num_workers

        if metadata_path is not None:
            if not os.path.isfile(metadata_path): 
                self._compute_frame_pts()
                torch.save({'video_paths':self.video_paths,
                            'video_pts':self.video_pts,
                            'video_fps':self.video_fps}, metadata_path)
            elif os.path.isfile(metadata_path):
                metadata=torch.load(metadata_path, weights_only=False)
                self.video_paths=metadata['video_paths']
                self.video_pts=metadata['video_pts']
                self.video_fps=metadata['video_fps']
        if metadata is not None:
            self.video_paths=metadata['video_paths']
            self.video_pts=metadata['video_pts']
            self.video_fps=metadata['video_fps']
            
        self.compute_clips(clip_length_in_frames, frames_between_clips, frame_rate)

    def _compute_frame_pts(self)->None:
        """
        Call `read_video_timestamps` to read Presentation Timestamp (PTS) per video per frames and FPS per video. The former is 
        returned, per video, as a tensor of length equal to the number of frames and the latter as a floating point number. 
        Returns:
            
        """
        self.video_pts=[] # len=num_videos. each entry is a tensor of shape (num_frames_in_video,)
        self.video_fps:list[float]=[]# len=num_videos.
        
        import torch.utils.data
        dl=torch.utils.data.DataLoader(_VideoTimestampsDataset(self.video_paths),batch_size=16, num_workers=self.num_workers, collate_fn=_collate_fn)
        
        for batch in dl:
            # batch is a list whose len=batch_size
            # batch is a list of size batch_size where each element is a tuple of
            #  - list of presentation timestamp (pts) whose length = frame number
            #  - floating point fps
            # batch_pts is a tuple of lists of pts per video per frames
            # batch_fps is a tuple of fps per video
            batch_pts, batch_fps=list(zip(*batch))
            # we need to specify dtype=torch.long because for empty list, torch.as_tensor will use torch.float as default dtype
            # This happens when decoding fails and no pts is returned in the list
            # batch_pts is a list of 1D tensor of pts per video, each tensor length=number of frames for each video
            batch_pts=[torch.as_tensor(pts, dtype=torch.long) for pts in batch_pts]
            self.video_pts.extend(batch_pts)
            self.video_fps.extend(batch_fps)

    @staticmethod
    def _resample_video_idx(num_frames:int, original_fps:float, new_fps:float)->Union[slice, torch.Tensor]:
        """
        Args:
            num_frames (int): required number of output frames
            original_fps (float): input fps
            new_fps (float): output fps
        Returns:
            (Union[slice, torch.Tensor]): A slice of a form (None, None, step) or 1D tensor of indices to the start frame indices of each clip to be 
                extracted
        """
        step=original_fps/new_fps
        if step.is_integer():
            # optimization: if step is integer, don't need to perform
            # advanced indexing
            return slice(None, None, int(step))
        idxs=torch.arange(num_frames, dtype=torch.float32)*step
        idxs=idxs.floor().to(torch.int64)
        return idxs

    @staticmethod
    def compute_clips_for_video(video_pts:torch.Tensor, num_frames:int, step:int, fps:Optional[float],
                                frame_rate:Optional[float]=None)->tuple[torch.Tensor, Union[list[slice], torch.Tensor]]:
        """
        Given video presentation timestamp (pts), the function determines pts for each clips based on moving/sliding windowing
        Args:
            video_pts (torch.Tensor): video presentation timestamp for a single video, whose length is equal to the number of frames
                in the original video
            num_frames (int): desired/output number of frames per clip
            step (int): distance between two clips
            fps (float, optional): input frame-per-second
            frame_rate (float, optional): output desired frame-per-second
        Returns:
            clips (torch.Tensor): extracted pts for each clip with shape (N, num_frames) where N is the number of clips and
                num_frames is the number of frames per clip
            idxs (torch.Tensor|list[slice]): indices into frames forming each clip. If tensor, it has the same shape as `clips`, 
                i.e., (N, num_frames); otherwise, it is a list of slices of length N
        """
        if fps is None:
            # if for some reason the video does not have fps (because does not have a video stream)
            # set the fps to 1. The value does not matter, because video_pts is empty anyway
            fps=1
        if frame_rate is None: frame_rate=fps
        total_frames=len(video_pts)*frame_rate/fps
         # slice or 1D tensor. If tensor, len(_idxs)> len(video_pts) if frame_rate>fps so some frames were sampled more than once
        _idxs=VideoClip._resample_video_idx(int(math.floor(total_frames)), fps, frame_rate)
        video_pts_=video_pts[_idxs]
        clips=unfold(video_pts_, num_frames, step) # (N, num_frames) where N is the number of windows/clips
        
        if not clips.numel():
            warnings.warn("There are not enough frames in the current video to get a clip for the given clip length and"
                         "frames between clips. The video (and potentially others) will be skipped")
        if isinstance(_idxs, slice): idxs=[_idxs]*len(clips) # list of len=N
        else: idxs=unfold(_idxs, num_frames, step) # (N, num_frames) same size as `clips`
        
        return clips, idxs

    def compute_clips(self, num_frames:int, step:int, frame_rate:Optional[float]=None)->None:
        """compute all consecutive sequences of clips from  video_pts. Always return clips of size `num_frames`, meaning that
        the last few frames in a video can potentially be dropped
        
        Args:
            num_frames (int): number of frames for the clip
            step (int): distance between two clips
            frame_rate (int, optional): output/desired frame rate
        """
        self.num_frames=num_frames
        self.step=step
        self.frame_rate=frame_rate
        self.clips=[]
        self.resampling_idxs=[]
        for video_pts, fps in zip(self.video_pts, self.video_fps):
            clips,idxs=self.compute_clips_for_video(video_pts, num_frames, step, fps, frame_rate)
            self.clips.append(clips)
            self.resampling_idxs.append(idxs)
        clip_lengths=torch.as_tensor([len(v) for v in self.clips])
        self.cumulative_sizes=clip_lengths.cumsum(0).tolist()

    def subset(self, indices:list[int])->"VideoClips":
        video_paths=[self.video_paths[i] for i in indices]
        video_pts=[self.video_pts[i] for i in indices]
        video_fps=[self.video_fps[i] for i in indices]
        metadata={"video_paths":video_paths,
                  "video_pts":video_pts,
                  "video_fps":video_fps}
        return type(self)(video_paths, clip_length_in_frames=self.num_frames,
                         frames_between_clips=self.step, frame_rate=self.frame_rate,
                         num_workers=self.num_workers, metadata=metadata)
    def num_clips(self)->int:
        """Number of clips that are available in the video list"""
        return self.cumulative_sizes[-1]
    def __len__(self)->int:
        return self.num_clips()
    def num_videos(self)->int:
        return len(self.video_paths)

    def get_clip_location(self, idx:int)->tuple[int, int]:
        """Convert a flattened representation of the indices into a video_idx, clip_idx representation based on `cumulative_sizes`
        which is a list of cumulative number of clips from each video
        Returns:
            video_idx (int): Which video based on cumulative_sizes 
            clip_idx (int): Which clip index inside the given video based on cumulative_sizes and video_idx
        """
        video_idx=bisect.bisect_right(self.cumulative_sizes, idx)
        if video_idx==0: clip_idx=idx
        else: clip_idx=idx-self.cumulative_sizes[video_idx-1]
        return video_idx, clip_idx

    def get_clip(self, idx:int, backend:str="pyav")->tuple[torch.Tensor, torch.Tensor, dict[str, Any],int]:
        """Get a clip from a list of videos
        Args:
            idx (int): index of the clip, must be in [0,num_clips)
        Returns:
            video (torch.Tensor): video tensor of shape (T,C,H,W) where T is the number of frames and C is the number of channels (e.g., 3 for RGB)
            audio (torch.Tensor): audio tensor of shape (C,S) where C is the number of channels (1 for mono and 2 for stereo) and S is the
                number of audio data points sampled 
            info (dict[str, Any]): dict storing `video_fps` video frame rate and `audio_fps` sampling rate of audio (e.g., 44100 Hz)
            video_idx (int): Index of the video in `video_paths`
        """
        if idx>=self.num_clips(): raise IndexError(f"Index {idx} out of range ({self.num_clips()} number of clips)")
        video_idx, clip_idx=self.get_clip_location(idx)
        video_path=self.video_paths[video_idx]
        # self.clips is a list of pts (per video per frame) for all video, thus len(self.clips)==len(video_paths)
        # self.clips[video_idx] is of size (N, num_frames) where N is the number of clips, and num_frames is number of frames per clips
        clip_pts=self.clips[video_idx][clip_idx] 
        if backend=='pyav':
            start_pts=clip_pts[0].item()
            end_pts=clip_pts[-1].item()
            # video is of shape (THWC) where T is the number of frames
            # audio is of shape (C,S) where C is the channels (1 for mono and 2 for stereo) and S is the number of audio data points extracted
            #         from the requested range
            # info is a dict[str, Any], storing `video_fps` video frame rate and `audio_fps` sampling rate of audio (e.g., 44100 Hz)
            video, audio, info=read_video(video_path, start_pts, end_pts)
        
        if self.frame_rate is not None:
            # self.resampling_idxs is list of len=len(self.video_paths)
            # self.resampling_idxs[video_idx] is a list/tensor of len=number of clips of video_idx video 
            resample_idx=self.resampling_idxs[video_idx][clip_idx] 
            if isinstance(resample_idx, torch.Tensor): resample_idx=resample_idx-resample_idx[0] # rescale to index within the clip
            video=video[resample_idx]
            info['video_fps']=self.frame_rate
        assert len(video)==self.num_frames, f"Number of video frames {video.shape[0]} is not equal to required/input number of frames {self.num_frames}"
        
        # [T,H,W,C]->[T,C,H,W]
        video=video.permute(0,3,1,2)
        return video, audio, info, video_idx

class UCF101(VisionDataset):
    def __init__(self, root:Union[str, Path], annotation_path:str, frames_per_clip:int, step_between_clips:int=1,
                frame_rate:Optional[int]=None,fold:int=1, train:bool=True, transforms:Optional[Callable]=None,
                num_workers:int=1, metadata_path:str=None)->None:
        super().__init__(root, transforms=transforms)

        if not 1<=fold<=3: raise ValueError(f"Fold should be between 1 and 3, but got {fold}")

        extensions=('avi',)
        self.fold=fold
        self.train=train

        self.classes, class_to_idx=find_classes(self.root)
        self.samples=make_dataset(self.root, class_to_idx, extensions, is_valid_file=None)
        video_list=[x[0] for x in self.samples]

        video_clips=VideoClip(video_paths=video_list, clip_length_in_frames=frames_per_clip,frames_between_clips=step_between_clips,
                          num_workers=num_workers,metadata_path=metadata_path)

        # We bookkeep the full version of video clips because we want to be able to return the metadata of full version rather than the
        # subset version of video clips
        self.full_video_clips=video_clips
        self.indices=self._select_fold(video_list, annotation_path, fold, train)
        self.video_clips=video_clips.subset(self.indices)
        self.transforms=transforms

    def _select_fold(self, video_list:list[str], annotation_path:str, fold:int, train:bool)->list[int]:
        """Read txt file listing video files for the specified fold and find the indices of those files in all `video_list`
        Args:
            video_list (list[str]): List of all absolute paths to all video files
            annotation_path (str): Path to directory containing annotation txt file for each fold
            fold (int): Data fold with options of 1, 2 or 3
            train (bool): Whether data is for training or testing
        Returns:
            (list[int]): Indices of selected video from video_list
        """
        name='train' if train else 'test'
        name=f"{name}list{fold:02d}.txt"
        f=os.path.join(annotation_path, name)
        selected_files=set()
        with open(f) as fid:
            data=fid.readlines()
            data=[x.strip().split(" ")[0] for x in data]
            data=[os.path.join(self.root, *x.split("/")) for x in data]
            selected_files.update(data)
        indices=[i for i in range(len(video_list)) if video_list[i] in selected_files]
        return indices

    @property
    def metadata(self)->dict[str, Any]:
        return self.full_video_clips.metadata

    def __len__(self)->int: return self.video_clips.num_clips()

    def __getitem__(self, idx:int)->tuple[torch.Tensor, torch.Tensor, int]:
        video, audio, info, video_idx=self.video_clips.get_clip(idx)
        label=self.samples[self.indices[video_idx]][1]
        info['name']=self.video_clips.video_paths[video_idx]
        if self.transforms is not None: video=self.transforms(video)
        return video, audio, label, info, video_idx


if __name__ == '__main__':

    data_dirpath=Path('D:/data/UCF101')
    root=data_dirpath/'UCF-101'
    annotation_path=data_dirpath/'UCF101TrainTestSplits-RecognitionTask'
    dataset=UCF101(root=root, annotation_path=annotation_path, frames_per_clip=16, step_between_clips=2, train=True) 
    video, audio, label, info, video_idx=dataset[0]
    print(f"{video.shape=}, {video.dtype=}") # video.shape=torch.Size([16, 3, 240, 320]), video.dtype=torch.uint8
    print(f"{audio.shape=}, {audio.dtype=}") # audio.shape=torch.Size([2, 18432]), audio.dtype=torch.float32
    print(f"{label=}") # label=0
    print(f"{info=}") # info={'video_fps': 25.0, 'audio_fps': 44100, 'name': 'D:\\data\\UCF101\\UCF-101\\ApplyEyeMakeup\\v_ApplyEyeMakeup_g08_c01.avi'}
    print(f"{video_idx=}") # video_idx=0
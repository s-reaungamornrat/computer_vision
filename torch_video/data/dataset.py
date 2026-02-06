import os
import bisect
from pathlib import Path
from typing import Any, Callable, Optional, Union, cast

import numpy as np

import torch
import torch.nn as nn
from torchvision.transforms import v2
from torchcodec.decoders import VideoDecoder, AudioDecoder
from torchcodec.samplers import clips_at_regular_timestamps, clips_at_random_timestamps

from .base import VisionDataset
from .utils import find_classes, has_file_allowed_extension, make_dataset, compute_clip_start_times

DATA_MEAN=(0.43216, 0.394666, 0.37645)
DATA_STD=(0.22803, 0.22145, 0.216989)
NUM_CLASSES=101

class ConvertTCHWtoCTHW(nn.Module):
    """Convert a tensor from (T,C,H,W) to (C,T,H,W) where T is the number of frames in the clip

    We convert (T,C,H,W) to (C,T,H,W) so the tensor is in a format ready to be inputted into 3D CNN where it expects input of shape
    (B,C,T,H,W)
    """
    def forward(self, video:torch.Tensor)->torch.Tensor:
        return video.permute(1,0,2,3)
         
class VideoClipMetadata: 
    """
    Args:
        video_paths (list[str],optional): List of absolute paths to video files
        clip_length_in_seconds (float, optional): How long each clip captures in second
        clip_stride_in_seconds (float, optional): Distance between the start of each clip in seconds.
        frame_rate (float, optional): Desired/output frame rate of video
        metadata (dict[str, Any], optional): Video data metadata, including 
            - 'video_paths': list of path to video files
            - 'video_pts': list of PTS tensors, each PTS tensor is for each video whose length is equal the number of frames in each video 
            - 'video_fps': list of FPS of each video, thus whose length equal the number of videos
            if provided, this overwrite `video_paths`
        sampling_type (str): Clip sampling type with options of 'regular' and 'random'
        use_audio (bool): Whether to extract audio data as well
        sample_rate (float,optional): Desired/output sampling rate of audio
        metadata_path (Union[str,Path]): Path to .pt storing video-clip metadata including clip-start timestamp, seconds_between_frame, etc.
        num_ffmpeg_threads (int): The number of threads to use for decoding. Use 1 for single-threaded decoding, use a higher number for multi-threaded 
            decoding, and use 0 lets FFmpeg decide on the number of threads. 
            see https://meta-pytorch.org/torchcodec/stable/generated/torchcodec.decoders.VideoDecoder.html#torchcodec.decoders.VideoDecoder
        n_retries (int): Number of retries to extract a clip
    Reference: https://github.com/pytorch/vision/blob/main/torchvision/datasets/video_utils.py#L137
    """
    def __init__(self, video_paths, clip_length_in_seconds=16, clip_stride_in_seconds=1, frame_rate=None, metadata:dict[str, Any]=None,
                 sampling_type:str='regular', use_audio=False, sample_rate=16000, metadata_path:Union[str,Path]=None, num_ffmpeg_threads:int=0,
                 n_retries:int=10):

        assert sampling_type in ('regular', 'random'), f"{sampling_type} must be either 'regular' or 'random'"
        
        self.sampling_type=sampling_type
        self.frame_rate=frame_rate
        self.clip_stride_in_seconds=clip_stride_in_seconds
        self.clip_length_in_seconds=clip_length_in_seconds
        self.use_audio=use_audio
        self.sample_rate=sample_rate
        self.num_ffmpeg_threads=num_ffmpeg_threads
        self.n_retries=n_retries

        if metadata_path is not None and os.path.isfile(metadata_path): metadata=torch.load(metadata_path, weights_only=False)
        if metadata is None: 
            self._compute_clip_metadata(video_paths, clip_length_in_seconds)
            if metadata_path is not None: torch.save(self.metadata, metadata_path)                                                                           
        else: self._initialize_metatdata(metadata)

    def _compute_clip_metadata(self, video_list:list[str], clip_length_in_seconds:float):
        """Compute video clip metadata 
        
        This function assigns frame-rate if users do not specify and calculate start timestamp in seconds of each clip, 
        strides between clips and strids between frames within each clip in seconds, number of frames per clips and number of clips per video
    
        Args:
            video_list (list[str]): List of absolute path to video files
            clip_length_in_seconds (float): Length of each clip in seconds
        """
        video_paths=[] # absolute paths to usable video given required clip-duration
        clip_start_times=[] # clip start timestamps, stored as a list of 1D tensor, where len=number of video and size of tensor=number of clips
    
        for video_path in video_list:
            decoder=VideoDecoder(video_path)
            start_times=compute_clip_start_times(video_duration=decoder.metadata.duration_seconds, clip_duration=clip_length_in_seconds, 
                                                 step_duration=self.clip_stride_in_seconds)
            if start_times is None: continue
    
            # save clip start-time and video-path as this is readable and usable video
            clip_start_times.append(start_times)
            video_paths.append(video_path) # readable & usable video
            # if frame_rate is not defined, we set to it to equal the frame_rate of the first video
            if self.frame_rate is None: self.frame_rate=decoder.metadata.average_fps
    
        self.video_paths=video_paths # usable video for the required clip duration
        self.clip_start_times=clip_start_times # list[Tensor] where len(self.video_paths)==len(self.clip_start_times)
        self.seconds_between_frames=1./self.frame_rate
        self.num_frames_per_clip=int(clip_length_in_seconds*self.frame_rate)
        self.num_clips_per_video=[x.numel() for x in clip_start_times] # len=len(self.video_paths)
        self.cumulative_sizes=np.cumsum(self.num_clips_per_video) # len=len(self.video_paths)

    def _initialize_metatdata(self, metadata):
        self.frame_rate=metadata['frame_rate']
        self.video_paths=metadata['video_paths']
        self.clip_start_times=metadata['clip_start_times']
        self.seconds_between_frames=metadata['seconds_between_frames']
        self.num_frames_per_clip=metadata['num_frames_per_clip']
        self.num_clips_per_video=metadata['num_clips_per_video']
        self.cumulative_sizes=metadata['cumulative_sizes']

    @property
    def metadata(self)->dict[str, Any]:
        """Return video metadata"""
        _metadata = {
            "frame_rate": self.frame_rate,
            "video_paths": self.video_paths,
            "clip_start_times": self.clip_start_times,
            "seconds_between_frames":self.seconds_between_frames,
            "num_frames_per_clip":self.num_frames_per_clip,
            "num_clips_per_video":self.num_clips_per_video,
            "cumulative_sizes":self.cumulative_sizes
        }
        return _metadata

    def subset(self, indices:list[int])->"VideoClipMetadata":
        """Get subset of video-clip metadata based on indices into the list of all videos
        Args:
            indices (list[int]): Indices of selected videos
        Returns:
            (VideoClipMetadata): Subset of video-clip metadata
        """
        video_paths=[self.video_paths[i] for i in indices]
        clip_start_times=[self.clip_start_times[i] for i in indices]
        num_clips_per_video=[self.num_clips_per_video[i] for i in indices]
        cumulative_sizes=np.cumsum(num_clips_per_video) # len=len(video_paths)
        
        metadata={
            "frame_rate": self.frame_rate,
            "video_paths": video_paths,
            "clip_start_times": clip_start_times,
            "seconds_between_frames":self.seconds_between_frames,
            "num_frames_per_clip":self.num_frames_per_clip,
            "num_clips_per_video":num_clips_per_video,
            "cumulative_sizes":cumulative_sizes
        }
        return type(self)(video_paths=video_paths,clip_length_in_seconds=self.clip_length_in_seconds, clip_stride_in_seconds=self.clip_stride_in_seconds,
                          frame_rate=self.frame_rate, metadata=metadata, sampling_type=self.sampling_type, use_audio=self.use_audio)
        
    def num_clips(self)->int:
        """Return the number of clips that are available"""
        return self.cumulative_sizes[-1]
        
    def num_videos(self)->int:
        return len(self.video_paths)

    def __len__(self)->int: return self.num_clips

    def get_clip_location(self, idx:int)->tuple[int, int]:
        """Convert a flattened representation of the indices into a video_idx and a clip_idx
        Args:
            idx (int): Clip index
        Returns:
            video_idx (int): Index to video
            clip_idx (Int): Index to clip based within this video
        """
        video_idx=bisect.bisect_right(self.cumulative_sizes, idx) 
        if video_idx==0: clip_idx=idx
        else: clip_idx=idx-self.cumulative_sizes[video_idx-1] 
        # remove sum of previous bins to get pure clip_idx relative to this video
        
        return video_idx, clip_idx

    def _get_random_clip(self, video_decoder, num_frames_per_clip, seconds_between_frames, sampling_range_start, num_clips=1, policy="wrap"):
        """Get random clip
        Args:
            video_decoder (VideoDecoder): See https://meta-pytorch.org/torchcodec/stable/generated/torchcodec.decoders.VideoDecoder.html#torchcodec.decoders.VideoDecoder
            num_frames_per_clip (int): Number of frames per clip
            seconds_between_frames (float): Seconds between each frame
            sampling_range_start (float): The start of the sampling range, which defines the first timestamp (in seconds) that a clip may start at
            num_clips (int): Number of clips to extract
            policy (str): Defines how to construct clips that span beyond the end of the video. See https://meta-pytorch.org/torchcodec/stable/generated/torchcodec.samplers.clips_at_random_timestamps.html#torchcodec.samplers.clips_at_random_timestamps
        Returns:
            (torch.Tensor): Video clip data of type uint8 and shape (T,C,H,W) where T is the number of frames, C is the number of channels,
                H is height and W is width
        """
        i=0
        while i<self.n_retries:
            try:
                video_clip = clips_at_random_timestamps(
                    video_decoder,
                    num_clips=num_clips,
                    num_frames_per_clip=self.num_frames_per_clip,
                    seconds_between_frames=self.seconds_between_frames,
                    sampling_range_start=sampling_range_start,
                    policy=policy,
                )
                return video_clip
            except Exception as e:  i+=1
                
        raise RuntimeError(f"Failed to request a random clip after trying {self.n_retries} times")
        
    def get_clip(self, idx:int, transforms:Optional[list[Callable]]=None)->tuple[torch.Tensor, torch.Tensor, dict[str,Any], int]:
        """Get a clip from a list of videos
        Args:
            idx (int): Index of the clip. Must be between [0, num_clips)
            transforms (list[callable], optional): List of decoder transforms based on torchvision.transforms.v2
        Returns:
            (torch.Tensor): Video clip data of type uint8 and shape (T,C,H,W) where T is the number of frames, C is the number of channels,
                H is height and W is width
            (torch.Tensor|None): Audio clip data of type float32 and shape (Channels, Samples)
            (dict[str, Any]): Frame rate of video and sample rate of audio
            (int): Index to selected video
        """
        if idx>=self.num_clips(): raise IndexError(f"Index {idx} out of range ({self.num_clips()} number of clips)")
        video_idx, clip_idx=self.get_clip_location(idx)
        video_path=self.video_paths[video_idx]
        clip_start_time=self.clip_start_times[video_idx][clip_idx].item() # list[tensor] where 1D tensor is a set of PTS of clips from each video
        
        # Sample clips
        video_decoder=VideoDecoder(video_path, transforms=transforms, num_ffmpeg_threads=self.num_ffmpeg_threads)
        # Extract video_clip
        # video_clip.data will be of uint tensor with shape (N,T,C,H,W) where N=1 number of clips and T number of frames per clip
        # video_clip.pts_seconds is (N,T) float tensor is the start timestamp of each frame in each clip in seconds
        # video_clip.duration_seconds is (N,T) float tensor is the duration of each frame in each clip in seconds
        if self.sampling_type=='regular':
            # we set sampling_range_end to clip_start_time+(half clip duration) since we only want to sample 1 clip
            video_clip = clips_at_regular_timestamps(
                video_decoder,
                seconds_between_clip_starts=self.clip_stride_in_seconds,
                num_frames_per_clip=self.num_frames_per_clip,
                seconds_between_frames=self.seconds_between_frames,
                sampling_range_start=clip_start_time,
                sampling_range_end=clip_start_time+(self.clip_length_in_seconds/2), # we want to only sample 1 clip
                policy="wrap",
            )
        else:
            video_clip=self._get_random_clip(video_decoder, num_frames_per_clip=self.num_frames_per_clip, 
                                             seconds_between_frames=self.seconds_between_frames,
                                             sampling_range_start=clip_start_time if np.random.uniform()>0.5 else None, num_clips=1)
            # video_clip = clips_at_random_timestamps(
            #     video_decoder,
            #     num_clips=1,
            #     num_frames_per_clip=self.num_frames_per_clip,
            #     seconds_between_frames=self.seconds_between_frames,
            #     sampling_range_start=clip_start_time if np.random.uniform()>0.5 else None,
            #     policy="wrap",
            # )
        info={'video_fps':self.frame_rate}
    
        audio_clip=None
        if self.use_audio:
            audio_decoder=AudioDecoder(video_path, sample_rate=self.sample_rate, num_channels=1)
            # video_clip.pts_seconds and video_clip.duration_seconds are of size (N,T) but for us, N=1 so we index it out by [0]
            # audio_clip has `data` tensor of size [Channels, Samples], of type float32 with values in range[-1,1], 
            # `pts` as floating number, `duration_seconds` as floating number
            # and `sample_rate` as int
            audio_clip=audio_decoder.get_samples_played_in_range(start_seconds=video_clip.pts_seconds[0][0].item(),
                                                                 stop_seconds=(video_clip.pts_seconds[0][-1]+\
                                                                               video_clip.duration_seconds[0][-1]).item())
            # We do not need code below since we ask torchcodec to return 1 channel
            # if audio_clip.data.shape[0]>1: audio=audio_clip.data.mean(dim=0, keepdim=True) # convert stereo signal to mono signal
            # else: audio=audio_clip.data # mono audio
            info['audio_fps']=audio_clip.sample_rate 
    
        return video_clip.data.squeeze(0), audio_clip.data if audio_clip is not None else None, info, video_idx

class UCF101(VisionDataset):
    """`UCF101 <https://www.crcv.ucf.edu/data/UCF101.php>`_ dataset.
    
    UCF101 is an action recognition video dataset. 
    This dataset consider every video as a collection of video clips of fixed size, specified by ``frames_per_clip``, where the step in frames
    between each clip is given by ``step_between_clips``. The dataset itself can be downloaded from the dataset website; annotations that 
    ``annotation_path`` should be pointing to can be downloaded from `here <https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip>`_.

    Args:
        root (str|Path): Root directory of the UCF101 dataset.
        annotation_path (str|Path): Path to the folder containing the split files
        frame_rate (float, optional): Desired frame rate in frame per second
        clip_duration (float, optional): How long each clip captures in second
        step_duration (float, optional): Distance between the start of each clip in seconds.
        train (bool, optional): If ``True``, create a dataset from the train spit, oetherwise from the ``test`` split
        transforms (callable, optional): A function/transform that takes in the video and annotation and returns the transformed versions
        decoder_transforms (list[callable, optional]): A list of decoder transformations see 
            https://meta-pytorch.org/torchcodec/stable/generated_examples/decoding/transforms.html
        metadata_path (str): Path to .pt file storing video clip metadata, including clip-start timestamp, seconds_per_frame, etc.
        fold (int): Data fold with options of 1, 2 or 3. Must specify since training fold1, fold2, and fold3 overlap and not worth training 
            all of them together
        sampling_type (str): Clip sampling type with options of 'regular' and 'random'
        use_audio (bool): Whether to return accompany audio data for each video data
        num_ffmpeg_threads (int): The number of threads to use for decoding. Use 1 for single-threaded decoding, use a higher number for multi-threaded 
            decoding, and use 0 lets FFmpeg decide on the number of threads. 
            see https://meta-pytorch.org/torchcodec/stable/generated/torchcodec.decoders.VideoDecoder.html#torchcodec.decoders.VideoDecoder
    Returns:
        (tuple): A 3-tuple with the following entries:
            - video (Tensor): A set of video frames with shape (T,C, H, W) where T is the number of video frames
            - audio (Tensor): A set of audio frames with shape (K,L) where K is the number of channels and L is the number of points, for this task
                we set K=1
            - label (int): Class of the video clip
            - video_idx (int): Index of video
    """
    def __init__(self, root: Union[str, Path], annotation_path: Union[str, Path], frame_rate:float=8, clip_duration:float=2, step_duration:float=1.7,
                 train:bool=True, fold:int=None, sampling_type:str='regular', use_audio:bool=False, transforms:Optional[Callable]=None, 
                 decoder_transforms:Optional[list[Callable]]=None, metadata_path:str=None, num_ffmpeg_threads:int=0 )->None:

        if not 1<=fold<=3: raise ValueError(f"Fold should be between 1 and 3, but got {fold}")
            
        super().__init__(root=root, transforms=transforms)
        
        extension=("avi",)
        self.train=train
        self.frame_rate=frame_rate
        self.clip_duration=clip_duration
        self.step_duration=step_duration
        self.classes, class_to_idx=find_classes(self.root)
        self.samples=make_dataset(self.root, class_to_idx, extension, is_valid_file=None)

        video_list=[x[0] for x in self.samples]
        video_clip_metadata=VideoClipMetadata(video_paths=video_list,clip_length_in_seconds=clip_duration, clip_stride_in_seconds=step_duration,
                                              frame_rate=frame_rate, sampling_type=sampling_type, use_audio=use_audio, metadata_path=metadata_path,
                                              num_ffmpeg_threads=num_ffmpeg_threads)
        # We bookkeep the full version of video clip metadata because we want to be able to return the metadata of full version rather than the
        # subset version of video clips
        self.full_video_clip_metadata=video_clip_metadata
        self.video_paths=video_clip_metadata.video_paths
        self.indices=self._select_fold(self.video_paths, annotation_path, fold, train)
        self.video_clip_metadata=self.full_video_clip_metadata.subset(self.indices)
        self.transforms=transforms
        self.decoder_transforms=decoder_transforms

    @property
    def metadata(self)->dict[str, Any]:
        """Return video clip metadata of the whole dataset"""
        return self.full_video_clip_metadata.metadata
        
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

    def __len__(self)->int: return self.video_clip_metadata.num_clips()

    def __getitem__(self, idx:int)->tuple[torch.Tensor, torch.Tensor, dict[str,Any], int]:
        """Get input video and target label"""
        
        video, audio, info, video_idx=self.video_clip_metadata.get_clip(idx, transforms=self.decoder_transforms)
        label=self.samples[self.indices[video_idx]][1]
        
        if self.transforms is not None: video=self.transforms(video)
        if self.video_clip_metadata.use_audio: return video, audio, label, video_idx, info
        return video, label, video_idx, info

if __name__ == '__main__':
    
    data_dirpath=Path('D:/data/UCF101')
    root=data_dirpath/'UCF-101'
    annotation_path=data_dirpath/'UCF101TrainTestSplits-RecognitionTask'
    metadata_path=data_dirpath/'metadata.pt'
    frame_rate=8
    clip_duration=2
    step_duration=1.7
    fold=1
    dataset=UCF101(root=root, annotation_path=annotation_path, frame_rate=frame_rate, clip_duration=clip_duration, step_duration=step_duration,
                   train=True, metadata_path=metadata_path, fold=fold, sampling_type='random', use_audio=True) 
    video, audio, label, video_idx, info=dataset[0]
    print(f"{video.shape=}, {video.dtype=}, {audio.shape=}, {audio.dtype=}")
    print(f"{info=}, {label=}, {video_idx=}")

    from torchcodec.encoders import VideoEncoder
    encoder=VideoEncoder(frames=video, frame_rate=info['video_fps']) # frame_rate is the frame rate of input video
    encoded_frames=encoder.to_tensor(format='mp4')
    # play_video(encoded_frames)
    # play_audio(audio, rate=info['audio_fps'])
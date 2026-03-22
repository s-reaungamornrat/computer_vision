import os

import torch
from torchvision import transforms

import numpy as np
from PIL import Image

from .loader import get_video_loader, get_image_loader
from .transforms import GroupMultiScaleCrop, Stack, ToTorchFormatTensor, GroupNormalize
from .masking_generator import TubeMaskingGenerator, RunningCellMaskingGenerator

# from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
IMAGENET_DEFAULT_MEAN=(0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD=(0.229, 0.224, 0.225)

class DataAugmentationForVideoMAEv2(object):

    def __init__(self, args, div=True, roll=False, input_mean=IMAGENET_DEFAULT_MEAN, input_std=IMAGENET_DEFAULT_STD, 
                 scales=[1., .875, .75, .66]):
        
        self.input_mean=input_mean
        self.input_std=input_std
        normalize=GroupNormalize(self.input_mean, self.input_std)
        self.train_augmentation=GroupMultiScaleCrop(args.input_size, scales)
        self.transform=transforms.Compose([self.train_augmentation, Stack(roll=roll), ToTorchFormatTensor(div=div), normalize])
        if args.mask_type=='tube': self.encoder_mask_map_generator=TubeMaskingGenerator(args.window_size, args.mask_ratio)
        else: raise NotImplementedError("Unsupported encoder masking strategy type")

        if args.decoder_mask_ratio>0.:
            if args.decoder_mask_type=='run_cell':
                self.decoder_mask_map_generator=RunningCellMaskingGenerator(args.window_size, args.decoder_mask_ratio)
            else: raise NotImplementedError("Unsupported decoder masking strategy type")

    def __call__(self, images):
        process_data, _=self.transform(images)
        encoder_mask_map=self.encoder_mask_map_generator()
        if hasattr(self, 'decoder_mask_map_generator'): decoder_mask_map=self.decoder_mask_map_generator()
        else: decoder_mask_map=1-encoder_mask_map
        return process_data, encoder_mask_map, decoder_mask_map

    def __repr__(self):
        repr="(DataAugmentationForVideoMAEv2,\n"
        repr+=f" transform={str(self.transform)},\n"
        repr+=f" Encoder Masking Generator= {str(self.encoder_mask_map_generator)},\n"
        if hasattr(self, 'decoder_mask_map_generator'): repr+=f" Decoder Masking Generator={self.decoder_mask_map_generator},\n"
        else: repr+=" Do not use decoder masking,\n"
        repr+=")"
        return repr

class HybridVideoMAE(torch.utils.data.Dataset):
    """Load your own videomae pretraining dataset
    Args:
        root (str): Path to the root folder storing the dataset
        setting (str): A text file descrining the dataset, each line per video sample. There are four items in each line
            (1) video path; (2) start_idx; (3) total frames; and (4) video label 
            for pretrain video data, if total frames<0, start_idx and video label meaningless
            for pretrain rawframe data, video label meaningless
        train (bool): Whether to load the training or validation set
        test_mode (bool): Whether to perform evaluation on the test set
        name_pattern (str): Name pattern of teh decoded video frame. Default: 'img_{:05}.jpg' (e.g., img_00012.jpg)
        video_ext (str): Video format if video_loader is set to True. Default: 'mp4'
        is_color (bool): Whether the loaded images are color or grayscale
        modality (str): Input modalities. Only support 'rgb' for now. Supports for rgb difference images and optical flow images will be added later.
            Default: 'rgb'
        num_segments (int): Number of segments to evenly divide the video into clips. A useful technique to obtain global video-level information. See
            Limin Wang, et al. Temporal Segment Networks: Towards Good Practices for Deep Action Recognition, ECCV 2016.
        num_crop (int): Number of crops for each image. Common choices are 3 and 10 crops during evaluation. Default is 1.
        num_length (int): The length of input video clip. Default is a single image, but it can be multiple video frames. For example, new_length=16 means
            we will extract a video clip of consecutive 18 frames. 
        new_step (int): Temporal sampling rate. For example, new_step=1 means we will extract a video clip of consecutive frames. new_steps=2 means we will
            extract a video clip of every other frame
        transform (callable): A function that takes data and label and transforms them
        temporal_jitter (bool): Whether to temporally jitter if new_step>1
        lazy_init (bool): If True, build a dataset instance without loading any dataset
        num_sample (int): Number of sampled views for repeated augmentation
    Reference: https://github.com/OpenGVLab/VideoMAEv2/blob/master/dataset/pretrain_datasets.py#L75
    """
    def __init__(self, root, setting, train=True, test_mode=False, name_pattern="img_{:05}.jpg", video_ext='mp4', is_color=True, modality='rgb', num_segments=1,
                num_crop=1, new_length=1, new_step=1, transform=None, temporal_jitter=False, lazy_init=False, num_sample=1):
    
        super(HybridVideoMAE, self).__init__()
        self.root=root
        self.setting=setting
        self.train=train
        self.test_mode=test_mode
        self.is_color=is_color
        self.modality=modality
        self.num_segments=num_segments
        self.num_crop=num_crop
        self.new_length=new_length # number of frames per an extracted video clip
        # temporal stride/dilate or frame sampling step, e.g., 4 for picking every 4th frames. Large new_step covers more time but might miss fast motions
        self.new_step=new_step 
        # total temporal span of a single sampled clip
        self.skip_length=self.new_length*self.new_step
        self.temporal_jitter=temporal_jitter
        self.name_pattern=name_pattern
        self.video_ext=video_ext
        self.transform=transform
        self.lazy_init=lazy_init
        self.num_sample=num_sample

        self.orig_new_step=new_step
        self.orig_skip_length=self.skip_length
        
        self.video_loader=get_video_loader()
        self.image_loader=get_image_loader()

        if not self.lazy_init:
            self.clips=self._make_dataset(root, setting)
            if len(self.clips)==0: raise RuntimeError("Found 0 video clip in subfolders of: "+root+"\nCheck your data directory")

    def _make_dataset(self, root, setting):
        """
        Read annotation file and return a list of tuples of video paths and labels
        Args:
            root (str): Path to video files
            settings (str): Path of annotation file listing (relative) paths to video and their labels
        Returns:
            (list[tuple[str, int]]): List of tuples of video paths and their labels
        """ 
        assert os.path.exists(setting), f"Setting file {setting} does not exist"
        
        clips=[]
        with open(setting) as split_f:
            data=split_f.readlines()
        
        for line in data:
            line_info=line.strip().split(' ')
            # video path label
            video_path=os.path.join(root, line_info[0]) if root is not None else line_info[0]
            label=int(line_info[1])
            clips.append((video_path, label))
        return clips

    def _sample_train_indices(self, num_frames):
        """ Compute `offset` which defines the starting frame positions for each temporal segment and `skip_offsets` based on temporal stride `new_step`
        to determine frame indices to extract as `frame_id=offset + i*new_step + skip_offsets[i]`. For example, offset=30, new_step=2 and
        skip_offsets=[1,0,1,1,1], give frame_id=[31,32,35,37,39].
        
        The originial code return `offsets+1` which we believe to convert frame indices from 0-based to 1-based indexing, but we return 0-based
        frame indexing
        
        Args:
            num_frames (int): Number of total frames
        Returns:
            (np.ndarray): Indices of the first frames of each clip, whose length is `num_segments`
            (np.ndarray): Random jitter whose extent is within `new_step` (temporal stride) and whose length is `new_length`
        """
        # self.skip_length: total temporal span of a single sampled clip
        # self.num_segments: number of segments to evenly divide the video into clips, default to 1
        # number of frames available in each segment
        average_duration=(num_frames-self.skip_length+1)//self.num_segments # in a unit of frames 
        # num_frames-self.skip_length prevent us from starting the video clip too late and we'll run out of frames
        
        # compute `offsets` which are the starting frame positions for each temporal segment
        if average_duration>0:
            offsets=np.arange(self.num_segments)*average_duration
            # randon jitter or stochastic sampling of the first frame of each segment
            offsets=offsets+np.random.randint(average_duration, size=self.num_segments)
        elif num_frames>max(self.num_segments, self.skip_length):
            offsets=np.sort(np.random.randint(num_frames-self.skip_length+1, size=self.num_segments))
        else: offsets=np.zeros((self.num_segments,))
        if self.temporal_jitter:
            # self.new_step: temporal sampling rate/stride
            skip_offsets=np.random.randint(self.new_step, size=self.skip_length//self.new_step)
        else: skip_offsets=np.zeros(self.skip_length//self.new_step, dtype=int)
        # Legacy video loaders expect frame numbers starting from 1, not 0
        #return offsets+1, skip_offsets
        return offsets, skip_offsets

    def get_frame_id_list(self, duration, indices, skip_offsets):
        """
        Get a list of frame indices to be extracted
        Args:
            duration (int): Total number of frames of this video
            indices (np.ndarray): Indices of the first frames of each clip, whose length is `num_segments`
            skip_offsets (np.ndarray): Random jitter whose extent is within `new_step` (temporal stride) and whose length is `new_length`==`num_frames`
        Returns:
            (list[np.int64]): List of frame indices to be extracted, whose length is `new_length`==`num_frames`
        """
        frame_id_list=[]
        for seg_ind in indices:
            offset=int(seg_ind)
            # self.skip_length: total temporal span of a single sampled clip
            # self.new_step: temporal stride of each frame
            # iterate for self.skip_length//self.new_step times.  For example, self.skip_length=64, self.new_step=4, will iterate for 16 times
            for i, _ in enumerate(range(0, self.skip_length, self.new_step)): 
                if offset+skip_offsets[i]<=duration: frame_id=offset+skip_offsets[i]-1
                else: frame_id=offset-1
                frame_id_list.append(frame_id)
                if offset+self.new_step<duration: offset+=self.new_step
        return frame_id_list

    def __getitem__(self, index):
        """
        Returns:
            (torch.Tensor): Processed video frames of shape (C*T, H, W) where C is the number of channels and T is the number of frames
            (np.ndarray): Encoder mask of shape (Tg,Hg*Wg) where Tg is the grid dimension along frame dimension, and Hg and Wg is the grid 
                dimension along height and width. Note Tg=T/Tp, Hg=H/Hp, and Wg=W/Wp where Tp,Hp,Wp is the patch size
            (np.ndarray): Decoder mask of shape (Tg,Hg*Wg) where Tg is the grid dimension along frame dimension, and Hg and Wg is the grid 
                dimension along height and width. Note Tg=T/Tp, Hg=H/Hp, and Wg=W/Wp where Tp,Hp,Wp is the patch size
        """
        try:
            video_name, label=self.clips[index]
            self.skip_length=self.orig_skip_length
            self.new_step=self.orig_new_step
            
            decord_vr=self.video_loader(video_name)
            duration=len(decord_vr) #number of total frames
            segment_indices,skip_offsets=self._sample_train_indices(duration)
            frame_id_list=self.get_frame_id_list(duration, segment_indices, skip_offsets)
            video_data=decord_vr.get_batch(frame_id_list).asnumpy() # (num_frames, H, W, C) where C is the number of channels
            images=[Image.fromarray(video_data[vid,]) for vid in range(len(frame_id_list))] # convert each frame to PIL.Image
        except Exception as e:
            print(f"Failed to load video from {video_name} with error {e}")
            index=np.random.randint(0, len(self.clips))
            return self.__getitem__(index)
        
        if self.num_sample>1: # get multiple augmented clips from the same extracted video clip
            process_data_list, encoder_mask_list, decoder_mask_list=[],[],[]
            for _ in range(self.num_sample):
                process_data, encoder_mask, decoder_mask=self.transform((images, None))
                process_data=process_data.view((self.new_length, 3)+process_data.size()[-2:]).transpose(0,1) # (T,C,H,W)->(C,T,H,W)
                process_data_list.append(process_data)
                encoder_mask_list.append(encoder_mask)
                decoder_mask_list.append(decoder_mask)
            return process_data_list, encoder_mask_list, decoder_mask_list
        else:
            process_data, encoder_mask, decoder_mask=self.transform((images, None))
            # (T*C,H,W)->(T,C,H,W)->(C,T,H,W)
            process_data=process_data.view((self.new_length,3)+process_data.size()[-2:]).transpose(0,1)
            return process_data, encoder_mask, decoder_mask

    def __len__(self): return len(self.clips)
import os
import warnings

import numpy as np

import torch

from .loader import get_video_loader
from .pretrained_datasets import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .video_transforms import random_short_side_scale_jitter, random_crop, random_resized_crop, random_resized_crop_with_shift, horizontal_flip, uniform_crop

class VideoClsDataset(torch.utils.data.Dataset):
    """Load video classification dataset"""
    def __init__(self, anno_path, data_root='', mode='train', clip_len=8, frame_sample_rate=2, crop_size=224, short_side_size=256, new_height=256, new_width=340,
                 keep_aspect_ratio=True, num_segment=1, num_crop=1, test_num_segment=10, test_num_crop=3, sparse_sample=False,args=None):
        self.anno_path=anno_path
        self.data_root=data_root
        self.mode=mode
        self.clip_len=clip_len
        self.frame_sample_rate=frame_sample_rate
        self.crop_size=crop_size
        self.short_side_size=short_side_size
        self.new_height=new_height
        self.new_width=new_width
        self.keep_aspect_ratio=keep_aspect_ratio
        self.num_segment=num_segment
        self.test_num_segment=test_num_segment
        self.num_crop=num_crop
        self.test_num_crop=test_num_crop
        self.sparse_sample=sparse_sample
        self.args=args
        self.aug=False
        self.rand_erase=False

        if self.mode in ['train']:
            self.aug=True
            if self.args.reprob>0: self.rand_erase=True

        self.video_loader=get_video_loader()
        
        self.dataset_samples, self.label_array=self._make_dataset(root=data_root, anno_path=anno_path)

        if mode=='validation':
            raise NotImplementedError('Implememt me, see https://github.com/OpenGVLab/VideoMAEv2/blob/master/dataset/datasets.py#L16')
        elif mode=='test':
            raise NotImplementedError('Implememt me, see https://github.com/OpenGVLab/VideoMAEv2/blob/master/dataset/datasets.py#L16')
            
    def _make_dataset(self, root, anno_path):
        """
        Read annotation file and return a list of tuples of video paths and labels
        Args:
            root (str): Path to video files
            settings (str): Path of annotation file listing (relative) paths to video and their labels
        Returns:
            (list[str]): List of video paths 
            (list[int]): List of labels
        """ 

        assert os.path.exists(anno_path), f"Setting file {anno_path} does not exist"
        
        with open(anno_path) as split_f:  data=split_f.readlines()
        
        dataset_samples, label_array=[],[]
        for line in data:
            line_info=line.strip().split(' ')
            # video path label
            dataset_samples.append( os.path.join(root, line_info[0]) if root is not None else line_info[0] )
            assert os.path.isfile(dataset_samples[-1]), f"{dataset_samples[-1]} does not exist"
            label_array.append( int(line_info[1]) )
        assert len(dataset_samples)==len(label_array)
        return dataset_samples, label_array

    def _load_test_video(self, vr):
        """
        Load and sample test video frames
        Args:
            vr (decord.video_reader.VideoReader): Video reader
        Returns:
            (np.ndarray): Sampled video frame of shape (T,H,W,C) and of dtype uint8, where T is the number of frames and C is the channels, e.g., 3 for RGB
        """
        length=len(vr)
        # Select frames to extract
        if self.sparse_sample: # sparse sampling for multi-view testing, ensuring selected frames are spread out across the whole timeline
            tick=length/float(self.num_segment) # divide the entire video into equal-sized chunks (segments); thus, tick=size of each segment
            all_index=[]
            for t_seg in range(self.test_num_segment): # loop over test segments to create multiple temporal test clips (like test-time augmentation)
                # for each segment x, pick on frame and slighly shifts the sampling position by offset 
                # t_seg*tick/dataset_train.test_num_segment so each t_seg produce a different temporal view
                tmp_index=[int(t_seg*tick/self.test_num_segment+tick*x) for x in range(self.num_segment)]
                all_index.extend(tmp_index)
            all_index=list(np.sort(np.array(all_index))) # of length dataset_train.num_segment*dataset_train.num_segment frames
        else: # dense/fixed sampling
            all_index=list(range(0, length, self.frame_sample_rate))  # of length dataset_train.clip_len
            while len(all_index)<self.clip_len: all_index.append(all_index[-1]) # repeat the last frame index until reaching the required length
        vr.seek(0)
        buffer=vr.get_batch(all_index).asnumpy()
        return buffer

    def _load_train_val_video(self,vr,sample_rate_scale=1):
        """
        Load and sample test video frames
        Args:
            vr (decord.video_reader.VideoReader): Video reader
            sample_rate_scale (int): Sampling step  
        Returns:
            (np.ndarray): Sampled video frame of shape (T,H,W,C) and of dtype uint8, where T is the number of frames and C is the channels, e.g., 3 for RGB
        """
        length=len(vr)
        
        # clip_len: number of frames per clip
        # frame_sample_rate: temporal stride bwteen sampled frame
        # converted_len: temporal span in the original video needed to produce one clip, in unit of frames
        converted_len=int(self.clip_len*self.frame_sample_rate)
        # seg_len: search area/bucket size for each individual frame selection
        seg_len=length//self.num_segment
        
        all_index=[]
        for i in range(self.num_segment):
            if seg_len<=converted_len:
                # where seg_len // self.frame_sample_rate is the number of frames that can be sampled from the bucket
                index=np.linspace(0, seg_len, num=seg_len//self.frame_sample_rate)
                index=np.concatenate( # pad index with last index
                    ( index, np.ones(self.clip_len - seg_len//self.frame_sample_rate)*(seg_len-1) )
                )
                index=np.clip(index, 0, seg_len-1).astype(np.int64)
            else:
                if self.mode=='validation': end_idx=(converted_len+seg_len)//2
                else: end_idx=np.random.randint(converted_len, seg_len)
                str_idx=end_idx-converted_len
                index=np.linspace(str_idx, end_idx, num=self.clip_len)
                index=np.clip(index, str_idx, end_idx-1).astype(np.int64)
            index=index+i*seg_len
            all_index.extend(list(index))

        all_index=all_index[::int(sample_rate_scale)] # sampling
        vr.seek(0)
        buffer=vr.get_batch(all_index).asnumpy()
        return buffer

    def load_video(self,sample,sample_rate_scale=1):
        """Load and sample video frames
        Args:
            sample (str|Path): Path to video file
            sample_rate_scale (int): Secondary temporal stride to further sampling frames
        Returns:
            (np.ndarray): Sampled video frame of shape (T,H,W,C) and of dtype uint8, where T is the number of frames and C is the channels, e.g., 3 for RGB
        """
        try: vr=self.video_loader(sample)
        except Exception as e: 
            print(f"Failed to load video from {sample} with error {e}!")
            return []

        if self.mode=='test': buffer=self._load_test_video(vr)
        else: buffer=self._load_train_val_video(vr,sample_rate_scale=sample_rate_scale)
            
        return buffer

    def __len__(self):
        if self.mode!='test': return len(self.dataset_samples)
        return len(self.test_dataset)


def tensor_normalize(tensor, mean, std):
    """Normalize a given tensor by subtracting the mean and dividing by std
    Args:
        tensor (torch.Tensor): Tensor to be normalized of shape (T,C,...)
        mean (sequence): Mean of image intensity after normalizing to range [0,1] as a sequence of values for each channel
        std (sequence): Standard deviation of image intensity after normalizing to range [0,1] as a sequence of values for each channel
    Return:
        (torch.Tensor): Tensor after normalizing by mean and std
    """
    if tensor.dtype==torch.uint8:
        tensor=tensor.float()
        tensor/=255.
    if isinstance(mean, (list, tuple)): 
        mean=torch.tensor(mean)
        shape=(1,len(mean))+(1,)*(tensor.ndim-2)
        mean=mean.view(*shape)
    if isinstance(std, (list, tuple)): 
        std=torch.tensor(std)
        shape=(1,len(std))+(1,)*(tensor.ndim-2)
        std=std.view(*shape)
    
    tensor=(tensor-mean)/std
    return tensor

def spatial_sampling(frames, spatial_idx=-1, min_scale=256, max_scale=320, crop_size=224, random_horizontal_flip=True,
                     inverse_uniform_sampling=False, aspect_ratio=None, scale=None, motion_shift=False):
    """Perform spatial sampling on the given video frames. If spatial_idx is -1, perform random scale, random crop, and random flip on the given frames.
    If spatial_idx is 0, 1,or 2, perform spatial uniform sampling with the given spatial_idx
    Args:
        frames (tensor): Frames of images sampled from the video. The dimension is (T,C,H,W) where C is the number of image channels and T is the number 
            of frames
        spatial_idx (int): If -1, perform random spatial sampling. If 0,1,or 2, perform left, center, right crop of width is larger than height, and
            perform top, center, buttom crop if height is larger than width
        min_scale (int): Minimum size of scaling
        max_scale (int): Maximum size of scaling
        crop_size (int): Size of height and width used to crop the frames
        inverse_uniform_sampling (bool): If True, sample uniformly in [1/max_scale, 1/min_scale] and take a reciprocal to get the scale. If False,
            take a uniform sample from [min_scale, max_scale]
        aspect_ratio (list): Aspect ratio (width/height) range for resizing 
        scale (list): Scale range for resizing
        motion_shift (bool): Whether to apply motion shift for resizing
    Returns:
        (torch.Tensor): Spatially sampled frames
    """
    
    assert spatial_idx in [-1,0,1,2]
    if spatial_idx==-1:
        if all(x is None for x in [aspect_ratio, scale]):
            frames,_=random_short_side_scale_jitter(images=frames, min_size=min_scale, max_size=max_scale, boxes=None, 
                                                    inverse_uniform_sampling=inverse_uniform_sampling)
            frames,_=random_crop(frames, crop_size) # (T,C,H,W) where H=W=crop_size
        else:
            transform_func=random_resized_crop_with_shift if motion_shift else random_resized_crop
            frames=transform_func(images=frames, target_height=crop_size, target_width=crop_size, scale=scale, ratio=aspect_ratio)
        if random_horizontal_flip:
            frames,_=horizontal_flip(0.5, frames)
    else:
        # The testing is deterministic and no jitter should be performed. min_scale, max_scale, and crop_size are expected to be the same
        assert len({min_scale, max_scale, crop_size})==1
        frames,_=random_short_side_scale_jitter(images=frames, min_size=min_scale, max_size=max_scale)
        frames,_=uniform_crop(frames, crop_size, spatial_idx)
    return frames

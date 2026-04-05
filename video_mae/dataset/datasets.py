import os
import warnings

import numpy as np

import torch
from torchvision import transforms

from .loader import get_video_loader
from .pretrained_datasets import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .video_transforms import (random_short_side_scale_jitter, random_crop, random_resized_crop, random_resized_crop_with_shift, horizontal_flip, 
                               uniform_crop, create_random_augment, Compose, Resize, CenterCrop, ClipToTensor, Normalize)
from .random_erasing import RandomErasing

class VideoClsDataset(torch.utils.data.Dataset):
    """Load video classification dataset"""
    def __init__(self, anno_path, data_root='', mode='train', clip_len=8, frame_sample_rate=2, crop_size=224, short_side_size=256, new_height=256, 
                 new_width=340, keep_aspect_ratio=True, num_segment=1, num_crop=1, test_num_segment=10, test_num_crop=3, sparse_sample=False,args=None):
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
            self.data_transform=Compose([Resize(self.short_side_size, interpolation='bilinear'),
                                         CenterCrop(size=(self.crop_size, self.crop_size)),
                                         ClipToTensor(),
                                         Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)])
        elif mode=='test':
            self.data_resize=Resize(size=self.short_side_size,interpolation='bilinear')
            self.data_transfrom=Compose([ClipToTensor(), Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)])
            # prepare for multi-view inference or ensemble testing
            self.test_seg,self.test_dataset,self.test_label_array=[],[],[]
            for ck in range(self.test_num_segment): # temporal segments: different time windows or clips sampled from the video
                for cp in range(self.test_num_crop): # spatial crops: different spatial view (e.g., left-crop, center-crop, right-crop, or flips)
                    for idx in range(len(self.label_array)): # dataset samples: original list of video paths and their labels
                        sample_label=self.label_array[idx]
                        # below has the size of cartesian product of (segments x crops x total_videos), __getitem__ can iterate through them as unique indices
                        self.test_label_array.append(sample_label)
                        self.test_dataset.append(self.dataset_samples[idx])
                        # instruction manual storing `segment index` (ck) and `crop index` (cp)
                        self.test_seg.append((ck, cp)) 
            
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
             # note label was count from 1 so we subtract 1 to start from 0
            label_array.append( int(line_info[1])-1 )
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

    def _aug_frame(self, buffer, args):
        """Augment video frames and reshape/crop so their shape match desired input size (`crop_size`)
        Args:
            buffer (np.ndarray): Video frames of shape (T,H,W,C) of type np.uint8 where T is the number of frames and C is the number of channels
        Returns:
            (torch.Tensor): Transformed video frames of shape (C,T,H,W) of type torch.float32
        """
        buffer=[transforms.ToPILImage()(frame) for frame in buffer] # list of (H,W,C)
        
        aug_transform=create_random_augment(input_size=(self.crop_size, self.crop_size),
                                            auto_augment=args.aa, interpolation=args.train_interpolation)
        buffer=aug_transform(buffer) # PIL.Image with image.size (W,H) and np.array(x).shape of (H,W,d)
        buffer=[transforms.ToTensor()(img) for img in buffer]
        buffer=torch.stack(buffer) # (T,C,H,W)
        
        buffer=tensor_normalize(buffer, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD) # (T,C,H,W)
        
        # Perform data augmentation
        scale, aspect_ratio=([0.08,1.0],[0.75,1.3333])
        buffer=spatial_sampling(buffer, spatial_idx=-1, min_scale=256, max_scale=320, crop_size=self.crop_size, 
                                random_horizontal_flip=True, inverse_uniform_sampling=False, aspect_ratio=aspect_ratio, 
                                scale=scale, motion_shift=False)
        
        if self.rand_erase:
            erase_transform=RandomErasing(args.reprob, mode=args.remode, max_count=args.recount, num_splits=args.recount, device='cpu')
            buffer=erase_transform(buffer) # (T,C,H,W)
            buffer=buffer.permute(1,0,2,3).contiguous() # (C,T,H,W)
        return buffer

    
    def _get_train_item(self, index, args, scale_t=1):
        """Get training item
        Args:
            index (int): Index of item to get
            args (Namespace): Input arguments
            scale_t (int): Secondary temporal stride to further sampling frames
        Returns:
            (list[torch.Tensor] | torch.Tensor): List of (C,T,H,W) video frames or (C,T,H,W) video frames
            (list[int] | int): List of labels or label
            (list[int] | int): List of index or index
            (set): ???
        """ 

        sample=self.dataset_samples[index] # video filepath
        # (T,H,W,C)
        buffer=self.load_video(sample, sample_rate_scale=scale_t) # read video 
        if len(buffer)==0:
            while len(buffer)==0:
                warnings.warn(f"Video {sample} was not loaded correctly during training")
                index=np.random.randint(len(self))
                sample=self.dataset_samples[index]
                buffer=self.load_video(sample, sample_rate_scale=scale_t)
                
        if args.num_sample>1:
            frame_list, label_list,index_list=[],[],[]
            for _ in range(args.num_sample):
                new_frames=self._aug_frame(buffer, args)
                label=self.label_array[index]
                frame_list.append(new_frames)
                label_list.append(label)
                index_list.append(index)
            return frame_list, label_list, index_list, {}
            
        buffer=self._aug_frame(buffer, args)
        return buffer, self.label_array[index], index, {}  

    def _get_val_item(self, index):
        """Get validation item
        Args:
            index (int): Index of item to get
        Returns:
            (torch.Tensor): (C,T,H,W) video frames
            (int): Label
            (str): Name of video file
        """ 
        sample=self.dataset_samples[index]
        buffer=self.load_video(sample)
        if len(buffer)==0:
            while len(buffer)==0:
                warnings.warn(f"video {sample} not correctly loaded during validation")
                index=np.random.randint(self.__len__())
                sample=self.dataset_samples[index]
                buffer=self.load_video(sample)
        buffer=self.data_transform(buffer)
        return buffer, self.label_array[index], os.path.splitext(os.path.basename(sample))[0]

    def _get_test_item(self,index):
        """Get test item
        Args:
            index (int): Index of item to get
        Returns:
            (torch.Tensor): (C,T,H,W) video frames
            (int): Label
            (str): Name of video file
            (int): Temporal sample index
            (int): Spatial crop index
        """ 
        sample=self.test_dataset[index]
        chunk_nb,split_nb=self.test_seg[index] # temporal segment index, and spatial crop index
        buffer=self.load_video(sample)
        
        while len(buffer)==0:
            warnings.warn(f"video {self.test_dataset[index]}, temporal {chunk_nb}, spatial {split_nb} not found during testing")
            index=np.random.randint(self.__len__())
            sample=self.test_dataset[index]
            chunk_nb, split_nb=self.test_seg[index]
            buffer=self.load_video(sample)
        
        buffer=self.data_resize(buffer) # list of (H,W,C) np.ndarray of type uint8
        if isinstance(buffer, list): buffer=np.stack(buffer, 0) # (T,H,W,C)

        # Below we perform temporal and spatial sample
        
        # Calculate a stride (step size) to evenly distribute a specific number of crops across the longer dimension
        # self.short_side_size is the target crop size (e.g., 224 pixel)
        # spatial_step: distance between the starting coordinates of each consecutive crop
        # example: A video of 324 pixels wide and target crop of 224 pixels, using 3 crop
        # - excess space: 324-224=100 pixels
        # - intervals: 3-1=2
        # - spatial_step= 100/2=50
        # crops would start at horizontal offsets of 0 (left), 50 (center), 100 (right-- 100+224=324 which is the edge)
        spatial_step=1.*(max(buffer.shape[1:3])-self.short_side_size)/(self.test_num_crop-1)
        spatial_start=int(split_nb*spatial_step) # split_nb: index of spatial crops
        if self.sparse_sample: # sparse in time
            # chunk_nb ranging from [0, self.test_num_segment-1]
            # the following allow us to sample every self.test_num_segment frame
            # Example: a video with 16 frames, self.test_num_segment=4, we have
            # chunk_nb     slice syntax       frames selected
            # 0             0::4              0,4,8,12
            # 1             1::4              1,5,9,13
            # 2             2::4              2,6,10,14
            # 3             3::4              3,7,11,15
            temporal_start=chunk_nb # chunk_nb:index of temporal segment sampled , temporal_start: index to frame the video reader should start decoding from
            if buffer.shape[1]>=buffer.shape[2]: # if H>=W
                buffer=buffer[temporal_start::self.test_num_segment, # sample T from temporal_start with the step of dataset_test.test_num_segment
                              spatial_start:spatial_start+self.short_side_size] # sample along H dimension
            else:
                buffer=buffer[temporal_start::self.test_num_segment,:,
                              spatial_start:spatial_start+self.short_side_size]
        else:
            temporal_step=max(1.*(buffer.shape[0]-self.clip_len)/(self.test_num_segment-1), 0)
            temporal_start=int(chunk_nb*temporal_step)
            if buffer.shape[1]>=buffer.shape[2]: # if H>=W
                buffer=buffer[temporal_start:temporal_start+self.clip_len, 
                              spatial_start:spatial_start+self.short_side_size]
            else:
                buffer=buffer[temporal_start:temporal_start+self.clip_len,:,
                              spatial_start:spatial_start+self.short_side_size]
        
        # After the above operation, buffer will be (T,H,W,C) np.ndarray where H and W will be at target desired size
        buffer=self.data_transfrom(buffer) # (C,T,H,W) float32 tensor after [0,1] followed ImageNet normalization
        return buffer, self.test_label_array[index], os.path.splitext(os.path.basename(sample))[0], chunk_nb, split_nb

    def __getitem__(self, index):
        """Get item. See `_get_train_item`, `_get_val_item`, and `_get_test_item` for docstring of each mode
        Args:
            index (int): Index of item to get
        """
        if self.mode=='train': 
            # outputs = frames_list, label_list, index_list, some_set
            outputs=self._get_train_item(index=index, args=self.args,  scale_t=1)
        elif self.mode=='validation': 
            # outputs = frames, label, fname
            outputs=self._get_val_item(index=index)
        elif self.mode=='test': 
            # outputs= frames, label, fname, chunk_nb, split_nb
            outputs=self._get_test_item(index=index)
        else: raise ValueError(f"mode {self.mode} is not supported, must be 'test', 'validation', 'train'")

        return outputs

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

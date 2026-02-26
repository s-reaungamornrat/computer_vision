from __future__ import annotations
from typing import Optional, Union

import os
import numpy as np

class DecordInit:
    """Using decord to initialize the video_reader.

    Decord: https://github.com/dmlc/decord

    Required Keys:
        - filename
    Added Keys:
        - video_reader
        - total_frames
        - fps
    Args:
        io_backend (str): IO backend where frames are store. Default to 'disk'. Actually, we do not need this anymore since we assume always read from file
        num_threads (int): Number of thread to decode the video. Default to 1
        kwargs (dict): Keyword arguments for file client
    """
    def __init__(self, io_backend:str='disk', num_threads:int=1, **kwargs)->None:
        self.io_backend=io_backend
        self.num_threads=num_threads
        self.kwargs=kwargs
        #self.file_client=None # we do not need this. We read from disk on local machine

    def __call__(self, results:dict)->Optional[Union[dict, tuple[list,list]]]:
        return self.transform(results)

    def _get_video_reader(self, filename:str)->decord.video_reader.VideoReader:
        if os.path.splitext(filename)[0]==filename: filename=filename+'.mp4'
        try: import decord
        except ImportError: raise ImportError('Please run "pip install decord" to install Decord first')
        container=decord.VideoReader(filename, num_threads=self.num_threads)
        return container

    def transform(self, results:dict)->dict:
        """Perform the Decord initialization
        Args:
            results (dict): The result dict
        Returns:
            (dict): The result dict
        """
        container=self._get_video_reader(results['filename'])
        results['total_frames']=len(container)
        results['video_reader']=container
        results['avg_fps']=container.get_avg_fps()
        return results

    def __repr__(self)->str:
        return f"{self.__class__.__name__}(io_backend={self.io_backend},num_threads={self.num_threads})"

class SampleFrames:
    """Sample frames from the video
    
    Required Keys:
        - total_frames
        - start_index
    Added Keys:
        - frame_inds
        - frame_interval
        - num_clips
    Args:
        clip_len (int): Number of frames of each sampled output clips
        frame_interval (int): Temporal interval of adjacent sampled frames. Default to 1
        num_clips (int): Number of clips to be sampled. Default to 1
        temporal_jitter (bool): Whether to apply temporal jittering. Default to False
        twice_sample (bool): Whether to use twice sample when testing. If set to True, it will sample frames with and without fixed shift,
            which is commonly used for testing in TSM model. Default to False
        out_of_bound_opt (str): The way to deal with out of bounds frame indices. Available options are 'loop', 'repeat_last'. Default to 'loop'
        test_mode (bool): Whether to build test or validation dataset
        keep_tail_frames (bool): Whether to keeo tail frames when sampling. Default to False
        target_fps (optional, int): Convert input videos with arbitrary frame rates to the unified target FPS before sampling frames. If None, 
            the frame rate will not be adjusted. Default to None
    """
    def __init__(self, clip_len:int, frame_interval:int=1, num_clips:int=1, temporal_jitter:bool=False, twice_sample:bool=False,
                 out_of_bound_opt:str='loop', test_mode:bool=False, keep_tail_frames:bool=False, target_fps:Optional[int]=None,
                **kwargs)->None:
        self.clip_len=clip_len
        self.frame_interval=frame_interval
        self.num_clips=num_clips
        self.temporal_jitter=temporal_jitter
        self.twice_sample=twice_sample
        self.out_of_bound_opt=out_of_bound_opt
        self.test_mode=test_mode
        self.keep_tail_frames=keep_tail_frames
        self.target_fps=target_fps
        assert self.out_of_bound_opt in ['loop', 'repeat_last']
        
    def __call__(self, results:dict)->Optional[Union[dict, tuple[list,list]]]:
        return self.transform(results)
    
    def _get_ori_clip_len(self, fps_scale_ratio:float)->float:
        """Calculate length of clip segment for different strategy. It computes the original temporal length in frames
        In other words, this function answers how long (in original frame index space) does one sampled clip span, or 
        how many frames in the original video are consumed by a single sampled clip
        
        Args:
            fps_scale_ratio (float): Scale ratio to adjust fps
        """
        if self.target_fps is not None:
            # align test sample strategy with `PySlowFast` repo
            ori_clip_len=self.clip_len*self.frame_interval # total span of original clip in number of frames
            ori_clip_len=np.maximum(1, ori_clip_len*fps_scale_ratio)
        elif self.test_mode: ori_clip_len=(self.clip_len-1)*self.frame_interval+1 # strict indexing
        else: ori_clip_len=self.clip_len*self.frame_interval # training mode without target_fps
        return ori_clip_len

    def _get_test_clips(self, num_frames:int, ori_clip_len:float)->np.ndarray:
        """Get clip offsets in test mode, where clip offsets are the start positions of each clip
        
        If the total number of frames is not enough, it will return all zero indices
        
        Args:
            num_frames (int): Total number of frame in the video
            ori_clip_len (float): Length of original sample clip
        Returns:
            (np.ndarray): Sampled frame indices in test mode
        """
        if self.clip_len==1: # 2D recognizer
            
            avg_interval=num_frames/float(self.num_clips) # how many frames in each clip
            base_offsets=np.arange(self.num_clips)*avg_interval # spaning of each clip, taking in account the interval of each clip
            clip_offsets=base_offsets+avg_interval/2. 
            if self.twice_sample: clip_offsets=np.concatenate([clip_offsets, base_offsets])
                
        else: # 3D recognizer
            
            max_offset=max(num_frames-ori_clip_len, 0) # maximum valid starting index for each clip
            if self.twice_sample: num_clips=self.num_clips*2
            else: num_clips=self.num_clips
            if num_clips>1:
                num_segments=self.num_clips-1 # consider each clip as a gap
                # align test sample strategy with `PySlowFast` repo
                if self.target_fps is not None:
                    offset_between=np.floor(max_offset/float(num_segments)) # floor guarantees that clip_offsets[num_clip-1]<=max_offset
                    clip_offsets=np.arange(num_clips)*offset_between
                else:
                    # We want clips to span the whole video, using floor will shrink clip spacing slightly and the last clip may end earlier
                    # than ideal. This provides better covering of the entire video
                    offset_between=max_offset/float(num_segments)
                    clip_offsets=np.arange(num_clips)*offset_between
                    clip_offsets=np.round(clip_offsets) 
            else: clip_offsets=np.array([max_offset//2])
                
        return clip_offsets

    def _get_train_clips(self, num_frames:int, ori_clip_len:float)->np.ndarray:
        """Get clip offsets in train mode
        
        It will calculate the average interval for selected frames, and randomly shift then within offsets between [0, avg_interval]. 
        If the total number of frames is smaller than clips number or original frames length, it will return all zero indices
        
        Args:
            num_frames (int): Total number of frames in the video
            ori_clip_len (float): Length of original sample clip
        Returns:
            (np.ndarray): Sampled frame indices in train mode
        """
        if self.keep_tail_frames:
            avg_interval=(num_frames-ori_clip_len+1)/float(self.num_clips)
            if num_frames>ori_clip_len-1:
                base_offsets=np.arange(self.num_clips)*avg_interval
                clip_offsets=(base_offsets+np.random.uniform(0, avg_interval, self.num_clips)).astype(np.int32)
            else:
                clip_offsets=np.zeros((self.num_clips), dtype=np.int32)
        else:
            avg_interval=(num_frames-ori_clip_len+1)//self.num_clips
            if avg_interval>0:
                base_offsets=np.arange(self.num_clips)*avg_interval
                clip_offsets=base_offsets+np.random.randint(avg_interval, size=self.num_clips)
            elif num_frames>max(self.num_clips, ori_clip_len):
                clip_offsets=np.sort(np.random.randint(num_frames-ori_clip_len+1, size=self.num_clips))
            elif avg_interval==0:
                ratio=(num_frames-ori_clip_len+1)/self.num_clips
                clip_offsets=np.around(np.arange(self.num_clips)*ratio)
            else: clip_offsets=np.zeros((self.num_clips,), dtype=np.int32)
        return clip_offsets

    def _sample_clips(self, num_frames:int, ori_clip_len:float)->np.ndarray:
        """Choose clip offsets for the video in a given mode
        Args:
            num_frames (int): Total number of frame in the video
        Returns:
            (np.ndarray): Sampled frame indices
        """
        if self.test_mode: clip_offsets=self._get_test_clips(num_frames, ori_clip_len)
        else: clip_offsets=self._get_train_clips(num_frames, ori_clip_len)
        return clip_offsets

    def transform(self, results:dict)->dict:
        """Perform the SampleFrames loading
        Args:
            results (dict): The resulting dict to be modified and passed to the next transform in the pipeline. The function adds the following
                - 'frame_inds': (M,T) int32 frame indices extracted for M clips, each with T frames
                - 'clip_len': Number of frames per clip, i.e., T
                - 'frame_interval': Distance between each frame within a clip in frame-number unit
                - 'num_clips': Number of clips extracted, i.e., M
        """
        total_frames=results['total_frames']
        
        fps=results.get('avg_fps')
        if self.target_fps is None or not fps: fps_scale_ratio=1.
        else: fps_scale_ratio=fps/self.target_fps
    
        # How long (in original frame index space) does one sampled clip span
        ori_clip_len=self._get_ori_clip_len(fps_scale_ratio)
        clip_offsets=self._sample_clips(total_frames,ori_clip_len)
        
        if self.target_fps: # calculate which frame indices to extracted
            # ori_clip_len : how long (in original frame index space) does one sampled clip span
            # frame_inds=start of the clip  + indices of frame within each clip segment starting from 0 to clip length in frames (in original index space)
            # (M,T) where M is the number of clips and T this is the number of frames in each clip
            frame_inds=clip_offsets[:,None]+np.linspace(0, ori_clip_len-1, self.clip_len).astype(np.int32) 
        else:
            frame_inds=clip_offsets[:,None]+np.arange(self.clip_len)[None,:]*self.frame_interval # (M,T)
        
        if self.temporal_jitter:
            perframe_offsets=np.random.randint(self.frame_interval, size=len(frame_inds)) # (M,)
            frame_inds+=perframe_offsets
        
        frame_inds=frame_inds.reshape((-1, self.clip_len)) # (M,T)
        if self.out_of_bound_opt=='loop': frame_inds=np.mod(frame_inds, total_frames)
        elif self.out_of_bound_opt=='repeat_last':
            safe_inds=frame_inds<total_frames
            unsafe_inds=1-safe_inds
            last_ind=np.max(safe_inds*frame_inds, axis=1) # (M,)
            # (unsafe_inds.T*last_ind) yields (T,M) matraix
            new_inds=(safe_inds*frame_inds+(unsafe_inds.T*last_ind).T)
            frame_inds=new_inds
        else: raise ValueError(f'The `out_of_bound` options include "loop" and "repeat_last", but got {self.out_of_bound_opt}') 
            
        start_index=results['start_index']
        frame_inds=np.concatenate(frame_inds)+start_index # from (M,T) to (M*T,)
        results['frame_inds']=frame_inds.astype(np.int32)
        results['clip_len']=self.clip_len
        results['frame_interval']=self.frame_interval
        results['num_clips']=self.num_clips
        
        return results

    def __repr__(self)->str:
        repr_str=(f"{self.__class__.__name__}(clip_len={self.clip_len}, frame_interval={self.frame_interval},"
                  f"num_clips={self.num_clips}, temporal_jitter={self.temporal_jitter},twice_sample={self.twice_sample},"
                  f"out_of_bound_opt={self.out_of_bound_opt},test_mode={self.test_mode})")
        return repr_str

class DecordDecode:
    """Using decord to decode the video
    
    Decord: https://github.com/dmlc/decord

    Required Keys:
        - video_reader
        - frame_inds
    Added Keys:
        - imgs
        - original_shape
        - img_shape
    Args:
        mode (str): Decoding mode with options of 'accurate' and 'efficient'. If set to 'accurate', it will decode videos in accurate frames.
            If set to 'efficient', it will adopt fast seeking but only return key frames which may be duplicated and inaccurate and more 
            suitable for large scene-based video datasets. Default to 'accurate'
    """
    def __init__(self, mode:str='accurate')->None:
        assert mode in ['accurate', 'efficient']
        self.mode=mode
        
    def __call__(self, results:dict)->Optional[Union[dict, tuple[list,list]]]:
        return self.transform(results)
        
    def _decord_load_frames(self, container:object, frame_inds:np.ndarray)->list[np.ndarray]:
        """
        Args:
            container (object): Video reader
            frame_inds (np.ndarray): Frame indices as 1D array
        Returns:
            (list[np.ndarray]): List of video frames, each is of size (H,W,C) 
        """
        if self.mode=='accurate':
            imgs=container.get_batch(frame_inds).asnumpy() # (T,H,W,C)
            imgs=list(imgs)
        elif self.mode=='efficient':
            # This mode is faster; however, it always returns I-FRAME
            # I-FRAME = Intra-coded frame (keyframe), a self-contained video frame that can be decoded independently of other frames.
            container.seek(0)
            imgs=[]
            for idx in frame_inds:
                container.seek(idx)
                frame=container.next()
                imgs.append(frame.asnumpy())
        return imgs
        
    def transform(self, results:dict)->dict:
        """Perform Decord decoding
        Args:
            results (dict): The result dict to be modified and passed to the next transform in the pipeline.
        """
        container=results['video_reader']
        if results['frame_inds'].ndim!=1: results['frame_inds']=np.squeeze(results['frame_inds'])
        
        frame_inds=results['frame_inds']
        imgs=self._decord_load_frames(container, frame_inds) # list of np.ndarray images of shape (H,W,C)
        
        results['video_reader']=None
        del container
        
        results['imgs']=imgs
        results['original_shape']=imgs[0].shape[:2] # (H,W)
        results['img_shape']=imgs[0].shape[:2] 
        
        # we resize the gt_bboxes and proposals to their real scale
        if 'gt_bboxes' in results:
            h, w=results['img_shape']
            scale_factor=np.array([w,h,w,h])
            gt_bboxes=results['gt_bboxes']
            gt_bboxes=(gt_bboxes*scale_factor).astype(np.float32)
            results['gt_bboxes']=gt_bboxes
            if 'proposals' in results and results['proposals'] is not None:
                proposals=results['proposals']
                proposals=(proposals*scale_factor).astype(np.float32)
                results['proposals']=proposals
        return results

    def __repr__(self)->str:
        return f"{self.__class__.__name__}(mode={self.mode})"
        
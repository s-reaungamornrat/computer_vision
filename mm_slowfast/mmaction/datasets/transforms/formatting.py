from __future__ import annotations
from typing import Optional, Union, Sequence

import torch

import numpy as np

from computer_vision.mm_slowfast.mmaction.evaluation.metrics.acc_metric import to_tensor
from computer_vision.mm_slowfast.mmaction.structures.action_data_sample import ActionDataSample
from computer_vision.mm_slowfast.mmengine.structures.instance_data import InstanceData

class FormatShape:
    """Format final imgs shape to the given input_format
    
    Required keys are 'imgs' (optional), 'heatmap_imgs' (optional), 'modality' (optional), 'num_clips', 'clip_len'. Modified keys are 'imgs' and
    added keys are 'input_shape', 'heatmap_input_shape' (optional).

    Args:
        input_format (str): Define the final data format,including 'NCTHW', 'NCHW', 'NCTHW_Heatmap', 'NPTCHW' where N is for batch size,
            P is for the number of proposals (persons/instances/objects), T for time, C for channels, H for height and W for width. 
        collapse (bool): Whether to collapse input_format N... to ... (e.g., NCTHW to CTHW) if N=1. Should be set as True when training and testing 
            detectors. Default to False
    """
    def __init__(self, input_format:str, collapse:bool=False)->None:

        assert input_format in ['NCTHW', 'NCHW', 'NCTHW_Heatmap', 'NPTCHW'], f"The input format {input_format} is not supported"
        
        self.input_format=input_format
        self.collapse=collapse

    def __call__(self, results:dict)->Optional[Union[dict, tuple[list,list]]]:
        return self.transform(results)

    def transform(self, results:dict)->dict:
        """Perform the FormatShape formatting
        
        Args:
            results (dict): The resulting dict to be modified and passed to the next transform in the pipeline
        """
        # list to np.ndarray of shape (M,H,W,C) where M=1 * n_crops * n_clips * T
        if isinstance(results['imgs'], list): results['imgs']=np.array(results['imgs']) 
        assert isinstance(results['imgs'], np.ndarray)
        
        # (M,H,W,C) where M=1 * n_crops * n_clips * T
        if self.collapse: assert results['num_clips']==1
        
        if self.input_format=='NCTHW':
            num_clips=results['num_clips']
            clip_len=results['clip_len']
            if 'imgs' in results:
                imgs=results['imgs']
                if isinstance(clip_len, dict): clip_len=clip_len['RGB']
        
                # (n_crops, n_clips, T, H, W, C)
                imgs=imgs.reshape((-1, num_clips, clip_len)+imgs.shape[1:]) # (M,H,W,C)
                imgs=np.transpose(imgs, (0,1,5,2,3,4)) # (n_crops,n_clips,C,T,H,W)
                imgs=imgs.reshape((-1,)+imgs.shape[2:]) # (n_crops*n_clips,C,T,H,W) = (M',C,T,H,W) where M'=n_crops*n_clips
                results['imgs']=imgs
                results['input_shape']=imgs.shape
        
            if 'heatmap_imgs' in results:
                imgs=results['heatmap_imgs']
                # clip_len must be a dict
                clip_len=clip_len['Pose']
        
                imgs=imgs.reshape((-1, num_clips, clip_len)+imgs.shape[1:]) 
                # fronm (n_crops, n_clips, T, C, H, W) to (n_crops, n_clips, C, T, H, W)
                imgs=imgs.transpose(imgs, (0,1,3,2,4,5))
                imgs=imgs.reshape((-1,)+imgs.shape[2:]) # (M',C,T,H,W) where M'=n_crops*n_clips
                results['heatmap_imgs']=imgs
                results['heatmap_input_shape']=imgs.shape
                
        elif self.input_format=='NCTHW_Heatmap':
            num_clips=results['num_clips']
            clip_len=results['clip_len']
            imgs=results['imgs']
        
            imgs=imgs.reshape((-1, num_clips, clip_len)+imgs.shape[1:]) # (n_crops, n_clips, T, C, H, W)
            imgs=np.transpose(imgs, (0,1,3,2,4,5)) # (n_crops, n_clips, C,T,H,W)
            imgs=imgs.reshape((-1,)+imgs.shape[2:]) # (M',C,T,H,W) where M'=n_crops*n_clips
            results['imgs']=imgs
            results['input_shape']=imgs.shape
            
        elif self.input_format=='NCHW':
            imgs=results['imgs'] # (M,H,W,C)
            imgs=np.transpose(imgs, (0,3,1,2)) # (M,C,H,W)
            if 'modality' in results and results['modality']=='Flow':
                clip_len=results['clip_len']
                imgs=imgs.reshape((-1, clip_len*imgs.shape[1])+imgs.shape[2:]) # (M', n_clips*C, H, W)
            results['imgs']=imgs # (M', n_clips*C, H, W) or (M,C,H,W)
            results['input_shape']=imgs.shape
            
        elif self.input_format=='NPTCHW':
            num_proposals=results['num_proposals']
            num_clips=results['num_clips']
            clip_len=results['clip_len']
            imgs=results['imgs']
            imgs=imgs.reshape((num_proposals, num_clips*clip_len)+imgs.shape[1:]) # (n_proposals, n_clips*T, H, W, C)
            imgs=np.transpose(imgs, (0,1,4,2,3)) # (n_proposals, n_clips*T, C, H, W)
            results['imgs']=imgs
            results['input_shape']=imgs.shape
            
        if self.collapse:
            assert results['imgs'].shape[0]==1
            results['imgs']=results['imgs'].squeeze(0)
            results['input_shape']=results['imgs'].shape
            
        return results

    def __repr__(self)->str:
        return f"{self.__class__.__name__}(input_format='{self.input_format}')"

class PackActionInputs:
    """Pack the input data
    Args:
        collect_keys (tuple[str],optional): The keys to be collected to `packed_results['input']`. Default to ''
        meta_keys (Sequence[str]): The meta keys to saved in the `metainfo` of the `data_sample`. Default to 
            ('img_shape', 'img_key', 'video_id', 'timestamp')
        algorithm_keys (Sequence[str]): The keys of custom elements to be used in the specific algorithm, e.g., custom/user-developed,
            Default to an empty tuple
    """
    
    mapping_table={'gt_bboxes':'bboxes', 'gt_labels':'labels'} # for bounding box only
    
    def __init__(self, collect_keys:Optional[tuple[str]]=None, meta_keys:Sequence[str]=('img_shape', 'img_key', 'video_id', 'timestamp'),
                 algorithm_keys:Sequence[str]=())->None:
        
        self.collect_keys=collect_keys
        self.meta_keys=meta_keys
        self.algorithm_keys=algorithm_keys
        
    def __call__(self, results:dict)->Optional[Union[dict, tuple[list,list]]]:
        return self.transform(results)

    def transform(self, results:dict)->dict:
        """
        Args:
            results (dict): The result dict
        Returns:
            (dict): The transformed result dict
        """
        packed_results=dict()
        if self.collect_keys is not None:
            packed_results['inputs']=dict()
            for key in self.collect_keys: packed_results['inputs'][key]=to_tensor(results[key])
        else:
            if 'imgs' in results: packed_results['inputs']=to_tensor(results['imgs'])
            elif 'heatmap_imgs' in results: packed_results['inputs']=to_tensor(results['heatmap_imgs'])
            elif 'keypoint' in results: packed_results['inputs']=to_tensor(results['keypoint'])
            elif 'audios' in results: packed_results['inputs']=to_tensor(results['audios'])
            elif 'text' in results: packed_results['inputs']=to_tensor(results['text'])
            else: raise ValueError("Cannot get 'imgs', 'keypoint', 'heatmap_imgs', 'audios', or 'text' in the input dict of 'PackActionInputs'")
        
        data_sample=ActionDataSample()
        if 'gt_bboxes' in results:
            instance_data=InstanceData()
            for key, mapped_key in self.mapping_table.items(): instance_data[mapped_key]=to_tensor(results[key])
            data_sample.gt_instances=instance_data
            if 'proposals' in results: data_sample.proposals=InstanceData(bboxes=to_tensor(results['proposals']))
        
        if 'label' in results: data_sample.set_gt_label(results['label'])
        
        # Set custom algorithm keys
        for key in self.algorithm_keys:
            if key in results: data_sample.set_field(results[key], key)
        
        # Set meta keys
        img_meta={k:results[k] for k in self.meta_keys if k in results}
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples']=data_sample
        
        return packed_results

    def __repr__(self)->str:
        return f"{self.__class__.__name__}(colect_keys={self.collect_keys}, meta_keys={self.meta_keys})"
        
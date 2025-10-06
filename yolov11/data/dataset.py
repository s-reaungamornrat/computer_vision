from __future__ import annotations
from typing import Any
from pathlib import Path

import os
import cv2
import yaml
import glob
import math
import copy
import warnings

import torch
import numpy as np

from computer_vision.yolov11.utils.ops import resample_segments
from computer_vision.yolov11.instance.instance import Instances
from .utils import get_image_files, verify_image_label, cache_labels, load_dataset_cache_file, imread

class YOLODataset(torch.utils.data.Dataset):
    def __init__(self, img_path: str|list[str], label_path: str|list[str], data:dict[str, Any] | str, hyp:dict[str, Any],   
                 imgsz:int=640, cache:bool|str=False, augment:bool=True, rect:bool=False, batch_size:int=16, stride:int=32, 
                 pad:float=0.5, single_cls:bool=False, classes:list[int]|None=None, fraction:float=1., channels:int=3):
        """
        data (dict | path): Dataset configuration dictionary.
        img_path (str | list[str]): Path to the image directory or list of paths to image files
        label_path (str | list[str]): Path to the label directory or list of paths to label files
        """
        super().__init__()

        if isinstance(hyp, str):
            hyp=Path(hyp)
            assert hyp.is_file(), f'{hyp} does not exist'
            with open(hyp) as f: hyp=yaml.load(f, Loader=yaml.SafeLoader)
        elif not isinstance(hyp, dict): raise TypeError(f'cfg must be dict/str but got {type(hyp)}')
            
        if isinstance(data, str):
            data=Path(data)
            assert data.is_file(), f'{data} does not exist'
            with open(data, encoding="utf8") as f: self.data=yaml.load(f, Loader=yaml.SafeLoader)
        elif not isinstance(data, dict): raise TypeError(f'cfg must be dict/str but got {type(data)}')
        else: self.data=data
            
        self.imgsz=imgsz
        self.augment=augment
        self.single_cls=single_cls
        self.channels=channels
        self.cv2_flag=cv2.IMREAD_GRAYSCALE if channels==1 else cv2.IMREAD_COLOR
        self.im_files=get_image_files(img_path, fraction=fraction)
        self.labels=self.update_images_labels(label_dirpath=label_path)
        # update labels if specific classes are mentioned or single_cls
        if classes is not None or self.single_cls: self.update_classes(include_classes=classes)
        self.ni=len(self.labels) # number of images
        self.rect=rect
        self.batch_size=batch_size
        self.stride=stride
        self.pad=pad
        if self.rect:
            assert self.batch_size is not None
            self.set_batch4rectangle_training()

        # Buffer storing indices to `self.ims` to access images for mosaic augmentation
        self.buffer=[] # buffer size = batch size
        self.max_buffer_length=min((self.ni,batch_size*8, 1000)) if self.augment else 0
        print('max_buffer_length ', self.max_buffer_length, ' ni ', self.ni)

        # Below lists store images, original-image size, and size of images after resize
        self.ims, self.im_hw0, self.im_hw=[None,]*self.ni, [None,]*self.ni, [None,]*self.ni

    def __len__(self) -> int:
        """Return the length of the labels list for the dataset."""
        return len(self.labels)
        
    def update_images_labels(self, label_dirpath):
        """
        Load previously processed labels stored in a cache file; otherwise, verify image and label files, then read, process, and store
        labels in cache
        Args:
            label_dirpath (str| Path): Path to folder containing labels
        Returns 
            (list[dict]): List of label dictionaries, each containing information about an image and its annotation
        """
        filenames=[os.path.splitext(os.path.basename(file))[0] for file in self.im_files]
        label_files=[os.path.join(label_dirpath, f'{file}.txt') for file in filenames]
        cache_path=Path(label_files[0]).parent.with_suffix('.cache')
        if not cache_path.is_file():
            print(f'In data.dataset.YOLODataset.update_images_labels cache path {cache_path} does not exist. Create it!!!')
            exist=False
            cache=cache_labels(path=cache_path, image_files=self.im_files, label_files=label_files, n_classes=len(self.data['names']), 
                                single_cls=self.single_cls)
            print(f'In data.dataset.YOLODataset.update_images_labels cache is None {cache is None} but type {type(cache)}')
        else: 
            print(f'In data.dataset.YOLODataset.update_images_labels cache path {cache_path} exist. Load it!!!')
            cache, exist=load_dataset_cache_file(cache_path), True
        # try: cache, exist=load_dataset_cache_file(cache_path), True
        # except FileNotFoundError: 
        #     exist=False
        #     cache=cache_labels(path=cache_path, image_files=self.im_files, label_files=label_files, n_classes=len(self.data['names']), 
        #                         single_cls=self.single_cls)
   
        # Display cache
        nf, nm, ne, nc, n=cache.pop("results") # found, missing, empty, corrupt, total
        if exist: print(f"Scanning {cache_path} ... {nf} images with {nm} missing and {ne} empty files as well as {nc} corrupt files")
        
        # Read cache
        labels=cache["labels"]
        if not labels: raise RuntimeError(f"No valid images found in {cache_path}. Images with incorrectly formatted labels are ignored")
        self.im_files=[lb["im_file"] for lb in labels] # update image files
        
        # Check if the labels are consistent, i.e., number of boxes, classes and segments
        lengths=((len(lb["cls"]), len(lb["bboxes"]), len(lb["segments"])) for lb in labels) # labels is a list of dict of each image
        len_cls, len_boxes, len_segments=(sum(x) for x in zip(*lengths))
        if len_segments and len_boxes!=len_segments:
            warnings.warn(f'Box and segment counts should be equal, but got {len_boxes} !={len_segments}'
                          'To resolve this, only boxes will be used and all segment will be removed'
                          'To avoid this, please supply either a detection or segment dataset, not a detection-segment mixed data')
            for lb in labels: lb["segments"]=[]
        if len_cls==0: warnings.warn(f"Class labels are missing or empty in {cache_path}, training may not work correctly")
        return labels

    def update_classes(self, include_classes):
        """
        Update class labels if specific/target classes are provided or if use single class
        Args:
            include_classes (list[int]): class indices/labels to use
        """
        include_class_array=np.array(include_classes).reshape(1,-1) # 1xK where K is the number of target classes
        for i in range(len(self.labels)):
            if include_classes is not None:
                cls=self.labels[i]["cls"] # Nx1 where N is the number of boxes and 1 for class index
                bboxes=self.labels[i]["bboxes"] # Nx4 where N is the number of boxes and 4 for normalized xywh
                segments=self.labels[i]["segments"] if "segments" in self.labels[i] else None
                j=(cls==include_class_array).any(1) # NxK -> N
                self.labels[i]["cls"]=cls[j] # Mx1 where 0<=M<=N
                self.labels[i]["bboxes"]=bboxes[j] # Mx4 where 0<=M<=N
                if segments:
                    self.labels[i]["segments"]=[segments[si] for si, idx in enumerate(j) if idx]
            if self.single_cls: self.labels[i]["cls"][:,0]=0
                
    def set_batch4rectangle_training(self):
        '''
        Determine items in a batch (group images) based on aspect ratios and compute optimal batch shape to minimize padding 
        and distortion.
        '''
        # batch index: we use floor so we do not have last batch with number of items < specified batch_size
        bi=np.floor(np.arange(self.ni)/self.batch_size).astype(int) 
        nb=bi[-1]+1 # number of batches
        s=np.array([x.pop("shape") for x in self.labels]) # height width
        
        # aspect ratio of height/width
        # ar=1-> square image
        # ar<1 -> wide (landscape)
        # ar>1 -> tall (portrait)
        ar=s[:,0]/s[:,1] 
        irect=ar.argsort() # small to large -> wide -> square -> tall
        # grouped by aspect ratio, making batching easier
        self.im_files=[self.im_files[i] for i in irect]
        self.labels=[self.labels[i] for i in irect]
        ar=ar[irect]
        
        # Set training image shapes, i.e., batch shape
        shapes=[[1,1]]*nb # initialize to square image
        for i in range(nb):
            # determine batch-wise shape constraint
            ari=ar[bi==i]
            mini, maxi=ari.min(), ari.max() # min and max aspect ratios
            if maxi<1: shapes[i]=[maxi, 1] # wide image
            elif mini>1: shapes[i]=[1,1/mini] # tall image, similar to max(width/height)
        
        # scale aspect ratios to the target image size, divided by stride (add padding) and multiplying by stride
        # to make sure that batch_shapes are divisible by stride
        self.batch_shapes = np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int) * self.stride
        self.batch = bi  # batch index of image, i.e., which image in which batch

    def load_image(self, index:int, rect_mode:bool=True)->tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        """
        Load an image from dataset index `i`
        Args:
            index (int): Index of the image to load
            rect_mode (bool): Whether to use rectangular resizing
        Returns:
            (np.ndarray): Loaded image as a Numpy array
            (tuple[int, int]): Original image size in (height, width) format
            (tuple[int, int]): Resized image size in (height, width) format
        """
        # if image has already been loaded, return it
        im=self.ims[index]
        if im is not None: return im, self.im_hw0[index],self.im_hw[index]
    
        # otherwise, read it
        im=imread(self.im_files[index], flag=self.cv2_flag) # BRG
        h0, w0=im.shape[:2] # original image size
        if rect_mode: # resize long side to imgsz while maintaining aspect ratio
            r=self.imgsz/max(h0, w0) # ratio
            if r!=1: # max(h0,w0) is equal to imgsz
                w, h=min(math.ceil(w0*r), self.imgsz), min(math.ceil(h0*r), self.imgsz)
                im=cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        elif not(h0==w0==self.imgsz): # resize by stretching image to square imgsz
            im=cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        
        if im.ndim==2: im=im[..., None]
            
        ## Add to buffer if training with augmentation
        if self.augment:
            # image, original-image size, size of image after rescaling
            self.ims[index], self.im_hw0[index], self.im_hw[index]=im, (h0, w0), im.shape[:2]
            self.buffer.append(index)
            print(f'In data.dataset.load_image self.buffer {len(self.buffer)}')
            if 1<len(self.buffer)>=self.max_buffer_length: 
                j=self.buffer.pop(0)
                self.ims[j],self.im_hw0[j],self.im_hw[j]=None, None, None
        return im, (h0,w0), im.shape[:2]

    def update_labels_info(self, label: dict) -> dict:
        """
        Update label format for different tasks
        Args:
            label (dict): Label dict containing bboxes, segments, keypoints, etc
        Returns:
             (dict): Updated label dict with instances
        Note: 
            class is not with bboxes now, classification and semantic segmentation need an independent class label
            Can also support classification and semantic segmentation by adding or removing dict keys there
        """
        
        bboxes=label.pop("bboxes")
        segments=label.pop("segments", [])
        keypoints=label.pop("keypoints", None)
        bbox_format=label.pop("bbox_format")
        normalized=label.pop('normalized')
        
        # NOTE: do NOT resample oriented boxes
        segment_resamples=100 #if self.use_obb else 1000
        if len(segments)>0:
            # make sure segments interpolate correctly if original length is greater than segment_resamples
            max_len=max(len(s) for s in segments)
            if segment_resamples<max_len: segment_resamples= max_len+1
            # NxSx2 where N is the number of masks/boxes and S is the number of points in each segment
            segments=np.stack(resample_segments(segments, n=segment_resamples), axis=0) 
        else: segments=np.zeros((0, segment_resamples, 2), dtype=np.float32)
            
        label['instances']=Instances(bboxes=bboxes, segments=segments, keypoints=keypoints, bbox_format=bbox_format, normalized=normalized)
        
        return label
    
    def get_image_and_label(self, index:int)->dict[str, Any]:
        """
        Get and return label information from the dataset.
        Args:
            index (int): Index of the image to retrieve
        Returns:
            (dict[str, Any]): Label dict with image and metadata
        """
        label=copy.deepcopy(self.labels[index]) # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
        label.pop("shape", None) # shape is for rect, remove it
        label['img'], label['ori_shape'], label['resized_shape']=self.load_image(index) # where all the shape is height, width
        label['ratio_pad']=(label['resized_shape'][0]/label['ori_shape'][0], label['resized_shape'][1]/label['ori_shape'][1]) # for evaluation
        if self.rect: label['rect_shape']=self.batch_shapes[self.batch[index]]
        return self.update_labels_info(label)

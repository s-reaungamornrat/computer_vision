from __future__ import annotations

import glob
import math
import yaml
import os
import random
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
    
from computer_vision.yolov11_pose.utils import DEFAULT_CFG_DICT
from computer_vision.yolov11_pose.utils.patches import imread
from computer_vision.yolov11_pose.cfg import get_cfg
from computer_vision.yolov11_pose.utils.instance import Instances
from computer_vision.yolov11_pose.utils.ops import resample_segments
from .augment import LetterBox, Compose, Format, v8_transforms
from .utils import IMG_FORMATS, img2label_paths, exif_size, verify_image_label, save_dataset_cache_file, load_dataset_cache_file

class YOLODataset(Dataset):
    """Base dataset class for loading and processing image data

    This class provides core functionality for loading images, caching, and preparing data for training and inference
    """
    def __init__(self, args, data:dict|None=None, task:str='detect', img_path:str|list[str]=None, imgsz:int=640, 
                 cache:bool|str=False,augment:bool=True, hyp:dict[str, any]=DEFAULT_CFG_DICT,
                 prefix:str='data.dataset.YOLODataset: ',rect:bool=False,batch_size:int=16,stride:int=32,pad:float=0.5,
                 single_cls:bool=False,classes:list[int]|None=None,
                 fraction:float=1.,channels:int=3):
        """Initialize YOLODataset with given configuration and options
        Args:
            data (dict|None): Dataset configuration dictionary
            task (str): Task type, one of `detect`, `segment`, `pose`, or `oob`
            img_path (str|list[str]): Path to the folder containing images or list of image paths
            imgsz (int): Image size for resizing
            cache (bool): Cache images to RAM or disk during training
            augment (bool): If True, data augmentation is applied
            hyp (dict[str, Any]): Hyperparameters to apply data augmentation
            prefix (str): Prefix to print log messages
            rect (bool): If True, rectangular training is used
            batch_size (int): Size of batches
            stride (int): Stride used in the model
            pad (float): Padding value
            single_cls (bool): If True, single class training is used
            classes (list[int],optional): List of included classes 
            fraction (float): Fraction of dataset to utilize
            channels (int): Number of channels in the images ( 1 for grayscale and 3 for RGB)
        """
        super().__init__()
        self.use_segments=task=='segment'
        self.use_keypoints=task=='pose'
        self.use_obb=task=='obb'
        if isinstance(data, dict): self.data=data
        elif isinstance(data, str): data=Path(data)
        if isinstance(data, Path):
            assert data.is_file(), f'{data} does not exist'
            with open(data, encoding="utf8") as f: self.data=yaml.load(f, Loader=yaml.SafeLoader)

        hyp=get_cfg(cfg=hyp, overrides=args)
        
        self.img_path=img_path
        self.imgsz=imgsz
        self.augment=augment
        self.single_cls=single_cls
        self.prefix=prefix
        self.fraction=fraction
        self.channels=channels
        self.cv2_flag=cv2.IMREAD_GRAYSCALE if channels==1 else cv2.IMREAD_COLOR
        self.im_files=self.get_img_files(self.img_path)
        self.labels=self.get_labels()
        self.update_labels(include_class=classes) # check both include-class and single-class
        self.ni=len(self.labels)
        self.rect=rect
        self.batch_size=batch_size
        self.stride=stride
        self.pad=pad
        if self.rect:
            assert self.batch_size is not None
            self.set_rectangle()
        print(f'In data.dataset.YOLODataset.__init__ augment {augment} rect {rect}')
        # Buffer thread for mosaic images
        self.buffer=[] # buffer size = batch size 
        self.max_buffer_length=min(self.ni, self.batch_size*8, 1000) if self.augment else 0

        # Cache images (options are cache=True, False, None, "ram", "disk")
        self.ims, self.im_hw0, self.im_hw=[None]*self.ni, [None]*self.ni, [None]*self.ni
        self.npy_files=[Path(f).with_suffix(".npy") for f in self.im_files]
        self.cache=cache.lower() if isinstance(cache, str) else "ram" if cache is True else None
        if self.cache=="ram" and self.check_cache_ram():
            if hyp.deterministic:
                warnings.warn("cache='ram' may produce non-deterministic training results"
                              "Consider cache='disk' as a deterministic alternative if your disk space allows")
            self.cache_image()
        elif self.cache=='disk' and self.check_cache_disk(): self.cache_image()

        # Transforms        
        self.transforms=self.build_transforms(hyp=hyp)

    def get_img_files(self, img_path:str|list[str])->list[str]:
        """Read image files from the specified path
        Args:
            img_path (str|list[str]): Path or list of paths to image directories or files
        Returns:
            (list[str]): List of image file paths
        """
        f=[] # image files
        for p in img_path if isinstance(img_path, list) else [img_path]:
            p=Path(p) # os agnosic
            if p.is_dir(): 
                f+=glob.glob(str(p/"**"/"*.*"), recursive=True)
            elif p.is_file():
                with open(p, encoding='utf-8') as t:
                    t=t.read().strip().splitlines()
                    parent=str(p.parent)+os.sep
                    f+=[x.replace('./', parent) if x.startwith('./') else x for x in t] # local to global path
            else: raise FileNotFoundError(f'{self.prefix}{p} does not exist')
        im_files=sorted(x.replace('/',os.sep) for x in f if x.rpartition('.')[-1].lower() in IMG_FORMATS)
        assert im_files, f'{self.prefix}No images found in {img_path}'
        if self.fraction<1:
            im_files=im_files[:round(len(im_files)*self.fraction)] # retain a fraction of the dataset
        return im_files


    def cache_labels(self, path: Path=Path('./labels.cache'))->dict:
        """Cache dataset labels, check images and read shapes
        Args:
            path (Path): Path where to save the cache file
        Returns:
            (dict): Dict containing cached labels and related information
        """
        x={'labels':[]}
        nm, nf, ne, nc, msgs=0,0,0,0,[] # number of missing, found, empty, corrupt, messages
        total=len(self.im_files)
        nkpt,ndim=self.data.get('kpt_shape', (0,0))
        if self.use_keypoints and (nkpt<=0 or ndim not in {2,3}):
            raise ValueError("'kpt_shape' in data.yaml missing or incorrect, Should be a list with [number of keypoints,"
                             "number of dims (2 for x,y or 3 for x,y,visible)], i.e.,'kpt_shape:[17,3]'")
        
        for i, (im_file, lb_file) in enumerate(zip(self.im_files, self.label_files)):
            print(i, end=',')
            im_file, lb, shape, segments, keypoints, nm_f, nf_f, ne_f, nc_f, msg=verify_image_label(im_file, lb_file, self.prefix, 
                                                                                                    self.use_keypoints, len(self.data["names"]), 
                                                                                                    nkpt, ndim, self.single_cls)
            nm+=nm_f
            nf+=nf_f
            ne+=ne_f
            nc+=nc_f
            if im_file:
                x['labels'].append(
                    {
                        "im_file":im_file,
                        "shape": shape, # (height, width)
                        "cls": lb[:,0:1], # n,1
                        "bboxes": lb[:,1:], # n,4
                        "segments": segments,
                        "keypoints": keypoints,
                        "normalized": True,
                        "bbox_format":"xywh",
                    }
                )
            if msg: msgs.append(msg)
        print()
        if msgs: print('\n'.join(msgs))
        if nf==0: warnings.warn(f'{self.prefix}No labels found in {path}')
        x["results"]=nf, nm, ne, nc, len(self.im_files)
        x["msgs"]=msgs # warnings
        
        save_dataset_cache_file(self.prefix, path, x)
        
        return x

    def get_labels(self)->list[dict]:
        """Return dict of labels for YOLO training

        This method loads labels from disk or cache, verifies their integrity, and prepares them for training
        
        Returns:
            (list[dict]): List of label dict, each containing information about an image and its annotations
        """
        self.label_files=img2label_paths(self.im_files)
        cache_path=Path(self.label_files[0]).parent.with_suffix(".cache")
        print(f'In data.dataset.YOLODataset.get_labels cache_path {cache_path}')
        try:
            cache, exists=load_dataset_cache_file(cache_path), True # attempt to load a *.cache file
        except Exception as err:#(FileNotFoundError, AssertionError, AttributeError, ModuleNotFoundError):
            print(f'In data.dataset.YOLODataset.get_labels create and load cache.... because of loading error {err}')
            cache, exists=self.cache_labels(cache_path), False # run cache ops

        # Display cache
        nf, nm, ne, nc, n=cache.pop("results") # found, missing, empty, corrupt, total
        print(f'Scanning {cache_path}... {nf} images, {nm+ne} backgrounds, {nc} corrupts')
        if exists and cache["msgs"]: warnings.warn('{}'.format('\n'.join(cache['msgs']))) # display warnings

        # Read cache
        [cache.pop(k) for k in ("msgs", )] # remove items
        labels=cache['labels']
        if not labels:
            raise RuntimeError(f'No valid images found in {cache_path}. Images with incorrectly formatted labels are ignored')
        self.im_files=[lb['im_file'] for lb in labels] # update im_files

        # Check if the dataset is all boxes or all segments
        lengths=((len(lb['cls']), len(lb['bboxes']), len(lb['segments'])) for lb in labels)
        len_cls, len_boxes, len_segments=(sum(x) for x in zip(*lengths))
        if len_segments and len_boxes!=len_segments:
            warnings.warn(f"Box and segment counts should be equal, but got len(segments)={len(segments)},"
                         f"len(boxes)={len(boxes)}. To resolve this only boxes will be used and all segments will be removed."
                         f"To avoid this, please supply either a detect or segment dataset, not a detect-segment mixed dataset"                        
                        )
            for lb in labels: lb['segments']=[]
        if len_cls==0: warnings.warn(f'Labels are missing or empty in {cache_path}, training may not work correctly')
        return labels
        
    def update_labels(self, include_class:list[int]|None)->None:
        """Update labels to include only specified classes
        Args:
            include_class (list[int], optional): List of classes to include. If None, all classes are included
        """
        include_class_array=np.array(include_class).reshape(1,-1) # 1xM if include_class is not None, else array([[None]], dtype=object)
        for i in range(len(self.labels)):
            if include_class is not None:
                cls=self.labels[i]['cls'] # Nx1
                bboxes=self.labels[i]['bboxes']
                segments=self.labels[i]['segments']
                keypoints=self.labels[i]['keypoints']
                j=(cls==include_class_array).any(1) # NxM -> Nx1 boolean
                self.labels[i]['cls']=cls[j]
                self.labels[i]['bboxes']=bboxes[j]
                if segments:
                    self.labels[i]["segments"]=[segments[si] for si, idx in enumerate(j) if idx]
                if keypoints is not None:
                    self.labels[i]["keypoints"]=keypoints[j]
            if self.single_cls:
                self.labels[i]["cls"][:,0]=0

    def set_rectangle(self)->None:
        """Set the shape of bounding boxes for YOLO detections as rectangles."""
        bi=np.floor(np.arange(self.ni)/self.batch_size).astype(int) # batch index
        nb=bi[-1]+1 # number of batches
        print(f'In data.dataset.YOLODataset.set_rectangle bi {bi} nb {nb}')
        
        # Nx2 where N is the number of images and 2 for (height, width)
        s=np.array([x.pop('shape') for x in self.labels]) 
        ar=s[:,0]/s[:,1] # aspect ratio
        irect=ar.argsort()
        self.im_files=[self.im_files[i] for i in irect]
        self.labels=[self.labels[i] for i in irect]
        ar=ar[irect]
        
        # Set training image shapes
        shapes=[[1,1]]*nb
        for i in range(nb):
            ari=ar[bi==i]
            mini, maxi=ari.min(), ari.max() # bigest height>width, biggest 
            if maxi<1: shapes[i]=[maxi, 1] # height < width
            elif mini>1:  shapes[i]=[1, 1/mini] # height>=width
        self.batch_shapes=np.ceil(np.array(shapes)*self.imgsz/self.stride+self.pad).astype(int)*self.stride
        self.batch=bi # batch index of image

    def check_cache_ram(self, safety_margin:float=0.5)->bool:
        """Check if there is enough RAM for caching images
        Args:
            safety_margin (float): Safety margin factor for RAM calculation
        Returns:
            (bool): True if there is enough RAM, False otherwise
        """
        # bytes of cached images, bytes per gigabytes
        b, gb=0, 1<<30  # a<<b shift binary representation of a to left by b bits, multiplying a by 2^b
        n=min(self.ni, 30) # extrapolate from 30 random images
        count=0
        while count<=n-1:
            im=imread(random.choice(self.im_files)) # sample image
            if im is None: continue
            ratio=self.imgsz/max(im.shape)
            b+=im.nbytes * (ratio**2)
            count+=1
        mem_required=b*(self.ni/n)*(1+safety_margin) # bytes required to cache dataset into RAM
        mem=__import__('psutil').virtual_memory()
        if mem_required>mem.available: 
            self.cache=None
            warnings.warn(f'{self.prefix}{mem_required/gb:.1f}GB RAM required to cache images'
                          f'with {(int(safety_margin*100))}% safety margin but only'
                          f'{mem.available/gb:.1f}/{mem.total/gb:.1f}GB available, not caching images')
            return False
        return True
    
    def check_cache_disk(self,safety_margin:float=0.5)->bool:
        """Check if there is enough disk space for caching images
        Args:
            safety_margin (float): Safety margin factor for disk space calculation
        Returns:
            (bool): True if there is enough disk space, False otherwise
        """
        
        import shutil
        
        b, gb=0, 1<<30 # bytes of cached images, bytes per gigabytes
        n=min(self.ni, 30) # extrapolate from 30 random images
        count=0
        while count<=n-1:
            im_file=random.choice(self.im_files)
            im=imread(im_file)
            if im is None: continue
            b+=im.nbytes
            if not os.access(Path(im_file).parent, os.W_OK):
                self.cache=None
                warnings.warn(f'{self.prefix}Skipping caching images to disk, directory not writable')
                return False
            count+=1
        disk_required=b*(self.ni/n)*(1+safety_margin) # bytes required to cache dataset to disk
        total, _used, free=shutil.disk_usage(Path(self.im_files[0]).parent)
        if disk_required>free:
            self.cache=None
            warnings.warn(f'{self.prefix}{disk_required/gb:.1f}GB disk space required,'
                          f'with {int(safety_margin*100)}% safety margin but only'
                          f'{free/gb:.1f}/{total/gb:.1f}GB free, not caching images to disk')
            return False
        return True

    def cache_images_to_disk(self, i:int)->None:
        """Save an image as an *.npy file for faster loading"""
        f=self.npy_files[i]
        if not f.exists(): 
            np.save(f.as_posix(), imread(self.im_files[i]), allow_pickle=False)

    def load_image(self, i:int, rect_mode:bool=True)->tuple[np.ndarray, tuple[int,int], tuple[int,int]]:
        """Load an image from dataset index `i`
        Args:
            i (int): Index of the image to load
            rect_mode (bool): Whether to use rectangular resizing
        Returns:
            im (np.ndarray): Loaded image as a Numpy array
            hw_original (tuple[int, int]): Original image dimensions in (height, width) format
            hw_resized (tuple[int, int]): Resized image dimensions in (height, width) format
        """
        im, f, fn=self.ims[i], self.im_files[i], self.npy_files[i]
        if im is not None: # cached in RAM
            return im, self.im_hw0[i], self.im_hw[i]
        if fn.exists(): # load npy
            try: im=np.load(fn)
            except Exception as e:
                warnings.warn(f'{self.prefix}Removing corrupt *.npy image file {fn} due to: {e}')
                Path(fn).unlink(missing_ok=True)
                im=imread(f, flags=self.cv2_flag) # BGR
        else: # read image
            im=imread(f, flags=self.cv2_flag) # BGR
        if im is None: raise FileNotFoundError(f'Image Not Found {f}')
        
        h0,w0=im.shape[:2] # original height, width
        # print(f'In data.dataset.YOLODataset.load_image (h0,w0) ({h0},{w0}), not(h0==w0==self.imgsz) {not(h0==w0==self.imgsz)}')
        if rect_mode: # resize long side to imgsz while maintaining aspect ratio
            r=self.imgsz/max(h0,w0) # ratio
            if r!=1: # here we do not resize if one of the image size (width or height) equal imgsz,
                # i.e., if imgsz=640, for (h0,w0)=(640,480), we return the original image
                w,h=min(math.ceil(w0*r), self.imgsz), min(math.ceil(h0*r), self.imgsz)
                im=cv2.resize(im, (w,h),interpolation=cv2.INTER_LINEAR)
        # elif not(h0==w0==self.imgsz): # resize by stretching image to square imgsz
        #     # somehow this block never got executed and I do not know why even if `not(h0==w0==self.imgsz)` evaluate to True
        #     print(f'In data.dataset.YOLODataset.load_image before resizing image shape {im.shape}')
        #     im=cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        #     print(f'In data.dataset.YOLODataset.load_image resized image to {im.shape}')
        # print(f'In data.dataset.YOLODataset.load_image not(h0==w0==self.imgsz) {not(h0==w0==self.imgsz)}')
        if im.ndim==2: im=[...,None]
        
        # Add to buffer if training with augmentation
        if self.augment:
            self.ims[i], self.im_hw0[i], self.im_hw[i]=im, (h0, w0), im.shape[:2] # im, how_original, hw_resized
            self.buffer.append(i)
            if 1< len(self.buffer)>=self.max_buffer_length: # prevent empty buffer
                j=self.buffer.pop(0)
                if self.cache!='ram':
                    self.ims[j],self.im_hw0[j],self.im_hw[j]=None,None,None
        return im, (h0,w0), im.shape[:2]

    def cache_image(self)->None:
        """Cache images to memory or disk for faster training"""
        b, gb=0, 1<<30 # bytes of cached images, bytes per gigabytes
        fcn, storage=(self.cache_images_to_disk, 'Disk') if self.cache=='disk' else (self.load_image, 'RAM')
        for i in range(self.ni):
            out=fcn(i)
            if self.cache=='disk': b+=self.npy_files[i].stat().st_size
            else: # ram
                self.ims[i], self.im_hw0[i], self.im_hw[i]=out
                b+=self.ims[i].nbytes
        print(f'{self.prefix}: Caching images ({b/gb:.1f}GB {storage})')

    def __len__(self)->int:
        """Return the length of the labels list for the dataset."""
        return len(self.labels)

    def update_labels_info(self, label:dict)->dict:
        """Update label format for different tasks
        
        Args:
            label (dict): Label dict containing bboxes, segments, keypoints, etc
        Returns:
            (dict): Updated label dict with instances
        Notes:
            cls is not with bboxes, classification and semantic segmentation need an independent cls label
            Can also support classification and semantic segmentatioon by adding or removing dict keys there
        """
        bboxes=label.pop('bboxes')
        segments=label.pop('segments',[])
        keypoints=label.pop('keypoints', None)
        bbox_format=label.pop('bbox_format')
        normalized=label.pop('normalized')
        
        # Note: do not resample oriented boxes
        segment_resamples=100 if self.use_obb else 1000
        if len(segments)>0:
            # make sure segments interpolate correctly if original length is greater than segment_resamples
            max_len=max(len(s) for s in segments)
            segment_resamples=(max_len+1) if segment_resamples < max_len else segment_resamples
            # list[np.array(segment_resamples, 2)]*num_samples
            segments=np.stack(resample_segments(segments, n=segment_resamples), axis=0)
        else: segments=np.zeros((0, segment_resamples, 2), dtype=np.float32)
        label['instances']=Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label

    def get_image_and_label(self, index:int)->dict[str, Any]:
        """Get and return label information from the dataset.
        Args:
            index (int): Index of the image to retrieve
        Returns:
            (dict[str, Any]): Label dict with image and metadata
        """
        label=deepcopy(self.labels[index]) # requires deepcopy https://github.com/ultralytics/ultralytics/pull/1948
        label.pop('shape', None) # shape is for rect, remove it
        label['img'],label['ori_shape'],label['resized_shape']=self.load_image(index)
        label['ratio_pad']=(
            label['resized_shape'][0]/label['ori_shape'][0],
            label['resized_shape'][1]/label['ori_shape'][1],
        ) # for evaluation
        if self.rect: label['rect_shape']=self.batch_shapes[self.batch[index]]
        return self.update_labels_info(label)

    def build_transforms(self, hyp:dict|None=None)->Compose:
        """Build and append transforms to the list

        Args:
            hyp (dict, optional): Hyperparameters for transforms.
        Returns:
            (Compose): Composed transforms
        """
        if self.augment:
            hyp.mosaic=hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup=hyp.mixup if self.augment and not self.rect else 0.0
            hyp.cutmix=hyp.cutmix if self.augment and not self.rect else 0.0
            transforms=v8_transforms(self, self.imgsz, hyp)
        else: transforms=Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(bbox_format='xywh',
                  normalize=True,
                  return_mask=self.use_segments,
                  return_keypoint=self.use_keypoints,
                  return_obb=self.use_obb,
                  batch_idx=True,
                  mask_ratio=hyp.mask_ratio,
                  mask_overlap=hyp.overlap_mask,
                  bgr=hyp.bgr if self.augment else 0., # only affect training
                  )
        )

        return transforms

    def __getitem__(self, index:int)->dict[str, Any]:
        """Return transformed label information for given index"""
        return self.transforms(self.get_image_and_label(index))

    @staticmethod
    def collate_fn(batch:list[dict])->dict:
        """Collate data sameples into batches
        Args:
            batch (list[dict]): List of dicts containing sample data
        Returns:
            (dict): Collated batch with stacked tensors
        """
        new_batch={}
        batch=[dict(sorted(b.items())) for b in batch] # make sure the keys are in the same order by sorting each dict by keys
        keys=batch[0].keys()
        values=list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value=values[i] # tuple of the items for this k from the whole batch
            if k in {'img', 'text_feats'}:
                value=torch.stack(value, 0) # from tuple of B of CxHxW to BxCxHxW
            elif k=='visuals':
                value=torch.nn.utils.rnn.pad_sequence(value, batch_first=True)
            if k in {'masks', 'keypoints', 'bboxes', 'cls', 'segments', 'obb'}:
                # each item in the batch doesn't have equal amount of objects so we concatenate them all along dim=0
                # e.g., change bboxes [torch.Size([2, 4]), torch.Size([7, 4]), torch.Size([0, 4]), torch.Size([3, 4]), torch.Size([3, 4])]
                # to torch.Size([15, 4])
                # change cls [torch.Size([2, 1]), torch.Size([7, 1]), torch.Size([0, 1]), torch.Size([3, 1]), torch.Size([3, 1])] to torch.Size([15, 1])
                # change multilabel masks [torch.Size([1, 160, 160]), torch.Size([1, 160, 160]), torch.Size([1, 160, 160]), torch.Size([1, 160, 160]), 
                # torch.Size([1, 160, 160])] to torch.Size([5, 160, 160])
                value=torch.cat(value, 0) 
            new_batch[k]=value
        # After for loop, batch-idx will look somewht like below
        #  (tensor([0., 0.]),
        #   tensor([0., 0., 0., 0., 0., 0., 0.]),
        #   tensor([]),
        #   tensor([0., 0., 0.]),
        #   tensor([0., 0., 0.])) 
        # a tuple of tensor of zeros where the number of zeros equal the number of objects in that item in the batch
        new_batch['batch_idx']=list(new_batch['batch_idx'])
        for i in range(len(new_batch['batch_idx'])):
            new_batch['batch_idx'][i]+=i # add target image index for build_targets()
        # Add item index (indexing item/image in the batch), so we get batch_idx looking somewhat like below
        # [tensor([0., 0.]),
        #  tensor([1., 1., 1., 1., 1., 1., 1.]),
        #  tensor([]),
        #  tensor([3., 3., 3.]),
        #  tensor([4., 4., 4.])]
        new_batch['batch_idx']=torch.cat(new_batch['batch_idx'], 0) # Then we stack them to a 1D array
        # looking like tensor([0., 0., 1., 1., 1., 1., 1., 1., 1., 3., 3., 3., 4., 4., 4.])
        return new_batch
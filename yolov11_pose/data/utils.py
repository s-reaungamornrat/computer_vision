from __future__ import annotations

import os
import yaml
import warnings
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from computer_vision.yolov11_pose.utils.ops import segments2boxes
from computer_vision.yolov11_pose.utils import is_dir_writeable

IMG_FORMATS = {"bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm", "heic"}  # image suffixes

def exif_size(img:Image.Image)->tuple[int,int]:
    """Return exif-corrected PIL size
    Args:
        img (Image.Image): PIL read image
    Returns:
        s (tuple): Image width and height
    """
    s=img.size # (width, height)
    if img.format=='JPEG': # only support JPEG images
        try:
            if exif:=img.getexif():
                rotation=exif.get(274, None) # the EXIF key for the orientation tag is 274
                if rotation in {6,8}: # rotatioon 270 or 90
                    s=s[1],s[0]
        except Exception: pass
    return s
    
def img2label_paths(img_paths:list[str])->list[str]:
    """Convert image paths to label paths by replacing 'images' with 'labels' and extension with '.txt'"""
    sa, sb=f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}' # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa,1)).rsplit('.',1)[0]+'.txt' for x in img_paths]

def load_dataset_cache_file(path:Path)->dict:
    """Load *.cache dict from path"""
    import gc
    gc.disable() # reduce pickle load time https://github.com/ultralytics/ultralytics/pull/1585
    cache=np.load(str(path), allow_pickle=True).item() # load dict
    gc.enable()
    return cache

def verify_labels(lb_file, keypoint:bool=False, num_cls:int=0, nkpt:int=0, ndim:int=0, single_cls:bool=False):
    """
    Read and verify annotation from label files
    Args:
        lb_file (str): Path to label .txt file
        keypoint (bool): Whether to the label file containing keypoints
        num_cls (int): Number of classes
        nkpt (int): Number of keypoints
        ndim (int): Dimension of keypoints (2 for x,y or 3 for x, y, visibility)
        single_cls (bool): Whether to consider multiple classes as a single class
    Returns:
        lb (np.ndarray): Target labels of size Nx5 of (cls, xywh) where xywh is normalized coordinates
        segments (list[np.ndarray]): List of Nx2 segments of each object
        keypoints (np.ndarray): Nx3 keypoint location (x, y, visibility)
        nm (int): Number of missing label files
        nf (int): Number of found label files
        ne (int): Number of empty label files
        msg (str): Warning or information message
    """
    nm=nf=ne=0 # number of missing, found, empty label files
    msg, segments, keypoints="", [],None
    # verify labels
    if not os.path.isfile(lb_file):
        nm=1 # label missing
        lb=np.zeros((0, (5+nkpt*ndim) if keypoint else 5), dtype=np.float32)
    else:
        nf=1
        with open(lb_file, encoding='utf-8') as f:
            lb=[x.split() for x in f.read().strip().splitlines() if len(x)]
            if any(len(x)>6 for x in lb) and (not keypoint): # is segment
                # each x line/list in lb is (cls, xy1,xy2,xy3,...)
                classes=np.array([x[0] for x in lb], dtype=np.float32) # cls
                segments=[np.array(x[1:], dtype=np.float32).reshape(-1,2) for x in lb] # xy1,xy2,xy3,...
                lb=np.concatenate((classes.reshape(-1,1), segments2boxes(segments)),1) # (cls, xywh)
            lb=np.array(lb, dtype=np.float32)
            if nl:=len(lb): # number of lines in text file, each line for 1 object annotation
                if keypoint:
                    assert lb.shape[1]==(5+nkpt*ndim), f'labels require {(5+nkpt*ndim)} columns each'
                    points=lb[:,5:].reshape(-1, ndim)[:,:2]
                else:
                    assert lb.shape[1]==5, f'labels require 5 columns, {lb.shape[1]} columns detected'
                    points=lb[:,1:]
                    
                # Coordinate points check with 1% tolerance
                assert points.max() <=1.01, f'non-normalized or out of bounds coordinates {points[points>1.01]}'
                assert lb.min()>=-0.01, f'negative class labels or coordinates {lb[lb<-0.01]}'
    
                # All labels
                max_cls=0 if single_cls else lb[:,0].max() # max label count
                assert max_cls < num_cls, (f'Label class {int(max_cls)} exceeds dataset class count {num_cls}.'
                                           f'Possible class labels are 0-{num_cls-1}')
                _, i=np.unique(lb, axis=0, return_index=True) # duplicate row check
                if len(i)<nl: # duplicate row check
                    lb=lb[i]
                    if segments: segments=[segments[x] for x in i]
                    msg=f'{prefix}{im_file}: {nl-len(i)} duplicate labels removed'
            else:
                ne=1 # label empty
                lb=np.zeros((0,(5+nkpt*ndim) if keypoint else 5), dtype=np.float32)

    if keypoint:
        keypoints=lb[:,5:].reshape(-1, nkpt, ndim)
        if ndim==2:
            kpt_mask=np.where((keypoints[...,0]<0) |(keypoints[...,1]<0), 0., 1.).astype(np.float32)
            keypoints=np.concatenate([keypoints, kpt_mask[...,None]], axis=-1) # (nl, nkpt, 3)
    lb=lb[:,:5] # (cls, xywh)
    return lb, segments, keypoints, nm, nf, ne, msg

def verify_image_label(im_file, lb_file, prefix, keypoint, num_cls, nkpt, ndim, single_cls):
    """Verify one image-label pair
    Args:
        im_file (str): Path to an image file
        lb_file (str): Path to the corresponding label file
        prefix (str): Prefix to add to warning and information messages
        keypoint (bool): Whether to read keypoints from the label file
        num_cls (int): Number of classes
        nkpt (int): Number of keypoints per object
        ndim (int): Dimension of keypoints (2 for x, y and 3 for x, y, visibility)
        single_cls (bool): Whether to consider multiple classes as a single class
    Returns:
        im_file (str): Path to the valid image file
        lb (np.ndarray): Target labels of size Nx5 of (cls, xywh) where xywh is normalized coordinates
        shape (tuple[int,int]): Shape of image as (height, width)
        segments (list[np.ndarray]): List of Nx2 segments of each object
        keypoints (np.ndarray): Nx3 keypoint location (x, y, visibility)
        nm (int): Number of missing label files
        nf (int): Number of found label files
        ne (int): Number of empty label files
        nc (int): Number of corrupted label files
        msg (str): Warning or information message
    """
    # Number (missing, found, empty, corrupt), message, segments, keypoints
    nm, nf, ne, nc, msg=0,0,0,0,""
    try:
        # Verify images
        im=Image.open(im_file)
        im.verify() # PIL verify
        shape=exif_size(im) # (width, height)
        shape=shape[::-1] # (height, width)
        assert all(s>9 for s in shape), f'image size {shape} <10 pixels'
        assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}.'
        if im.format.lower() in {'jpg', 'jpeg'}:
            with open(im_file, 'rb') as f:
                f.seek(-2,2)
                if f.read()!=b"\xff\xd9": # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
                    msg=f'{prefix}{im_file}: corrupt JPEG restored and saved'
        # Verify labels
        lb, segments, keypoints, nm, nf, ne, msg=verify_labels(lb_file=lb_file, keypoint=keypoint, num_cls=num_cls, nkpt=nkpt, 
                                                               ndim=ndim, single_cls=single_cls)
        return (im_file, lb, shape, segments, keypoints, nm, nf, ne, nc, msg)
    except Exception as e:
        nc=1
        msg=f'{prefix}{im_file}: ignoring corrupt image/label: {e}'
        return (None,None,None,None,None,nm,nf,ne,nc,msg)

def save_dataset_cache_file(prefix: str, path: Path, x: dict):
    """Save a dataset *.cache  dict x to path"""

    if is_dir_writeable(path.parent):
        if path.exists(): path.unlink() # remove *.cach file if exists
        with open(str(path), 'wb') as file: # context manager here fixes windows async np.save bug
            np.save(file, x)
        print(f'{prefix}New cache created: {path}')
    else: warnings.warn(f'{prefix}Cache directory {path.parent} is not writable, cache not saved')

def polygon2mask(imgsz:tuple[int, int], polygons:list[np.ndarray], color:int=1, downsample_ratio:int=1)->np.ndarray:
    """Convert a list of polygons to a binary mask of the specified image size
    Args:
        imgsz (tuple[int,int]): The size of the image (height, width)
        polygons (list[np.ndarray]): A list of polygons. Each is an array of shape (M,) where M is the number of points such that M%2=0.
            i.e., [x1,y1,x2,y2,x3,y3,...,xM,yM]
        color (int, optional): The color value to fill in the polygons on the masks
        downsample_ratio (int,optional): Downsample step of the mask in pixels
    Returns:
        (np.ndarray): A binary mask of the specified image size with the polygon filled in
    """
    mask=np.zeros(imgsz, dtype=np.uint8)
    polygons=np.asarray(polygons, dtype=np.int32) # 1xM
    polygons=polygons.reshape((polygons.shape[0],-1,2)) # 1xMx2
    cv2.fillPoly(mask, polygons, color=color)
    nh, nw=(imgsz[0]//downsample_ratio, imgsz[1]//downsample_ratio)
    # Note: fillPoly first then resize is trying to keep the same loss calculation method when mask-ratio=1
    return cv2.resize(mask, (nw,nh))

def polygon2masks_overlap(imgsz:tuple[int,int], segments:list[np.ndarray]|np.ndarray, downsample_ratio:int=1)->tuple[np.ndarray, np.ndarray]:
    """
    Convert a set polygons to multilabel/overlap mask
    Args:
        imgsz (tuple[int, int]): Image size (height, width)
        segments (list[np.ndarray]|np.ndarray): A set of polygons as a list of N of Mx2 segments or as a NxMx2 array
        downsample_ratio (int): Downsample rate in pixel unit
    Returns:
        (np.ndarray[int32/int8]): Multilabel/overlap masks of shape (H,W)
        (np.ndarray[int]): Sorted indices of objects from biggest to smallest of size (N,)
    """
    masks=np.zeros((imgsz[0]//downsample_ratio, imgsz[1]//downsample_ratio), dtype=np.int32 if len(segments)>255 else np.uint8)
    areas, ms=[], []
    for segment in segments:
        mask=polygon2mask(imgsz, [segment.reshape(-1)], downsample_ratio=downsample_ratio, color=1)
        ms.append(mask.astype(masks.dtype)) # a list of HxW 
        areas.append(mask.sum())
    areas=np.asarray(areas)
    index=np.argsort(-areas) # sort from large area to small area
    ms=np.array(ms)[index] # NxHxW, where N is the number of objects
    for i in range(len(segments)):
        mask=ms[i]*(i+1) # assign label 1 to biggest object, 2 to smaller, and so on until the smallest
        masks=masks+mask
        masks=np.clip(masks, a_min=0, a_max=i+1)
    return masks,index

def polygons2masks(imgsz:tuple[int,int], polygons:list[np.ndarray]|np.ndarray, color:int, downsample_ratio:int=1)->np.ndarray:
    """Convert a list of polygons to a set of binary masks of the specified image size
    
    Args:
        imgsz (tuple[int,int]): The size of the image as (height, width)
        polygons (list[np.ndarray]|np.ndarray): A list of N polygons (each of size Mx2) or an NxMx2 array of polygons
        color (int): The color value to fill in the polygons on the masks
        downsample_ratio (int): Downsample step of the mask
    Returns:
        (np.ndarray): NxHxW array representing N of HxW binary masks of the specified image size with the polygons filled in
    """
    return np.array([polygon2mask(imgsz, [x.reshape(-1)], color, downsample_ratio) for x in polygons])

def check_det_dataset(dataset:str)->dict[str, Any]:
    """Read data configuration
    Args:
        dataset (str): Path to dataset descriptor file in YAML format
    Returns:
        (dict[str, Any]): Parsed dataset information and paths
    """
    print(f'In data.utils.check_det_dataset type(dataset) {type(dataset)} ')
    if isinstance(dataset, str): dataset=Path(dataset)
    if isinstance(dataset, Path):
        assert dataset.is_file(), f'{dataset} does not exist'
        with open(dataset, encoding="utf-8") as f: data=yaml.load(f, Loader=yaml.SafeLoader)
    # Check
    for k in 'train','val':
        if k not in data:
            if k!='val' or 'validation' not in data:
                raise SyntaxError(f'{dataset} "{k}:" key missing.\n "train" and "val" are required in all data YAMLs')
            warnings.warn("renaming data YAML 'validation' key to 'val' to match YOLO format")
            data['val']=data.pop('validation')
    if 'names' not in data and 'nc' not in data:
        raise SyntaxError(f'{dataset} key missing.\n either "names" or "nc" are required in all data YAMLs ')
    if 'names' in data and 'nc' in data and len(data['name'])!=data['nc']:
        raise SyntaxError(f'{dataset} "names" length {len(data["names"])} and nc:{data["nc"]} must match')
    if "names" not in data: data["names"]=[f'class_{i}' for i in range(data['nc'])]
    else: data['nc']=len(data['names'])
    data['channels']=data.get("channels", 3) # default to 3
    return data
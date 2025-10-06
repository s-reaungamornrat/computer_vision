from __future__ import annotations

import os

import cv2
import glob
import torch

import numpy as np
from pathlib import Path
from PIL import Image, ImageOps, ExifTags

from computer_vision.yolov11.utils.ops import segments2boxes

IMG_FORMATS = {"bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm", "heic"}  # image suffixes
# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation': break
        

def imread(filename: str, flag: int=cv2.IMREAD_COLOR)->np.ndarray | None:
    """
    Read an image from a file with multilanguage filename support
    Args:
        filename (str)L Path to the file
        flag (int, optional): Flag that can take values of cv2.IMREAD_*. Control how the image is read
    Returns:
        (np.ndarray | None): The read image array, or None if reading fails. If a color image (cv2.IMREAD_COLOR), return BGR
    Examples:
        >>> img=imread("path/to/image.jpg")
        >>> img=imread("path/to/image.jpg", cv2.IMREAD_GRAYSCALE)
    Source: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/patches.py
    """
    file_bytes=np.fromfile(filename, np.uint8)

    if filename.endswith((".tiff", ".tif")):
        success, frames=cv2.imdecodemulti(file_bytes, cv2.IMREAD_UNCHANGED)
        if success:
            # Handle RGB images in tif/tiff format
            return frames[0] if len(frames)==1 and frames[0].ndim==3 else np.stack(frames, axis=2)
        return None
    else:
        im=cv2.imdecode(file_bytes, flag)
        return im[...,None] if im is not None and im.ndim==2 else im # always ensure 3 dimensions

def get_image_files(image_dirpath, fraction=1.):
    """
    Get a list of the absolute paths to images. If fraction<1, maintain just fraction of the image files
    Args:
        image_dirpath (str): Path to the folder containing images
        fraction (float): Fraction of data to use
    Returns:
        (list): Absolute paths to image files
    """

    files=glob.glob(str(Path(image_dirpath)/ "**" / "*.*"), recursive=True)
    im_files=sorted(x.replace('/', os.sep) for x in files if x.rpartition('.')[-1].lower() in IMG_FORMATS)
    
    if fraction<1.:
        im_files=im_files[:round(len(im_files)*fraction)] # retain a fraction of data
    return im_files

def exif_size(img: Image.Image)->tuple[int, int]:
    """
    Return exif-corrected PIL size
    Args:
        img (PIL.Image)
    Returns
        (tuple[int,int]): width and height of img after EXIF correction 
    """
    s=img.size # (width, height)
    try:
        if exif:=img.getexif(): # first assign exif=img.getexif(), then check whether exif exist
            rotation=exif.get(orientation, None) # the EXIF key for the orientation tag is 
            if rotation in {6,8}: # rotation 270 or 90
                s=s[1],s[0]
    except Exception: pass
    return s

def verify_image_label(im_file, lb_file, num_cls, single_cls=False, min_imgsz=9):
    """
    Verify whether image and label files for 1 pair are readable and contain data
    Args:
        im_file (str): Path to image file
        lb_file (str): Path to label file
        num_cls (int): Number of classes
        single_cls (bool): Whether to consider all classes as 1
        min_imgsz (int): Minimum with/height
    Returns:
        im_file (str): Path to valid image file 
        lb (np.ndarray): Nx5 labels where N is the number of boxes and 5 is for class, xywh after normalization with
            (x,y) represents the box center
        shape (tuple[int]): Image height/width
        segments (list[np.ndarray]): list of N of Mx2 segments where M is the number of points in each segment and 2 for x, y
            after normalization
        nm (int): Number of missing label files
        nf (int): Number of found label files
        ne (int): Number of empty label files
        nc (int): Number of corrupt image or label files
        msg (str): Warning/error message
    """
    # Number of (missing, found, empty, corrupt), message, segment, keypoints
    nm, nf, ne, nc, msg, segments=0,0,0,0,"",[]
    try:
        # Verify images
        im=Image.open(im_file)
        im.verify() # PIL verify
        shape=exif_size(im)[::-1] # height, width
        assert all(s>min_imgsz for s in shape), f'image size {shape} < 10 pixels' 
        assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}'
        if im.format.lower() in {"jpg", "jpeg"}:
            with open(im_file, 'rb') as f:
                f.seek(-2,2) # move the file pointer to the 2 byte before the end of file
                if f.read()!=b"\xff\xd9": # valid JPEG must end with 2 byte marker 0xFF 0xD9 (b"\xff\xd9")
                    # Attempt to correct and save image with subsampling=0 for full chorma resolution (highest quality) and quality=100 maximum quality
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
                    msg=f" data.utils.verify_image_label {im_file}: corrupt JPEG restored and saved"
                    
        # Verify labels
        if not os.path.isfile(lb_file):
            nm=1 # label missing
            lb=np.zeros((0, 5), dtype=np.float32) # cls, xywh in normalized space
        else:
            nf=1 # label found
            with open(lb_file, encoding="utf-8") as f:
                lb=[x.split() for x in f.read().strip().splitlines() if len(x)]
            if any(len(x)>6 for x in lb): # is segment -> (cls, xy1, xy2, ...)
                classes=np.array([x[0] for x in lb], dtype=np.float32)
                segments=[np.array(x[1:], dtype=np.float32).reshape(-1,2) for x in lb]
                # Nx5 where N is the number of boxes and 5 for cls, xywh in normalize unit
                lb=np.concatenate((classes.reshape(-1,1), segments2boxes(segments)), axis=1) 
            lb=np.array(lb, dtype=np.float32)
            if nl:=len(lb): # 1) assign nl=len(lb), 2) check if nl>0
                assert lb.shape[1]==5, f'labels require 5 columns, {lb.shape[1]} columns detected'
                points=lb[:,1:]
                # Coordinate points check with 1% tolerance
                assert points.max()<=1.01, f"non-normalized or out of bound coordinates {points[points>1.01]}"
                assert lb.min()>=-0.01, f"negative class labels or coordinate {lb[lb<-0.01]}"
                # All labels
                max_cls=0 if single_cls else lb[:,0].max() # max label count
                assert max_cls < num_cls, (
                    f"Label class {int(max_cls)} exceeds dataset class count {num_cls}"
                    f"Possible class labels are 0-{num_cls-1}"
                )
                _, i=np.unique(lb, axis=0, return_index=True)
                if len(i) < nl: # deplicate row
                    lb=lb[i] # filter out duplicates
                    if segments: segments=[segments[x] for x in i]
            else: # label file is empty -> no label        
                ne=1
                lb=np.zeros((0,5), dtype=np.float32)
        lb=lb[:,:5]
        return im_file, lb, shape, segments, nm, nf, ne, nc, msg     
    except Exception as e:
        nc=1
        msg=f'data.utils.verify_image_label {im_file}: ignoring corrupt image/label: {e}'
        return None, None, None, None, nm, nf, ne, nc, msg 
        
def load_dataset_cache_file(path: Path)->dict:
    """
    Load *.cache dictionary from path
    """
    cache=torch.load(path, map_location=torch.device("cpu"), weights_only=False,)
    return cache

def save_dataset_cache_file(path: Path, x: dict):
    """
    Save cache dictionary to path
    Args:
        path (Path): Path to save cache
        x (dict): data to cache containing following keys: `labels`, `results`
    """
    torch.save(x, path)

def cache_labels(path: Path, image_files: list[str], label_files: list[str], n_classes: int, single_cls: bool=False):
    """
    Read, format, and save all labels into 1 files for fast future load
    Args:
        path (Path): Path to save cache
        image_files (list[str]): List of path to images
        label_files (list[str]): List of path to corresponding labels
        n_classes (int): Number of classes
        single_cls (bool): Whether to consider all classes as a single class
    Returns:
        cache (dict[str, Any]): annotation cache with keys `labels` and `results`
    """

    x={"labels":[]}
    nm=nf=ne=nc=0 # number missing, found, empty, corrupt
    msgs, total=[],len(image_files)
    
    
    for i, (im_file, lb_file) in enumerate(zip(image_files, label_files)):
        im_file, lb, shape, segments, nm_f, nf_f, ne_f, nc_f, msg =verify_image_label(im_file=im_file, lb_file=lb_file, num_cls=n_classes, 
                                                                              single_cls=single_cls, min_imgsz=9)
        nm+=nm_f; nf+=nf_f; ne+=ne_f; nc+=nc_f
        if im_file:
            x["labels"].append(
                {
                    "im_file":im_file,   
                    "shape":shape,
                    "cls": lb[:, 0:1], # Nx1
                    "bboxes": lb[:, 1:], # Nx4
                    "segments": segments,
                    "normalized": True,
                    "bbox_format": "xywh",
                }
            )
        if msg: print(f'In data.utils.cach_label {i}-th file -> {msg}')
    
    if nf==0: print(f'No label found!!!')
    x["results"]=nf, nm, ne, nc, total
    save_dataset_cache_file(path, x)
    return x

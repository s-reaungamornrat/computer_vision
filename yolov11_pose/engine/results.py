from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
from functools import lru_cache

import numpy as np
import torch

from computer_vision.yolov11_pose.utils import ops

class Boxes:
    """
    A class for managing and manipulating detection boxes
    This class provides comprehensive functionality for handling detection boxes, including their coordinates, confidence scores,
    class labels, and optional tracking IDs. It supports various box formats and offers methods for easy manipultion and conversion
    between different coordinate systems
    Examples:
        >>> boxes_data=torch.tensor([[100, 50, 150, 100, 0.9, 0], [200, 150, 300, 250, 0.8, 1]])
        >>> orig_shape=(480, 640) # height, width
        >>> boxes = Boxes(boxes_data, orig_shape)
        >>> print(boxes.xyxy)
        >>> print(boxes.conf, boxes.cls)
        >>> print(boxes.xywhn)
    """
    def __init__(self, boxes: torch.Tensor | np.ndarray, orig_shape: tuple[int, int])->None:
        """
        Initialize the Boxes object with detection box data and the original image shape
        This class manages detection boxes, providing easy access and manipulation of box coordinates, confidence scores, 
        class identifiers, and optional tracking IDs. It supports multiple formats for box coordinates, including both absolute and
        normalized forms.
        Args:
            boxes (torch.Tensor|np.ndarray): A tensor or numpy array with detection boxes of shape (num_boxes, 6) or (num_boxes, 7)
                Columns should contain [x1,y1,x2,y2,(optional) track_id, confidence, class]
            orig_shape (tuple[int, int]): The original image shape as (height, width). Used for normalization
        """
        assert isinstance(boxes, (torch.Tensor, np.ndarray)), f'boxes must be torch.Tensor or np.ndarray, but got {type(boxes)}'
        if boxes.ndim==1: 
            boxes=boxes[None,:]
        n=boxes.shape[-1]
        assert n in {6,7}, f"expect 6 or 7 dimensions but got {n}" # xyxy, track-id, conf, cls
        self.is_track=n==7
        self.data=boxes
        self.orig_shape=orig_shape
        
    @property
    def xyxy(self)->torch.Tensor|np.ndarray:
        """
        Return bounding boxes in [x1,y1,x2,y2] format
        Returns:
            (torch.Tensor|np.ndarray): A tensor or numpy array of shape (n,4) containing bounding box coordinates in [x1,y1,x2,y2] format,
                where n is the number of boxes
        """
        return self.data[:,:4]

    @property
    def conf(self)->torch.Tensor|np.ndarray:
        """
        Return the confidence scores for each detection box
        Returns:
            (torch.Tensor | np.ndarray): A 1D tensor or array containing confidence socres for each detection, with shape (N,) where
                N is the number of detections
        """
        return self.data[:,-2]

    @property
    def cls(self)->torch.Tensor|np.ndarray:
        """
        Return the class ID tensor representing category predictions for each bounding box
        Returns:
            (torch.Tensor|np.ndarray): A tensor or numpy array containing the class IDs for each detection box. The shape (N,) where
                N is the number of boxes
        """
        return self.data[:,-1]
    
    @property
    @lru_cache(maxsize=2)
    def xywh(self)->torch.Tensor|np.ndarray:
        """
        Convert bounding boxes from [x1,y1,x2,y2] to [x,y,width,height] format
        Returns:
            (torch.Tensor | np.ndarray): Boxes in [x_center, y_center,width, height] format, where x_center, y_center are the coordinates
                of the center point of the bounding box, width and height are the dimensions of the bounding box and the shape of the 
                returned tensor is (N,4), where N is the number of boxes
        """
        return ops.xyxy2xywh(self.xyxy)
        
    @property
    @lru_cache(maxsize=2)
    def xyxyn(self)->torch.Tensor|np.ndarray:
        """
        Return normalized bounding box coordinates relative to the original image size
        This property returns the bounding box coordinates in [x1,y1,x2,y2] format, normalized to the range [0,1] based on
        the original image size
        
        Returns:
            (torch.Tensor|np.ndarray): Normalized bounding box coordinates with shape (N,4), where N is the number of boxes. 
                Each row containing [x1,y1,x2,y2] values normalized to [0,1]
        """
        xyxy=self.xyxy.clone() if isinstance(self.xyxy,torch.Tensor) else np.copy(self.xyxy) 
        xyxy[...,[0,2]]/=self.orig_shape[1]
        xyxy[...,[1,3]]/=self.orig_shape[0]
        return xyxy

    @property
    @lru_cache(maxsize=2)
    def xywhn(self)->torch.Tensor|np.ndarray:
        """
        Return normalized bounding boxes in [x, y, width, height] format

        This property calculates and returns the normalized bounding box coordinates in the format [x_center, y_center,
        width, height], where all values are relative to the original image dimensions

        Returns:
            (torch.Tensor | np.ndarray): Normalized bounding boxes with shape (N,4), where N is the number of boxes. Each
                row contains [x_center, y_center, width, height] values normalized to [0,1] based on the original image 
                dimensions
        """
        xywh=ops.xyxy2xywh(self.xyxy)
        xywh[...,[0,2]]/=self.orig_shape[1]
        xywh[...,[1,3]]/=self.orig_shape[0]
        return xywh

class Keypoints:
    """A class for storing and manipulating detection keypoints

    This class encapsulates functionality for handling keypoint data, including coordinate manipulation, normalization, and confidence
    values. It supports keypoint detection results with optional visibility information

    Examples:
        >>> keypoints_data=torch.rand(1,17,3) # 1 detection, 17 keypoints, (x, y, conf)
        >>> orig_shape=(480,640) # original image shape (height, width)
        >>> keypoints=Keypoints(keypoints_data, orig_shape)
        >>> print(keypoints.xy.shape) # Access xy coordinates
        >>> print(keypoints.conf) # Access confidence values
    """ 
    def __init__(self, keypoints:torch.Tensor | np.ndarray, orig_shape: tuple[int, int])->None:
        """Initialize the Keypoints object with detected keypoints and original image dimension

        The method processes the input keypoints tensor, handling 2D and 3D formats. For 3D tensors (x,y, confidence),
        it masks out low-confidence keypoints by setting their coordinates to zero

        Args:
            keypoints (torch.Tensor | np.ndarray): A tensor containing keypoint data. Shape can be either:
                - (num_objects, num_keypoints, 2) for x, y coordinates 
                - (num_objects, num_keypoints, 3) for x, y coordinates and confidence scores
            orig_shape (tuple[int, int]): The original image dimension (height, width)
        """ 
        assert isinstance(keypoints,(torch.Tensor, np.ndarray)),f'keypoints must be torch.Tensor or np.ndarray,but got {type(keypoints)}'
        self.data=keypoints
        self.orig_shape=orig_shape
        if keypoints.ndim==2: keypoints=keypoints[None,:]
        self.has_visible=self.data.shape[-1]==3

    @property
    @lru_cache(maxsize=1)
    def xy(self)->torch.Tensor | np.ndarray:
        """Return x, y coordinates of keypoints
        
        Returns:
            (torch.Tensor | np.ndarray): A tensor/array containing the x, y coordinates of keypoints with shape (N,K,2), where N is the
                number of detections and K is the number of kypoints per detection
        Notes:
            - The returned coordinates are in pixel units relative to the original image dimension
            - If keypoints were initialized with confidence values, only keypoints with confidence >=0.5 are returned
            - This property uses LRU caching to improve performance on repeated access
        """
        return self.data[...,:2]

    @property
    @lru_cache(maxsize=1)
    def xyn(self)->torch.Tensor | np.ndarray:
        """Return normalized coordinates (x,y) of keypoints relative to the original image size

        Returns:
            (torch.Tensor | np.ndarray): A tensor or array of (N, K, 2) containing normalized keypoint coordinates, where N is the number
                of instances, K is the number of keypoints, and the last dimension contains [x, y] values in the range [0,1].
        """
        xy=self.xy.clone() if isinstance(self.xy, torch.Tensor) else np.copy(self.xy)
        xy[...,0]/=self.orig_shape[1]
        xy[...,1]/=self.orig_shape[0]

    @property
    @lru_cache(maxsize=1)
    def conf(self)->torch.Tensor | np.ndarray | None:
        """Return confidence values for each keypoint

        Returns:
            (torch.Tensor | np.ndarray | None): A tensor containing confidence scores for each keypoint if available, otherwise None.
                Shape is (num_detections,num_keypoints) for batched data or (num_keypoints,) for single detection
        """
        return self.data[...,2] if self.has_visible else None
        
class Results:
    """
    A class for storing and manipulating inference results
    Examples:
        >>> results=model('path/to/image.jpg')
        >>> result=results[0] # Get the first result
        >>> boxes=result.boxes # Get the boxes for the first result
        >>> masks=result.masks # Get the masks for the first result
        >>> for result in results:
        >>>    result.plot() # Plot detection result
    """
    def __init__(self, orig_img: np.ndarray, path:str, names:dict[int, str], boxes:torch.Tensor|None=None,
                 masks:torch.Tensor | None=None, probs:torch.Tensor|None=None,keypoints:torch.Tensor|None=None,
                 obb:torch.Tensor|None=None, speed:dict[str,float]|None=None)->None:
        """
        Initialize the Results object to store and manipulate inference results
        Args:
            orig_img (np.ndarray): The original image as a numpy array
            path (str): The path to the image file
            names (dict): A dict of class names
            boxes (torch.Tensor | None): A 2D (N,4) tensor of bounding box coordinates for N boxes
            masks (torch.Tensor | None): A 3D tensor of detection masks, where each mask is a binary image
            probs (torch.Tensor | None): A 1D (N,) tensor of probabilities of each class for a classification task
            keypoints (torch.Tensor | None): A 2D tensor of keypoint coordinates for each detection
            obb (torch.Tensor | None): A 2D tensor of oriented bounding box coordinates for each detection
            speed (dict | None): A dict containing preprocess, inference, and postprocess speeds (ms/image)
        Notes:
            For the default pose model, keypoint indices for human body pose estimation are:
            0: Nose, 1: Left Eye, 2: Right Eye, 3: Left Ear, 4: Right Ear
            5: Left Shoulder, 6: Right Shoulder, 7: Left Elbow, 8: Right Elbow
            9: Left Wrist, 10: Right Wrist, 11: Left Hip, 12: Right Hip
            13: Left Knee, 14: Right Knee, 15: Left Ankle, 16: Right Ankle
        """
        self.orig_img=orig_img # HxWxC
        self.orig_shape=orig_img.shape[:2]

        self.boxes=Boxes(boxes, self.orig_shape) if boxes is not None else None # native size
        self.masks=Masks(masks, self.orig_shape) if masks is not None else None # native size of imgsz size
        self.keypoints=Keypoints(keypoints, self.orig_shape) if keypoints is not None else None
        self.obb=OBB(obb, self.orig_shape) if obb is not None else None
        self.speed=speed if speed is not None else {'preprocess':None, 'inference':None, 'postprocess':None}
        self.names=names
        self.path=path
        self.save_dir=None
        self._keys="boxes","keypoints" #"masks","probs","obb"

    def __getitem__(self, idx):
        """Return a Result object for a specific index of inference results

        Args:
            idx (int|slice): Index or slice to retrieve from the Result object
        Returns:
            (Results): A new Result object containing the specified subset of inference results
        Examples:
            >>> results=model("path/to/image.jpg") # Performce inference
            >>> single_result=results[0] # Get the first result
            >>> subset_results=results[1:4] # Get a slice of results
        """
        return self._apply("__getitem__", idx)
        
    def _apply(self, fn:str, *args, **kwargs):
        """Apply a function to all non-empty attributes and return a new Results object with modified attributes.

        This method is internally called by methods like .to(), .cuda(), .cpu(), etc.
        Args:
            fn (str): The name of the function to apply
            *args (Any): Variable length argument list to pass to the function
            **kwargs (Any): Arbitrary keyword arguments to pass to the function
        Returns:
            (Results): A new Result object with attributes modified by the applied function
        Examples:
            >>> results=model("path/to/image.jpg")
            >>> for result in results:
            >>>     result_cuda=result.cuda()
            >>>     result_cpu=result.cpu()
        """
        r=self.new()
        for k in self._keys:
            v=getattr(self,k)
            if v is not None: setattr(r, k, getattr(v, fn)(*args, **kwargs))

    def new(self):
        """Create a new Results object with the same image, path, names, and speed attributes.

        Returns:
            (Results): A new Results object with copied attributes from the original instance.

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> new_result = results[0].new()
        """
        return Results(orig_img=self.orig_img, path=self.path, names=self.names, speed=self.speed)

    def __len__(self)->int:
        """Return the number of detections in the Result object
        Returns:
            (int): The number of detections, determined by the length of the first non-empty attribute in (masks, probs,
                keypoints,or obb)
        Examples:
            >>> results=Results(orig_img, path, names, boxes=torch.rand(5,4))
            >>> len(results)
            5
        """
        for k in self._keys:
            v=getattr(self,k)
            if v is not None: return len(v)

    def update(self, boxes:torch.Tensor|None=None, keypoints:torch.Tensor|None=None, masks:torch.Tensor|None=None,
               probs:torch.Tensor|None=None, obb:torch.Tensor|None=None ):
        """Update the Results object with new detection data
        
        This method allows updating the boxes, keypoints, masks, probabilities, and oriented bounding boxes (OBB) of the Results object. 
        It ensures that boxes are clipped to the original image shape

        Args:
            boxes (torch.Tensor | None): A tensor of shape (N,6) containing bounding box coordinates and confidence socres. The format
                is (x1, y1, x2, y2, conf, class)
            masks (torch.Tensor | None): A tensor of shape (N,H,W) containing segmentation masks
            probs (torch.Tensor | None): A tensor of shape (num_classes, ) containing class probabilities
            obb (torch.Tensor | None): a tensor of shape (N, 5) containing oriented bounding box coordinates.
            keypoints (torch.Tensor | None): A tensor of shape (N, 17, 3) containing keypoints.
        Examples:
            >>> results=model("image.jpg")
            >>> new_boxes=torch.tensor([[100,100,200,200,0.9,0]])
            >>> results[0].update(boxes=new_boxes)
        """
        if boxes is not None: self.boxes=Boxes(ops.clip_boxes(boxes, self.orig_shape), self.orig_shape)
        if keypoints is not None: self.keypoints=Keypoints(keypoints, self.orig_shape)
            
    def save_txt(self, txt_file: str | Path, save_conf: bool = False) -> str:
        """Save detection results to a text file.

        Args:
            txt_file (str | Path): Path to the output text file.
            save_conf (bool): Whether to include confidence scores in the output.

        Returns:
            (str): Path to the saved text file.

        Examples:
            >>> from ultralytics import YOLO
            >>> model = YOLO("yolo11n.pt")
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            >>>     result.save_txt("output.txt")

        Notes:
            - The file will contain one line per detection or classification with the following structure:
              - For detections: `class confidence x_center y_center width height`
              - For classifications: `confidence class_name`
              - For masks and keypoints, the specific formats will vary accordingly.
            - The function will create the output directory if it does not exist.
            - If save_conf is False, the confidence scores will be excluded from the output.
            - Existing contents of the file will not be overwritten; new results will be appended.
        """
        is_obb = self.obb is not None
        boxes = self.obb if is_obb else self.boxes
        masks = self.masks
        probs = self.probs
        kpts = self.keypoints
        texts = []
        if probs is not None:
            # Classify
            [texts.append(f"{probs.data[j]:.2f} {self.names[j]}") for j in probs.top5]
        elif boxes:
            # Detect/segment/pose
            for j, d in enumerate(boxes):
                c, conf, id = int(d.cls), float(d.conf), int(d.id.item()) if d.is_track else None
                line = (c, *(d.xyxyxyxyn.view(-1) if is_obb else d.xywhn.view(-1)))
                if masks:
                    seg = masks[j].xyn[0].copy().reshape(-1)  # reversed mask.xyn, (n,2) to (n*2)
                    line = (c, *seg)
                if kpts is not None:
                    kpt = torch.cat((kpts[j].xyn, kpts[j].conf[..., None]), 2) if kpts[j].has_visible else kpts[j].xyn
                    line += (*kpt.reshape(-1).tolist(),)
                line += (conf,) * save_conf + (() if id is None else (id,))
                texts.append(("%g " * len(line)).rstrip() % line)

        if texts:
            Path(txt_file).parent.mkdir(parents=True, exist_ok=True)  # make directory
            with open(txt_file, "a", encoding="utf-8") as f:
                f.writelines(text + "\n" for text in texts)

        return str(txt_file)  

class OBB:
    """A class for storing and manipulating Oriented Bounding Boxes (OBB)

    This class provides functionality to handle oriented bounding boxes, including conversion between different formats, normalization, and 
    access to various properties of the boxes. It supports both tracking and non-tracking scenarios

    Examples:
        >>> boxes=torch.tensor([[100,50,150,100,30,0.9,0]]) # xywhr, conf, cls
        >>> obb=OBB(boxes, orig_shape=(480,640))
        >>> print(obb.xyxyxyxy)
        >>> print(obb.conf)
    """
    def __int__(self, boxes:torch.Tensor|np.ndarray, orig_shape:tuple[int, int])->None:
        """Initialize an OBB(Oriented Bounding Box) intsnace with oriented bounding box data and original image shape.

        This class stores and manipulates OBB for object detection tasks. It provides various properties and methods to access and transform
        the OBB data

        Args:
            boxes (torch.Tensor | np.ndarray): A tensor or numpy array containing the detection boxes, with shape (num_boxes, 7) or (num_boxes, 8).
                The last two columns contain confidence and class values. If present, the third last column contains track IDs, and the fifth
                column contains rotation, i.e., [x, y, w, h, rotation, confidence, class] or [x, y, w, h, rotation, track_id, confidence, class]
            orig_shape (tuple[int, int]): Original image size, in the format (height, width)
        """
        if boxes.ndim==1: boxes=boxes[None,:]
        n=boxes.shape[-1]
        assert n in {7,8}, f'expected 7 or 8 values but got {n}' # xywh, rotation, track_id, conf, cls
        self.is_track=n==8
        self.orig_shape=orig_shape
        self.data=boxes

    @property
    def xywhr(self)->torch.Tensor | np.ndarray:
        """Return boxes in [x_center, y_center, width, height, rotation] format
        Returns:
            (torch.Tensor|np.ndarray): A tensor or numpy array containing the oriented bounding boxes with format 
                [x_center, y_center, width, height, rotation]. The shape is (N,5) where N is the number of boxes.
        Examples:
            >>> results=model("image.jpg")
            >>> obb=results[0].obb
            >>> xywhr=obb.xywhr
            >>> print(xywhr.shape)
            torch.Size([3,5])
        """
        return self.data[:,:5]

    @property
    def conf(self)->torch.Tensor|np.ndarray:
        """Return the confidence scores for OBB

        This property retrieves the confidence values associated with each OBB detection. The confidence score represents teh model's certainty in
        the detection
        
        Returns:
            (torch.Tensor | np.ndarray): A tensor or numpy array of shape (N,) containing confidence socres for N detections, where each score is 
                in the range [0,1]
        """
        return self.data[:,-2]
        
    @property
    def cls(self)->torch.Tensor|np.ndarray:
        """Return the class values of the oriented bounding boxes
        Returns:
            (torch.Tensor|np.ndarray): A tensor or numpy array containing the class values for each oriented bounding box. The shape is (N,), 
                where N is the number of boxes\
        """
        return self.data[:,-1]

    @property
    def id(self)->torch.Tensor|np.ndarray|None:
        """Return the tracking IDs of the oriented bounding boxes (if available)
        Returns:
            (torch.Tensor|np.ndarray|None): A tensor or numpy array containing the tracking IDs for each OBB. Returns None if tracking IDs
                are not available
        """
        return self.data[:,-3] if self.is_track else None

    
    @property
    @lru_cache(maxsize=2)
    def xyxyxyxy(self)->torch.Tensor|np.ndarray:
        """Convert OBB format to 8-point (xyxyxyxy) coordinate format for rotated bounding boxes
        Returns:
            (torch.Tensor | np.ndarray): Rotated bounding boxes in xyxyxyxy format with shape (N, 4, 2), where N is the number of boxes, 
                4 for the four corners, and 2 for x,y. Each box is represented by 4 points (x,y), starting from top-left corner and moving 
                clockwise
        """
        return ops.xywhr2xyxyxyxy(self.xywhr)

    @property
    @lru_cache(maxsize=2)
    def xyxyxyxyn(self)->torch.Tensor|np.ndarray:
        """Convert rotated bounding boxes to normalized xyxyxyxy format

        Returns:
            (torch.Tensor|np.ndarray): Normalized rotated bounding boxes in xyxyxyxy format with shape (N, 4, 2), where
                N is the number of boxes. Each box is represented by 4 corners (x, y), normalized to relative to the original 
                image dimensions
        """
        xyxyxyxyn=self.xyxyxyxy.clone() if isinstance(self.xyxyxyxy, torch.Tensor) else np.copy(self.xyxyxyxy)
        xyxyxyxyn[...,0]/=self.orig_shape[1]
        xyxyxyxyn[...,1]/=self.orig_shape[0]
        return xyxyxyxyn

    @property
    @lru_cache(maxsize=2)
    def xyxy(self)->torch.Tensor|np.ndarray:
        """Convert oriented bounding boxes (OBB) to axis-aligned bounding boxes in xyxy format

        This property calculates the minimal enclosing rectangle for each oriented bounding box and returns it in xyxy format
        (x1, y1, x2, y2). This is useful for operations that require axis-aligned bounding boxes, such as IoU calculation with 
        non-rotated boxes.

        Returns:
            (torch.Tensor|np.ndarray): Axis-aligned bounding boxes in xyxy format with shape (N,4), where N is the number of boxes. 
                Each row contains [x1, y1, x2, y2] coordinates.
        Notes:
            - This method approximates the OBB by its minimal enclosing rectangle
            - The returned format is compatible with standard obect detection metrics and visualization tools
            - This propertu uses caching to improve performace for repeated access
        """
        x=self.xyxyxyxy[...,0]
        y=self.xyxyxyxy[...,1]
        return (torch.stack([x.amin(1), y.amin(1), x.amax(1), y.amax(1)], -1)
               if isinstance(x, torch.Tensor)
               else np.stack([x.min(1), y.min(1), x.max(1), y.max(1)],-1)
               )
        
class Masks:
    """A class for storing and manipulating detection masks

    This class provides functionality for handling segmentation masks, including methods for converting between pixel and normalized coordinates

    Examples:
        >>> masks_data=torch.rand(1,160,160)
        >>> orig_shape=(720,1280)
        >>> masks=Masks(masks_data, orig_shape)
        >>> pixel_coords=masks.xy
        >>> normalized_coords=masks.xyn
    """
    def __init__(self, masks:torch.Tensor|np.ndarray, orig_shape:tuple[int, int])->None:
        """Initialize the Masks class with detection mask data and the original image shape
        Args:
            masks (torch.Tensor|np.ndarray): Detection masks with shape (num_masks, height, width)
            orig_shape (tuple[int,int]): The original image shape as (height, width). Used for normalization
        """
        if masks.ndim==2: masks==masks[None,:]
        self.data=masks
        self.orig_shape=orig_shape

    @property
    @lru_cache(maxsize=1)
    def xyn(self)->list[np.ndarray]:
        """Return normalized xy-coordinates of the segmentation masks

        This property calculates and caches the normalized xy-coordinates of the segmentation masks. The coordinates are normalized relative to the 
        original image shape

        Returns:
            (list[np.ndarray]): A list of numpy arrays, where each array contains the normalized xy-coordinates of a single mask. Each array has 
                shape (N,2) where N is the number of points in the mask contour
        """
        return [
            ops.scale_coords(self.data.shape[1:], x, self.orig_shape, normalize=True)
            for x in ops.masks2segments(self.data)
        ]
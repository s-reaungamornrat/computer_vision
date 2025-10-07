from __future__ import annotations

from typing import Any

import cv2
import math
import copy
import torch
import random
import numbers
import numpy as np

from computer_vision.yolov11.instance.instance import Instances
from computer_vision.yolov11.utils.metrics import bbox_ioa
from computer_vision.yolov11.utils.ops import segment2box
from .utils import polygon2masks_overlap, polygon2masks

class BaseMixTransform:
    """
    Base class for mix transformations like Cutmix, MixUp and Mosaic.

    This class provides a foundation for implementing mix transformations on datasets. It handles the 
    probability-based application of transforms and manage the mixing of multiple images and labels
    """
    def __init__(self, dataset, pre_transform=None, p=0.0)->None:
        """
        Initialize the BaseMixTransform object for mix transformation like CutMix, MixUp and Mosaic

        This class serves as a base for implementing mix transformations in image processing pipelines
        Args:
            dataset (Any): The dataset object containing images and labels for mixing
            pre_transform (Callable | None): Optional transformation to apply before mixing to additional images but not the main image
            p (float): Probability of applying the mix transformation. Should be in the range [0., 1.]
        """
        self.dataset=dataset
        self.pre_transform=pre_transform
        self.p=p
        
    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        """
        Apply pre-processing transforms and cutmix/mixup/mosaic transforms to labels 

        This method determines whether to apply the mix transform based on a probabilty factor. 
        If applied, it selects additional images, applies pre-transforms if specified, and then
        performs the mix transform

        Args:
            labels (dict[str, Any]): A dict containing label data for an image
        Returns:
            (dict[str, Any]): The transformed label dict
        """
        if random.uniform(0,1)>self.p: return labels

        # Get index of one or three other images
        indices=self.get_indices()
        if isinstance(indices, int): indices=[indices]

        # Get images associated with the indices
        mix_labels=[self.dataset.get_image_and_label(i) for i in indices]

        if self.pre_transform is not None:
            for i, data in enumerate(mix_labels): mix_labels[i]=self.pre_transform(data)
        labels['mix_labels']=mix_labels

        # Mosaic, CutMix, MixUp
        labels=self._mix_transform(labels)
        labels.pop("mix_labels", None)
        return labels
        
    def _mix_transform(self, labels: dict[str, Any])->dict[str, Any]:
        raise NotImplementedError
        
    def get_indices(self):
        return random.randint(0,len(self.dataset)-1)

class Mosaic(BaseMixTransform):
    """
    Mosaic augmentation for image datasets

    This class performs mosaic augmentation by combining multiple (3, 4, or 9) images into a single mosaic image.
    This augmentation is applied to a dataset with a given probability
    """

    def __init__(self, dataset, imgsz:int=640, p:float=1.0, n:int=4, buffer_enabled:bool=False):
        """
        Initialize the Mosaic augmentation
        Args:
            dataset (Any): The dataset on which the mosaic augmentation is applied.
            imgsz (int): Image size (height and width) after mosaic pipeline of a single image
            p (float): Probability of applying the mosaic augmentation. Must be in the range 0-1
            n (int): The grid size, either 3 (for 1x3) or 4 (for 2x2) or 9 (for 3x3)
            buffer_enabled (bool): Whether to use image buffer
        """
        assert 0.<=p<=1.0, f'The probability must be in the range [0, 1] but got {p}.'
        assert n in {3,4,9}, "grid must be 3, 4, or 9"
        super().__init__(dataset=dataset, pre_transform=None, p=p)
        self.imgsz=imgsz
        self.border=(-imgsz//2, -imgsz//2) # width, height
        self.n=n
        self.buffer_enabled=buffer_enabled
        
    def get_indices(self):
        """
        Return a list of random indices from the dataset

        This method selects random image indices either from a buffer or from the entire dataset, depending on
        the `buffer` parameter. 
        Returns:
            (list[int]): A list of random image indices. The length of the list is n-1, where n is the number
                of images used in mosaic.
        """
        # select images from buffer
        if self.buffer_enabled and len(self.dataset.buffer)>self.n-1: 
            return random.choices(list(self.dataset.buffer), k=self.n-1) # with replacement
        else: return np.random.randint(low=0, high=len(self.dataset), size=self.n-1)
            
    def _mix_transform(self, labels: dict[str, Any])->dict[str, Any]:
        """
        Apply mosaic augmentation to the input image and labels

        This method combines 3, 4, or 9 images into a single mosaic image based on the `n` attribute.
        It ensures the retangular annotations are not present and that there are other images avialable
        for mosaic augmentation.

        Args:
            labels (dict[str, Any]): A dict containing image and annotations. Expected keys include:
                -`rect_shape`: Should be None as rect and mosaic are mutually exclusive
                -`mix_labels`: A list of dict containing data for other images to be used in the mosaic
        Returns:
            (dict[str, Any]): A dict containing the mosaic-augmented image and updated annotations.
        Raise:
            AssertionError: If `rect_shape` is not None or if `mix_labels` is empty
        Examples:
            >>> mosaic=Mosaic(dataset, imgsz=640, p=1., n=4)
            >>> augmented_data=mosaic._mix_transform(labels)
        """
        assert labels.get("rect_shape") is None, "rect and mosaic are mutually exclusive"
        assert len(labels.get("mix_labels", [])), "There are no other images for mosaic augment"
        return (self._mosaic3(labels) if self.n==3 else self._mosaic4(labels) if self.n==4 else self._mosaic9(labels))
    
    @staticmethod
    def _update_labels(labels: dict[str, Any], padw:int, padh: int)->dict[str, Any]:
        """
        Update label coordinates with padding values
        
        This method adjusts the bounding box coordinates of object instances in the labels by
        adding padding values. It also denormalizes the coordinates if they were previously 
        normalized
        Args:
            labels (dict[str, Any]): A dict containing image and instance information
            padw (int): Padding width to be added to the x-coordinates
            padh (int): Padding height to be added to the y-coordinates
        Returns:
            (dict[str, Any]): Updated label dict with adjusted instance coordinates, with bounding boxes in unnormalized xyxy
                (i.e., in pixel units)
        """
        h, w=labels['img'].shape[:2]
        labels['instances'].convert_bbox(format='xyxy')
        labels['instances'].denormalize(w, h)
        labels['instances'].add_padding(padw, padh)
        return labels

    def _cat_labels(self, mosaic_labels: list[dict[str, Any]])->dict[str, Any]:
        """
        Concatenate amd process labels for mosaic augmentation

        This method combines labels from multiple images used in mosaic augmentation, 
        clips instances to the mosaic border, and removes zero-area boxes.
        Args:
            mosaic_labels (list[dict[str, Any]]): A list of label dicts for each image in the mosaic
        Returns:
            (dict[str, Any]): A dict containing concatenated and processed labels for the mosaic image,including:
                - im_file (str): File path of the first image in the mosaic
                - ori_shape (tuple[int, int]): Original shape of the first image.
                - resized_shape (tuple[int, int]): Shape of the mosaic image (imgsz*2, imgsz*2)
                - cls (np.ndarray): Concatenated class labels.
                - instances (Instances): Concatenated instance annotations
                - mosaic_border (tuple[int, int]): Mosaic border size
                - texts (list[str], optional): Text labels if present in the original labels
        """
        if not mosaic_labels: return dict()
        cls, instances, = [], []
        imgsz=self.imgsz*2  # mosaic image size
        for labels in mosaic_labels:
            cls.append(labels['cls'])
            instances.append(labels['instances'])
        # Final labels
        final_labels={
            'im_file':mosaic_labels[0]['im_file'],
            'ori_shape':mosaic_labels[0]['ori_shape'],
            'resized_shape':(imgsz, imgsz),
            'cls':np.concatenate(cls, 0),
            'instances': Instances.concatenate(instances, axis=0),
            'mosaic_border': self.border,
        }
        final_labels['instances'].clip(imgsz, imgsz)
        good=final_labels['instances'].remove_zero_area_boxes()
        final_labels['cls']=final_labels['cls'][good]
        if "texts" in mosaic_labels[0]: final_labels["text"]=mosaic_labels[0]["texts"]
            
        return final_labels
        
    def _mosaic3(self, labels: dict[str, Any]) -> dict[str, Any]:
        """
        Create a 1x3 mosaic image by combining 3 images in a horizontal layout, with the
        main image in the center and two additional images on either side. It's part of
        the Mosaic augmentation technique used in object detection
        Args:
            labels (dict[str, Any]): A dict containing image and label information for the main (center)
                image. Must include `img` key with the image array, and `mix_labels` key with a list of
                two dicts containing information for the side images.
        Returns:
            (dict[str, Any]): A dict with the mosaic image and updated labels. Keys include:
                - `img` (np.ndarray): The mosaic image array with shape (H,W,C)
                - Other keys from the input labels, updated to reflect the new image dimension with boxes
                    defined as xyxy in pixel units (can be verified via 
                                                    final_labels['instances']._bboxes.format, 
                                                    final_labels['instances'].normalized)
        """
        mosaic_labels=[]
        s=self.imgsz
        for i in range(self.n):
            labels_patch=labels if i==0 else labels['mix_labels'][i-1]
            # Load image
            img=labels_patch['img']
            h, w=labels_patch.pop('resized_shape')
            
            # Place img in img3
            if i==0: # center
                # base image with 3 tiles
                img3=np.full((s*3, s*3, img.shape[2]), 114, dtype=np.uint8)     
                h0, w0 = h, w
                c=s, s, s+w, s+h # xmin, ymin, xmax, ymax (base) coordinates
            elif i==1: c=s+w0, s, s+w0+w, s+h # right
            elif i==2: c=s-w, s+h0-h, s, s+h0 # left
        
            padw, padh=c[:2]
            x1,y1,x2,y2=(max(x, 0) for x in c) # allocate coordinates
            img3[y1:y2, x1:x2]=img[(y1-padh):, (x1-padw):]
        
            # Labels assuming imgsz*mosaic size
            labels_patch=self._update_labels(labels_patch, padw+self.border[0], padh+self.border[1])
            mosaic_labels.append(labels_patch)
        final_labels=self._cat_labels(mosaic_labels)
        final_labels['img']=img3[-self.border[0]:self.border[0], -self.border[1]:self.border[1]]
        
        return final_labels
        
    def _mosaic4(self, labels:dict[str, Any]) -> dict[str, Any]:
        """
        Create a 2x2 image mosaic from four input images
    
        This method combines four images into a single mosaic image by placing them in a 2x2 grid. 
        It also updates the corresponding labels for each image in the mosaic.
    
        Args:
            labels (dict[str, Any]): A dict containing image data and labels for the base image (index 0) and three
                additional images (indices 1-3) on the `mix_labels` key.
        Returns:
            (dict[str, Any]): A dict containing the mosaic image and updated labels.  Keys include:
                - `img` (np.ndarray): The mosaic image array with shape (H,W,C)
                - Other keys from the input labels, updated to reflect the new image dimension with boxes
                    defined as xyxy in pixel units (can be verified via 
                                                    final_labels['instances']._bboxes.format, 
                                                    final_labels['instances'].normalized)
        """
        mosaic_labels=[]
        s=self.imgsz
        # mosaic center x, y
        yc, xc=(int(random.uniform(-x, 2*s+x)) for x in self.border)
        
        for i in range(self.n):
            labels_patch=labels if i==0 else labels["mix_labels"][i-1]
            # load image
            img=labels_patch['img']
            h, w=labels_patch.pop("resized_shape")
        
            # Place img in img4
            if i==0: # top left
                img4=np.full((s*2, s*2, img.shape[2]), 114, dtype=np.uint8) # base image with 4 tiles
                # xmin, ymin, xmax, ymax of large image
                x1a, y1a, x2a, y2a=max(xc-w, 0), max(yc-h, 0), xc, yc
                # xmin, ymin, xmax, ymax of small image
                x1b, y1b, x2b, y2b=w-(x2a-x1a), h-(y2a-y1a), w, h # cut the top left part if needed
            elif i==1: # top right
                x1a, y1a, x2a, y2a = xc, max(yc-h, 0), min(xc+w, s*2), yc
                x1b, y1b, x2b, y2b = 0, h-(y2a-y1a), min(w, x2a-x1a), h # cut the top right if needed
            elif i==2: # bottom left
                x1a, y1a, x2a, y2a = max(xc-w, 0), yc, xc, min(s*2, yc+h)
                x1b, y1b, x2b, y2b = w-(x2a-x1a), 0, w, min(y2a-y1a, h) # cut the bottom left if needed
            elif i==3: # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc+w, s*2), min(s*2, yc+h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a-x1a), min(y2a-y1a, h) # cute the bootom right if needed
        
            img4[y1a:y2a, x1a:x2a]=img[y1b:y2b, x1b:x2b]
            padw=x1a-x1b # cut original box position by x1b and add x1a offset
            padh=y1a-y1b # cut original box position by y1b and add y1a offset
        
            labels_patch=self._update_labels(labels_patch, padw, padh)
            mosaic_labels.append(labels_patch)
        final_labels=self._cat_labels(mosaic_labels)
        final_labels["img"]=img4
    
        return final_labels

    def _mosaic9(self, labels:dict[str, Any])->dict[str, Any]:
        """
        Create a 3x3 mosaic image from the input image and eight additional images.
    
        This method combines nine images into a single mosaic image. The input image is
        placed at the center, and eight additional images from the dataset are placed 
        around it in a 3x3 grid pattern.
    
        Args:
            labels (dict[str, Any]): A dict containing the input image and its associated labels.
                It should have the following keys:
                - `img` (np.ndarray): The input image
                - `resized_shape` (tuple[int,int]): The height, width of the resized image
                - `mix_labels` (list[dict]): A list of dict containing information about the 
                    additional eight images, each with thhe same structure as the input labels.
        Returns:
            (dict[str, Any]) A dict containing the mosaic image and updated labels.  Keys include:
                - `img` (np.ndarray): The mosaic image array with shape (H,W,C)
                - Other keys from the input labels, updated to reflect the new image dimension with boxes
                        defined as xyxy in pixel units (can be verified via 
                                                        final_labels['instances']._bboxes.format, 
                                                        final_labels['instances'].normalized)
        """
        mosaic_labels=[]
        s=self.imgsz
        # previous height, width 
        hp=wp=-1
        
        for i in range(self.n):
            labels_patch=labels if i==0 else labels['mix_labels'][i-1]
            # Load image
            img=labels_patch['img']
            h, w=labels_patch.pop('resized_shape')
        
            # Place img in img9
            # Note that we do not check whether x2,y2 go over img9 or not since numpy takes care that
            # and only return element within the range, e.g., img9 of size 20x20 and x1,y1,x2,y2=0,0,50,80
            # img9[y1:y2,x1:x2] will be equal to img9 without throwing error
            if i==0: # center
                img9=np.full((s*3, s*3, img.shape[2]), 114, dtype=np.uint8)
                h0, w0=h, w
                c=s,s,s+w,s+h # xmin, ymin, xmax, ymax 
            elif i==1: c=s,s-h,s+w, s # top center
            elif i==2: c=s+wp,s-h, s+wp+w, s # top right
            elif i==3: c=s+w0, s, s+w0+w, s+h # center right
            elif i==4: c=s+w0, s+hp, s+w0+w, s+hp+h # bottom right
            elif i==5: c=s+w0-w, s+h0, s+w0, s+h0+h  # bottom center
            elif i==6: c=s+w0-wp-w, s+h0, s+w0-wp, s+h0+h # bottom left
            elif i==7: c=s-w, s+h0-h, s, s+h0 # center left
            elif i==8: c=s-w, s+h0-hp-h, s, s+h0-hp # top left
        
            padw, padh=c[:2]
            x1,y1,x2,y2=(max(x, 0) for x in c) # allocate coordinates
            # Note that we do not check whether x2,y2 go over img9 or not since numpy takes care that
            # and only return element within the range, e.g., img9 of size 20x20 and x1,y1,x2,y2=0,0,50,80
            # img9[y1:y2,x1:x2] will be equal to img9 without throwing error
        
            #Image
            img9[y1:y2, x1:x2]=img[(y1-padh):, (x1-padw):] 
            hp,wp=h,w # previous height, width for the next iteration
        
            # Labels assuming mosaic-size=imgsz*2
            labels_patch=self._update_labels(labels_patch, padw+self.border[0], padh+self.border[1])
            mosaic_labels.append(labels_patch)
        
        final_labels=self._cat_labels(mosaic_labels)
        final_labels['img']=img9[-self.border[0]:self.border[0],-self.border[1]:self.border[1]]
    
        return final_labels

class CopyPaste(BaseMixTransform):
    """
    CopyPaste class for applying Copy-Paste augmentation to image datasets.

    This class implements the Copy-Paste augmentation technique as described in the paper "Simple Copy-Paste is a Strong Data Augmentation Method
    for Instance Segmentation" (https://arxiv.org/abs/2012.07177). It combines objects from different images to create new training samples.

    It has 2 modes `mixup` and `flip`. For `mixup`, it requires that the two images to mix must have the same size so `pre_transform` must be applied
    to the main and additional images to make the two images having the same size. For `flip`, since the same image is used, `pre_transform` 
    is not needed. 

    It requires segments to present in `instances` since segments are used to define regions to be copied and pasted onto the other image. It selects
    regions based non-overlapping bounding boxes in the two images or based on bounding boxes that are less overlapped
    """
    def __init__(self, dataset=None, pre_transform=None, p:float=0.5, mode:str='flip')->None:
        """
        Initialize CopyPaste object with dataset, pre_transform, and probability of applying mix transformation
        """
        super().__init__(dataset=dataset, pre_transform=pre_transform, p=p)
        assert mode in {'flip', 'mixup'}, f'Expected `mode` to be `flip` or `mixup`, but got {mode}'
        self.mode=mode

    def _mix_transform(self, labels:dict[str, Any])->dict[str, Any]:
        """
        Apply Copy-Paste augmentation to combine objects from another image into the current image.
        """
        labels2=labels['mix_labels'][0]
        return self._transform(labels, labels2)

    def _transform(self, labels1: dict[str, Any], labels2:dict[str, Any]={})->dict[str, Any]:
        """
        Apply Copy-Paste augmentation to combine objects from another image into the current image. If just 1 image
        provided, create a flipped left-right version of the input and copy labels to the original input. Either cases
        copy and paste only occur if bounding boxes in two images are less overlapped
        Args:
            labels1 (dict[str, Any]): A dict containing the original image and label information
            labels2 (dict[str, Any]): A dict containing the original image and label information
        Returns:
            (dict[str, Any]): An updated dict containing the updated image and label information
        """
        
        im=labels1['img']
        if "mosaic_border" not in labels1: im=im.copy() # avoid modifying original non-mosaic image
        cls=labels1['cls']
        h,w=im.shape[:2]
        instances=labels1.pop('instances')
        instances.convert_bbox(format='xyxy')
        instances.denormalize(w, h)
        
        instances2=labels2.pop('instances', None)
        if instances2 is None:
            instances2=copy.deepcopy(instances)
            instances2.fliplr(w)
        
        # See whether there are boxes in two instances that do not intersect or having small
        # intersection
        # MxN = Mx4 Nx4
        ioa=bbox_ioa(instances2.bboxes, instances.bboxes) # intersection over area
        # indices of boxes in instances2 that less intersect boxes in instances
        indices=np.nonzero((ioa<0.3).all(axis=1))[0] 
        n=len(indices)
        # sorted_idx of indices that lead to boxes in instances2 that less intersect with boxes in 
        # instances
        sorted_idx=np.argsort(ioa.max(axis=1)[indices]) # sort from small to large
    
        indices=indices[sorted_idx] # get indices of boxes in instances 2 that less intersect to instances
        # place holder for bool-mask masking region in im to be copied-pasted by image in labels2
        im_new=np.zeros(im.shape, np.uint8) 
        for j in indices[:round(self.p*n)]:
            cls=np.concatenate((cls, labels2.get('cls', cls)[[j]]), axis=0)
            instances=Instances.concatenate((instances, instances2[[j]]), axis=0)
            cv2.drawContours(im_new, instances2.segments[[j]].astype(np.int32), -1, (1,1,1), cv2.FILLED)
    
        result=labels2.get('img', cv2.flip(im, 1)) # flip horizontally, i.e. around y axis
        # cv2.clip would eliminate the last dimension for grayscale images
        if result.ndim==2: result=result[...,None]
        
        # area or region in im to be replace by the same region in result
        i=im_new.astype(bool)
        im[i]=result[i]
        
        labels1['img']=im
        labels1['cls']=cls
        labels1['instances']=instances
    
        return labels1

    def __call__(self, labels:dict[str, Any])->dict[str, Any]:
        """
        Apply Copy-Paste augmentation to an image and its labels. For `mixup`, it requires that
        the two images to mix must have the same size so `pre_transform` must to apply to the main
        and additional images to make the two images having the same size. For `flip`, since the same
        image is used, `pre_transform` is not needed.
        Args:
            labels (dict[str, Any]): A dict containing the original image and label information
        Returns:
            (dict[str, Any]): An updated dict containing the updated image and label information
        """
        if len(labels['instances'].segments)==0 or self.p==0: return labels
        
        if self.mode=='flip': return self._transform(labels)
        
        # Get an additional image index
        index=self.get_indices()
        if isinstance(index, int): index=[index]
        
        # Get image information 
        mix_labels=[self.dataset.get_image_and_label(i) for i in index]
        
        if self.pre_transform is not None:
            for i, data in enumerate(mix_labels): mix_labels[i]=self.pre_transform(data)
        labels['mix_labels']=mix_labels
        
        labels=self._mix_transform(labels)
        labels.pop('mix_labels', None)
        return labels
        
class MixUp(BaseMixTransform):
    """
    Apply MixUp augmentation to image datasets.

    This class implements the MixUp augmentation technique as described in the paper 
    mixup: Beyond Empirical Risk Minimization(https://arxiv.org/abs/1710.09412)
    MixUp combines two images and their labels using a random weight.

    Note that to apply this, `pre_transform` must make sure that the images to mix up have the same size, e.g., use LetterBox
    Example:
        >>> labels=dataset.get_image_and_label(index=90)
        >>> pre_transform=LetterBox(new_shape=(dataset.imgsz, dataset.imgsz), scaleup=False)
        >>> mixup=MixUp(dataset, pre_transform=pre_transform, p=0.9)
        >>> final_labels=mixup(labels)
    """
    def __init__(self, dataset, pre_transform=None, p:float=0.)->None:
        """
        Initialize the MixUp augmentation object

        MixUp is an image augmentation technique that combines two images by taking a weighted sum of their 
        pixel values and labels. This implementation is designed for use with the Ultralytics YOLO framework.

        Args:
            dataset (Any): The dataset to which MixUp augmentation will be applied
            pre_transform (Callable | None): Optional transformation to apply to images before MixUp 
                to additional images but not the main image
            p (float): Probability of applying MixUp augmentation to an image. Must be in the range [0,1]
        """

        super().__init__(dataset=dataset, pre_transform=pre_transform, p=p)
        
    def _mix_transform(self, labels:dict[str, Any])->dict[str, Any]:
        """
        This method implements the MixUp augmentation technique as described in the paper
        mixup: Beyond Empirical Risk Minimization (https://arxiv.org/abs/1710.09412).

        Args:
            labels (dict[str, Any]): A dict containing the original image and label information
        Returns:
            (dict[str, Any]): A dict containing the mixed-up image and combined label information, with boxes in the same format
                as input if pre_transform is None. Otherwise, boxes are in format and unit defined by pre_transform
        """
        r=np.random.beta(32.,32.) # mixup ratio, alpha=beta=32.
        labels2=labels['mix_labels'][0]
        labels['img']=(labels['img']*r + labels2['img']*(1-r)).astype(np.uint8)
        labels['instances']=Instances.concatenate([labels['instances'], labels2['instances']], axis=0)
        labels['cls']=np.concatenate([labels['cls'], labels2['cls']], axis=0)
        return labels

class CutMix(BaseMixTransform):
    """
    Apply CutMix augmentation to image datasets as described in the paper https://arxiv.org/abs/1905.04899.

    CutMix combines two images by replacing a random rectangular region of one image with the corresponding region
    from another image, and adjusts the labels proportionally to the area of the mixed region.
    """
    def __init__(self, dataset, pre_transform=None, p:float=0., beta:float=1., num_areas:int=3)->None:
        """
        Initialize the CutMix augmentation object.
        Args:
            dataset (Any): The dataset to which CutMix augmentation will be applied
            pre_transform (Callable | None): Optional transform to apply before CutMix
            p (float): Probability of applying CutMix augmentation
            beta (float): Beta distribution parameter for sampling the mixing ratio
            num_areas (int): Number of areas to try to cut and mix
        """
        super().__init__(dataset=dataset, pre_transform=pre_transform, p=p)
        self.beta=beta
        self.num_areas=num_areas

    def _rand_bbox(self, width:int, height:int)-> tuple[int, int, int, int]:
        """
        Generate random bounding box coordinates for the cut region.
        Args:
            width (int): Width of the image.
            height (int): Height of the image.
        Returns:
            (tuple[int, int, int, int]): (x1, y1, x2, y2) coordinates of the bounding box in pixel units
        """
        # Sampling mixing ratio from Beta distribution
        lam=np.random.beta(self.beta, self.beta)
        
        cut_ratio=np.sqrt(1.-lam)
        cut_w=int(width*cut_ratio)
        cut_h=int(height*cut_ratio)
        
        # Random center
        cx=np.random.randint(width)
        cy=np.random.randint(height)
        
        # Bounding box coordinates
        x1=np.clip(cx-cut_w//2, 0, width)
        y1=np.clip(cy-cut_h//2, 0, height)
        x2=np.clip(cx+cut_w//2, 0, width)
        y2=np.clip(cy+cut_h//2, 0, height)
        
        return x1, y1, x2, y2 

    def _mix_transform(self, labels: dict[str, Any])->dict[str, Any]:
        """
        Apply CutMix augmentation to the input labels
        
        Args:
            labels (dict[str,Any]): A dict containing the original image and label information
        Returns:
            (dict[str, Any]): A dict containing the mixing image and adjusted labels
        """
        
        h, w=labels['img'].shape[:2]
        
        # A x 4 where A is the number of areas and 4 for x1,y1,x2,y2 in pixel units
        cut_areas=np.asarray([self._rand_bbox(w, h) for _ in range(self.num_areas)], dtype=np.float32)
        assert labels["instances"]._bboxes.format=='xyxy', f'labels must be in xyxy format but got {labels["instances"]._bboxes.format}'
        assert not labels["instances"].normalized, f'labels be denormalized but got normalized {labels["instances"].normalized}'
        ioa1=bbox_ioa(cut_areas, labels["instances"].bboxes) # AxN where N is the number of boxes in labels[instances]
        idx=np.nonzero(ioa1.sum(axis=1)<=0)[0] # AxN -> A
        if len(idx)==0: return labels
        
        labels2=labels.pop('mix_labels')[0] # mix_labels is a list of 1 dict so we just get the dict
        area=cut_areas[np.random.choice(idx)] # randomly select one
        
        assert labels2['img'].shape[:2]==labels['img'].shape[:2], f'main and additional images must be of the same size but got {labels["img"].shape[:2]}, {labels2["img"].shape[:2]} respectively'
        assert labels2["instances"]._bboxes.format=='xyxy', f'additional labels must be in xyxy format but got {labels2["instances"]._bboxes.format}'
        assert not labels2["instances"].normalized, f'additional labels be denormalized but got normalized {labels2["instances"].normalized}'
        # (1x4, Px4)->1xP -> P
        ioa2=bbox_ioa(area[None], label2["instances"].bboxes).squeeze(0)
        indices2=np.nonzero(ioa2>=(0.01 if len(label2["instances"].segments) else 0.1))[0]
        if len(indices2)==0: return labels
        
        instances2=labels2["instances"][indices2]
      
        # Apply CutMix
        x1,y1,x2,y2=area.astype(np.int32)
        labels["img"][y1:y2,x1:x2]=labels2["img"][y1:y2, x1:x2]
        
        # Restrain instances2 to the random bounding border
        instances2.add_padding(-x1, -y1)
        instances2.clip(w=x2-x1, h=y2-y1)
        instances2.add_padding(x1,y1)
        
        # labels['cls'] is Nx1 where N is the number of objects
        labels['cls']=np.concatenate([labels['cls'], labels2['cls'][indices2]], axis=0)
        labels['instances']=Instances.concatenate([labels['instances'], instances2], axis=0)
    
        return labels
        
class LetterBox:
    """
    Resize image and padding for detection, instance segmentation, pose.

    This class resizes and pads images to a specified shape while preserving aspect ratio. It also updates
    corresponding labels and bounding boxes
    """
    def __init__(self, new_shape:tuple[int,int]=(640,640), auto:bool=False, scale_fill:bool=False, scaleup:bool=True,
        center:bool=True, stride=32, padding_value:int=114, interpolation=cv2.INTER_LINEAR):
        """
        Initialize LetterBox object for resizing and padding images
        This class is designed to resize and pad images for object detection, instance segmentation, and pose 
        estimation tasks. It supports various resizing modes including auto-sizing, scale-fill, and letterboxing.
        Args:
            new_shape(tuple[int, int]): Target size (height, width) for resized image
            auto (bool): If True, use minimum rectangle to resize. If False, use new_shape directly
            scale_fill (bool): If True, stretch the image to new_shape without padding
            scaleup (bool): If True, allow scaling up. If False, only scale down
            center (bool): If True, center the placed image. If False, place image in top-left corner.
            stride (int): Stride of the model (e.g., 32 for yolov11)
            padding_value (int): Value for padding the image. Default is 114
            interpolation (int): Interpolation method for resizing. Default is cv2.INTER_LINEAR.
        Examples:
            >>> letterbox=LetterBox(new_shape=(640,640), auto=False, scale_fill=False, scaleup=True, stride=32)
            >>> resized_img=letterbox(original_img)
        """
        self.new_shape=new_shape
        self.auto=auto
        self.scale_fill=scale_fill
        self.scaleup=scaleup
        self.stride=stride
        self.center=center # put the image in the middle or top-left
        self.padding_value=padding_value
        self.interpolation=interpolation
    
    def __call__(self,labels:dict[str, Any]=None, image:np.ndaary=None)->np.ndarray | dict[str, Any]:
        """
        Resize and pad an image for object detection, instance segmentation, or pose estimation tasks.

        This method applies letterboxing to the input image, which involves resizing the image while 
        maintaining its aspect ratio and adding padding to fit the new shape. It also updates any associated 
        labels accordingly.

        Args:
            image (np.ndarray | None): The input image as a numpy array. If None, the image is taken from `labels`
            labels (dict[str, Any] | None): A dict containing imag edata and labels or None
        Returns:
            (np.ndarray | dict[str, Any]): If `image` is provided, return the resized and padded image; otherwise,
                return an updated dict with the resized and padded image and update labels, with boxes in xyxy and in pixel units
        Examples:
            >>> letterbox=LetterBox(new_shape=(640,640))
            >>> result=letterbox(labels={'img':np.zeros((480,640,3)), 'instances':Instances(...)})
            >>> resized_img=result['img']
            >>> updated_instances=result['instances']
        """
        if labels is None: labels=dict()
        img=labels.get('img') if image is None else image
        assert isinstance(img, np.ndarray), f'img must be np.ndarry but got {type(img)}: {img}'
        shape=img.shape[:2] # H, W
        new_shape=labels.pop('rect_shape', self.new_shape) # if rect_shape not in `labels`, return self.new_shape
        if isinstance(new_shape, numbers.Number): new_shape=(new_shape, new_shape)

        # scale ratio: new/old
        r=min(n/o for n,o in zip(new_shape, shape))
        # only scale down, do not scale up for better mAP
        if not self.scaleup: r=min(r, 1.)

        ratio=r,r
        # computing padding
        new_unpad=[int(round(s*r)) for s in shape] # H, W
        dh,dw=(s-u for s, u in zip(new_shape, new_unpad)) 
        if self.auto: # minimum rectangle
            dh,dw=np.mod(dh, self.stride), np.mod(dw, self.stride)
        elif self.scale_fill: # stretch
            dh=dw=0.
            new_unpad=new_shape # H W
            ratio=(n/o for n, o in zip(new_shape[::-1], shape[::-1])) # width, height ratio

        if self.center: # divide padding into 2 sides
            dw/=2; dh/=2
        if shape!=new_unpad: # resize
            img=cv2.resize(img, new_unpad[::-1], interpolation=self.interpolation) # input size to CV must be width, height
            if img.ndim==2: img=img[...,None]

        top, bottom=int(round(dh-0.1)) if self.center else 0, int(round(dh+0.1))
        left, right=int(round(dw-0.1)) if self.center else 0, int(round(dw+0.1))
        h, w, c=img.shape
        if c==3: img=cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(self.padding_value,)*3)
        else: # multispectral 
            pad_img=np.full((h+top+bottom, w+left+right, c), fill_value=self.padding_value, dtype=img.dtype)
            pad_img=img[top:(top+h), left:(left+w)]
            img=pad_img

        if labels.get('ratio_pad'): labels['ratio_pad']=(labels['ratio_pad'], (left, top)) # for evaluation
        if len(labels):
            labels=self._update_labels(labels, ratio, left, top)
            labels['img']=img
            labels['resized_shape']=new_shape
            return labels
        return img

    @staticmethod
    def _update_labels(labels:dict[str, Any], ratio:tuple[float, float], padw:float, padh:float)->dict[str, Any]:
        """
        Update labels after applying letterboxing to an image

        This method modifies the bounding box coordinates of instances in the labels to account for resizing and padding applied 
        during letterboxing.
        Args:
            labels (dict[str, Any]): A dict containing image labels and instances
            ratio (tuple[float, float]): Scaling ratios (width, height) applied to the image
            padw (float): Padding width added to the image
            padh (float): Padding height added to the image
        Returns:
            (dict[str, Any]): Update labels dict with modified instance coordinates
        Examples:
            >>> letterbox=LetterBox(new_shape=(640,640))
            >>> labels={'instances':Instances(...)}
            >>> ratio=(.5,.5)
            >>> padw,padh=10,20
            >>> updated_labels=letterbox._update_labels(labels, ratio, padw, padh)
        """
        labels['instances'].convert_bbox(format='xyxy')
        labels['instances'].denormalize(*labels['img'].shape[:2][::-1])
        labels['instances'].scale(*ratio)
        labels['instances'].add_padding(padw, padh)
        return labels

class RandomPerspective:
    """
    Implement random perspective and affine transformations on images and corresponding annotations.

    This class applies random rotations, translations, scaling, shearing, and perspective transformations to images and their associated
    bounding boxes, segments, and keypoints. It can be used as part of an augmentation pipeline for object detection and instance segmentation
    tasks.
    """
    def __init__(self, degrees:float=0., translate:float=0.1, scale:float=0.5, shear:float=0.,perspective:float=0., 
                 border:tuple[int,int]=(0,0),pre_transform=None):
        """
        Initialize RandomPerspective with transform parameters
        Args:
            degrees (float): Degree range for random rotations
            translate (float): Fraction of total width and height for random translation
            scale (float): Scaling factor interval, e.g., a scale factor of 0.5 allows resizing between 50-150%
            shear (float): Shear intensity (angle in degrees)
            perspective (float): Perspective distortion factor
            border (tuple[int,int]): Tuple specifying mosaic border (top/bottom, left/right) -> height, width direction
            pre_transform (Callable | None): Function/transform to apply to the image before starting the random
                transformation
        """
        self.degrees=degrees
        self.translate=translate
        self.scale=scale
        self.shear=shear
        self.perspective=perspective
        self.border=border
        self.pre_transform=pre_transform
        
    def affine_transform(self, img: np.ndarray, border: tuple[int, int])->tuple[np.ndarray, np.ndarray, float]:
        """
        Apply a sequence of affine transformation centered around the image center
        Args:
            img (np.ndarray): Input image of size HxWxC to be transformed
            border (tuple[int,int]): Border dimensions for the transformed image
        Returns:
            img (np.ndarray): Transformed image of size HxWxC
            M (np.ndarray): 3x3 transformation matrix
            s (float): Scale factor applied during the transformation
        """
        # Center
        C=np.eye(3, dtype=np.float32)
        C[0,2]=-img.shape[1]/2 # x translation in pixels
        C[1,2]=-img.shape[0]/2 # y translation in pixels

        # Perspective
        P=np.eye(3, dtype=np.float32)
        P[2,0]=random.uniform(-self.perspective, self.perspective) # x perspective about y
        P[2,1]=random.uniform(-self.perspective, self.perspective) # y perspective about x

        # Rotation and Scale
        R=np.eye(3, dtype=np.float32)
        a=random.uniform(-self.degrees, self.degrees)
        s=random.uniform(1.-self.scale, 1.+self.scale)
        R[:2]=cv2.getRotationMatrix2D(angle=a, center=(0,0), scale=s) # 2x3 matrix

        # Shear
        S=np.eye(3, dtype=np.float32)
        S[0,1]=math.tan(random.uniform(-self.shear, self.shear)*math.pi/180.) # x shear in degrees
        S[1,0]=math.tan(random.uniform(-self.shear, self.shear)*math.pi/180.) # y shear in degrees

        # Translation
        T=np.eye(3, dtype=np.float32)
        T[0,2]=random.uniform(0.5-self.translate, 0.5+self.translate)*self.size[0] # x translation in pixels
        T[1,2]=random.uniform(0.5-self.translate, 0.5+self.translate)*self.size[1] # y translation in pixels

        # Combined matrices
        M=T @ S @ R @ P @ C # order of operations (right to left) is important

        # Affine image
        if (border[0]!=0) or (border[1]!=0) or (M!=np.eye(3)).any(): # image chaged
            if self.perspective: img=cv2.warpPerspective(img, M, dsize=self.size, borderValue=(114,)*img.shape[-1])
            else: img=cv2.warpAffine(img, M[:2], dsize=self.size, borderValue=(114,)*img.shape[-1]) # use 2x3 matrix
            if img.ndim==2: img=img[...,None]
        return img, M, s
        
    def apply_bboxes(self, bboxes:np.ndarray, M:np.ndarray)->np.ndarray:
        """
        Apply affine transformation to bounding boxes
        Args:
            bboxes (np.ndarray): Bounding boxes in xyxy format and in pixel units with shape (N,4),
                where N is the number of boxes
            M (np.ndarray): Affine transformation matrix with shape (3,3)
        Returns:
            (np.ndarray): Transformed bounding boxes in xyxy format and in pixel units with shape (N,4)
        """
        n=len(bboxes)
        if n==0: return bboxes
        # form a matrix, for each box, containing 4 corners of the box
        # [[x1,y1,1],
        #  [x2,y2,1],
        #  [x1,y2,1],
        # [x2,y1,1],
        # ...]
        xy=np.ones((n*4, 3), dtype=bboxes.dtype)
        # 4 corners of the box:  x1y1, x2y2, x1y2, x2,y1
        xy[:, :2]=bboxes[:,[0,1,2,3,0,3,2,1]].reshape(n*4, 2) 
        xy=xy@M.T # transform
        xy=(xy[:,:2]/xy[:,2:3] if self.perspective else xy[:,:2]).reshape(n,8) # perspective rescale or affine

        # Create new boxes
        x=xy[:,[0,2,4,6]] # x1, x2, x1, x2
        y=xy[:,[1,3,5,7]] # y1, y2, y2, y1
        return np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1)), dtype=bboxes.dtype).reshape(4,n).T

    def apply_segments(self, segments: np.ndarray, M: np.ndarray)->tuple[np.ndarray, np.ndarray]:
        """
        Apply affine transformations to segments and generate a new bounding boxes

        This function applies affine transformations to input segments and generates new bounding boxes based on
        the transformed segments. It clips the transformed segments to fit within the new bounding boxes
        Args:
            segments (np.ndarray): Input segments with shape (N, M, 2), where N is the number of segments, M is the 
                number of points in each segment, and 2 for x, y
            M (np.ndarray): Affine transformation matrix with shape (3,3)
        Returns:
            bboxes (np.ndarray): New bounding boxes with shape (N, 4) in xyxy format in pixel units
            segments (np.ndarray): Transformed and clipped segments with shape (N,M,2) in pixel units
            indices (list[int]): List of remaining valid segments
        """
        #print(f'In RandomPerspective.apply_segments segments {segments.shape}')
        n, num=segments.shape[:2]
        if n==0: return np.zeros((0,4), dtype=np.float32), segments

        xy=np.ones((n*num, 3), dtype=segments.dtype)
        segments=segments.reshape(-1,segments.shape[-1]) # NxMx2 -> (NM)x2
        #print(f'xy {xy.shape} segments {segments.shape}')
        xy[:,:2]=segments
        xy=xy@M.T # transform
        xy=xy[:,:2]/xy[:,2:3]
        segments=xy.reshape(n, -1, 2)
        #print(f'reshape segments {segments.shape}')
        # indices store indices of valid segments to be used to filter out old boxes 
        # bboxes store boxes in x1,y1,x2,y2
        indices, bboxes, filtered_segments=[], [],[] 
        for i, xy in enumerate(segments):
            box=segment2box(xy, width=self.size[0], height=self.size[1])
            if len(box)==0: continue
            bboxes.append(box)
            #print('\t box', box.shape, ' xy ', xy.shape,  ' xy[:,0] ', xy[:,0].shape, ' box ', box)
            xy[:,0]=xy[:,0].clip(box[0], box[2])
            xy[:,1]=xy[:,1].clip(box[1], box[3])
            filtered_segments.append(xy)
            indices.append(i)
            
        bboxes=np.stack(bboxes, 0) # Nx4
        segments=np.stack(filtered_segments, 0)
        #print('bboxes ', bboxes.shape, ' segments ', segments.shape)
        # bounding boxes have been clipped to be inside images but segments have not, so we use bounding boxes to
        # clip segments
        #print(f'In RandomPerspective.apply_segments segments[...,0] {segments[...,0].shape} bboxes[:,0:1] {bboxes[:,0:1].shape} bboxes[:,2:3] {bboxes[:,2:3].shape}')
        # segments[...,0]=segments[...,0].clip(bboxes[:,0:1], bboxes[:,2:3])
        # segments[...,1]=segments[...,1].clip(bboxes[:,1:2], bboxes[:,3:4])
        return bboxes, segments, indices

    def apply_keypoints(self, keypoints:np.ndarray, M:np.ndarray)->np.ndarray:
        """
        Apply affine transformation to keypoints

        This method transforms the input keypoints using the provided affine transformation matrix. It handles
        perspective rescaling if necessary and updates the visibility of keypoints that fall outside the image
        boundaries after transformation

        Args:
            keypoints (np.ndarray): Array of keypoints with shape (N, 17, 3), where N is the number of instances,
                17 is the number of keypoints per instance, and 3 represents (x, y, visibility)
            M (np.ndarray): 3x3 affine transformation matrix
        Returns:
            (np.ndarray): Transformed keypoint array with the same shape as input (N, 17, 3)
        """
        n, nkpt=keypoints.shape[:2]
        if n==0: return keypoints

        xy=np.ones((n*nkpt, 3), dtype=keypoints.dtype)
        visible=keypoints[...,2].reshape(n*nkpt, 1)
        xy[:,:2]=keypoints[...,:2].reshape(n*nkpt, 2)
        xy=xy@M.T # transform
        xy=xy[:,:2]/xy[:,2:3] # perspective rescale or affine
        outside_mask=(xy[:,0]<0) | (xy[:,1]<0) | (xy[:,0]>self.size[0]) | (xy[:,1]>self.size[1])
        visible[outside_mask]=0
        return np.concatenate([xy, visible], axis=-1).reshape(n, nkpt, 3)

    @staticmethod
    def box_candidates(box1:np.ndarray, box2:np.ndarray, wh_thr:int=2, ar_thr:int=100, area_thr:float=0.1, eps:float=1.e-16)->np.ndarray:
        """
        Compute candidate boxes for further processing based on size and aspect ratio criteria
    
        This method compares boxes before and after augmentation to determine if they meet specified 
        thresholds for width, height, aspect ratio, and area. It's used to filter out boxes that have 
        been overly distorted or reduced by the augmentation process.
        Args:
            box1 (np.ndarray): Original boxes before augmentation, shape (4,N) where N is the number of boxes.
                Format is [x1,y1,x2,y2] in pixel units
            box2 (np.ndarray): Augmented boxes after transformation, shape (4,N). Format is [x1,y1,x2,y2] in pixel units
            wh_thr (int): Width and height threshold in pixels. Boxes smaller than this in either direction are rejected
            ar_thr (int): Aspect ratio threshold. Boxes with an aspect ratio greater than this value are rejected.
            area_thr (float): Area ratio threshold. Boxes with an area ratio (new/old) less than this value are rejected.
            eps (float): Small epsilon value to prevent division by zero
        Returns:
            (np.ndarray): Boolean array of size N indicating which boxes are candidates. True values correspond to boxes that 
                meet all criteria.
        """
        w1,h1=box1[2]-box1[0], box1[3]-box1[1] # N
        w2,h2=box2[2]-box2[0], box2[3]-box2[1] # N
        ar=np.minimum(w2/(h2+eps), h2/(w2+eps)) # N aspect ratio
        return (w2>wh_thr)&(h2>wh_thr)&( w2*h2/(w1*h1+eps) > area_thr ) & (ar < ar_thr) # N
    
    def __call__(self, labels: dict[str, Any])->dict[str, Any]:
        """
        Apply random perspective and affine transformations to an image and its associated labels.
    
        This method performs a series of transformations including rotation, translation, scaling, shearing, and
        perspective distortion on the input images and adjusts the corresponding bounding boxes, segments, 
        and keypoints accordingly
        Args:
            labels (dict[str, Any]): A dict containing image data and annotations. Must include
                `img` (np.ndarray): The input image
                `cls` (np.ndarray): Class labels
                `instamces` (Instances): Object onstances with bounding boxes, segments, and keypoints
                `mosaic_border` (tuple[int, int], optional): Border size for mosaic augmentation
        Returns:
            (dict[str, Any]): Transformed label dict containing:
                `img` (np.ndarray): The transformed image
                `cls` (np.ndarray): Updated class labels
                `instances` (Instances): Updated object instances
                `resized_shape` (tuple[int, int]): New image shape after transformation
        """
    
        if self.pre_transform and "mosaic_border" not in labels: labels=self.pre_transform(labels)
        labels.pop("ratio_pad", None) # do not need ratio pad
        
        img=labels['img']
        cls=labels['cls']
        instances=labels.pop("instances")
        # Make sure the coordinate format is xyxy
        instances.convert_bbox(format='xyxy')
        instances.denormalize(*(img.shape[:2][::-1])) # passing in width and height
        
        border=labels.pop("mosaic_border", self.border)
        self.size=img.shape[1]+border[1]*2, img.shape[0]+border[0]*2 # width and height
        # M is an affine matrix
        # Scale for func: `box_candidates`
        img, M, scale=self.affine_transform(img, border)
        
        bboxes=self.apply_bboxes(instances.bboxes, M)
        
        segments=instances.segments
        keypoints=instances.keypoints
        # Update bboxes if there are segments
        indices=None # indices to valid boxes after transformation
        if len(segments): bboxes, segments, indices=self.apply_segments(segments, M)
        if keypoints is not None: keypoints=self.apply_keypoints(keypoints, M)
        new_instances=Instances(bboxes, segments, keypoints, bbox_format='xyxy', normalized=False)
        # Clip
        new_instances.clip(*self.size)
        
        # Filter instances
        instances.scale(scale_w=scale, scale_h=scale, bbox_only=True)
        box1=instances.bboxes
        if indices is not None:
            box1=box1[indices]
            cls=cls[indices]
        # Make the bboxes have the same scale with new_bboxes
        i=self.box_candidates(box1=box1.T, box2=new_instances.bboxes.T, area_thr=0.01 if len(segments) else 0.1)
        labels['instances']=new_instances[i]
        labels['cls']=cls[i]
        labels['img']=img
        labels['resized_shape']=img.shape[:2]
    
        return labels

class RandomHSV:
    """
    Randomly adjust the Hue, Saturation, and Value (HSV) channels of an image
    This class applies random HSV augmentation to images within predefined limits set by 
    hgain, sgain, and vgain.

    Unlike mixing augmentation and letterbox, this does not modify annotation instances (including bounding boxes, class indices,
    segments and keypoints) so however annotations were defined as input are passing through without any changes
    """
    def __init__(self, hgain:float=0.5, sgain:float=0.5, vgain:float=0.5)->None:
        """
        Initialize the RandomHSV object for random HSV (Hue, Saturation, Value) augmentation.
        Args:
            hgain (float): Maximum variation for hue. Should be in the range [0, 1]
            sgain (float): Maximum variation for saturation. Should be in the range [0,1].
            vgain (float): Maximum variation for value. Should be in the range [0,1].
        """
        self.hgain=hgain
        self.sgain=sgain
        self.vgain=vgain
        
    def __call__(self, labels: dict[str, Any])->dict[str, Any]:
        """
        Apply random HSV augmentation to an image within predefined limits.
        
        This method modifies the input image by randomly adjusting its Hue, Saturation, and Value
        (HSV) channels. The adjustments are made within the limits set by hgain, sgain, and vgain 
        during initialization.
        
        Args:
            labels (dict[str, Any]): A dict containing image data and metadata. Must include an `img`
                key with the image as a numpy array
        Returns:
            (dict[str, Any]): A dict containing the modified image
        """

        img=labels['img']
        
        if img.shape[-1]!=3: return labels # only applicable to 3 channel images
        
        if not (self.hgain or self.sgain or self.vgain): return labels
        
        dtype=img.dtype # uint8
        r=np.random.uniform(low=-1., high=1.0, size=3)*[self.hgain, self.sgain, self.vgain] # random gains
        x=np.arange(0, 256, dtype=r.dtype)
        lut_hue=((x+r[0]*180)%180).astype(dtype)
        # scale down and up by r+1, e.g.  r=[0.5,0.5] -> scale down and up by 50% and 150%
        # r+1 also make sure that r near zero does not destroy saturation and value
        # and that r+1 is always positive
        lut_sat=np.clip(x*(r[1]+1), 0, 255).astype(dtype) 
        lut_val=np.clip(x*(r[2]+1), 0, 255).astype(dtype)
        lut_sat[0]=0 # prevent pure white changing color
        
        
        hue, sat, val=cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        im_hsv=cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=img) # no return needed
        return labels

class RandomFlip:
    """
    Apply a random horizontal or vertical flip to an image with a given probability

    This class performs random image flipping and updates corresponding instance annotations such as 
    bounding boxes and keypoints. It internally converts bounding box format to xywh without modifying the 
    normalization status of bounding boxes.
    """
    def __init__(self, p:float=0.5, direction:str='horizontal', flip_idx:list[int]=None)->None:
        """
        Initialize the RandomFlip class with probability and direction
        Args:
            p (float): The probility of applying the flip. Must be between 0 and 1
            direction (str): The direction to apply the flip. Must be `horizontal` or `vertical`
            flip_idx (list[int] | None): Index mapping for flipping keypoints, if any.
        """
        assert direction in {"horizontal", "vertical"}, f"Support direction `horizontal` or `vertical`, got {direction}"
        assert 0<=p<=1., f'The probability must be in the range [0, 1], got {p}'

        self.p=p
        self.direction=direction
        self.flip_idx=flip_idx

    def __call__(self,labels:dict[str, Any])->dict[str, Any]:
        """
        Apply random flip to an image and update any instances like bounding boxes and keypoints accordingly

        Args:
            labels (dict[str, Any]): A dict containing the following keys:
                `img` (np.ndarray): The image to be flipped
                `instances` (Instances): An object containing bounding boxes and optionally keypoints
        Returns:
            (dict[str, Any]): The same dict with the flipped image and updated instances:
                `img` (np.ndarray): The flipped image
                `instances` (Instances): Updated instances matching the flipped image
        """
        img=labels['img']
        instances=labels.pop('instances')
        instances.convert_bbox(format='xywh')
        h, w=img.shape[:2]
        if instances.normalized: h=w=1
            
        if self.direction=='horizontal' and random.random()<self.p:
            img=np.fliplr(img)
            instances.fliplr(w)
        if self.direction=='vertical' and random.random()<self.p:
            img=np.flipud(img)
            instances.flipud(h)
        labels['img']=np.ascontiguousarray(img)
        labels['instances']=instances
        return labels

class Compose:
    """
    A class for composing multiple image transformations.
    """
    def __init__(self, transforms):
        """
        Initialize the Compose object with a list of transforms
        Args:
            transforms (list[Callable]): A list of callable transform objects to be applied sequentially
        """
        self.transforms=transforms if isinstance(transforms, list) else [transforms]
        
    def __call__(self, data):
        """
        Apply a series of transformations to input data.

        This method sequentially applies each transformation in the Compose object's transforms to the input 
        data
        
        Args:
            data (Any): The input data to be transformed. This can be of any type, depending on the 
                transformations in the list.
        Returns:
            (Any): The transformed data after applying all transformations in sequence
        """
        for t in self.transforms: data=t(data)
        return data
        
    def append(self, transform):
        """
        Append a new transform to the existing list
        Args:
            transform (Callable): The transformation to be added to the composition
        Examples:
            >>> compose=Compose([RandomFlip(), RandomPerspective()])
            >>> compose.append(RandomHSV)
        """
        self.transforms.append(transform)

    def insert(self, index, transform):
        """
        Insert a new transform at a specified index in the existing list of transform
        Args:
            index (int): The index at which to insert the new transform
            transform (callable): The transform object to be inserted
        Examples:
            >>> compose = Compose([Transform1(), Transform2()])
            >>> compose.insert(1, Transform3())
            >>> len(compose.transforms)
            3
        """
        self.transforms.insert(index, transform)
        
    def __getitem__(self, index: list[int] | tuple[int] | int)-> Compose:
        """
        Retrieve a specific transform or a set of transforms using indexing
        Args:
            index (int | list[int] | tuple[int]): Index or list of indices of the transforms to retrieve
        Returns:
            (Compose): A new Compose object containing the selected transform(s).
        Examples:
            >>> transforms = [RandomFlip(), RandomPerspective(10), RandomHSV(.5, .5, .5)]
            >>> compose = Compose(transforms)
            >>> single_transform=compose[1] # Return a Compose object with only RandomPerspective
            >>> multiple_transforms=compose[0:2] # Return a Compose object with RandomFlip and RandomPerspective
        """
        assert isinstance(index, (int, list, tuple)), f'The indices should be either list, int, or tuple but got {type(index)}'
        return self.transforms[index] if isinstance(index, int) else Compose([self.transforms[i] for i in index])

    def __setitem__(self, index:list[int]|tuple[int]|int, value: list[Any]|tuple[Any]|Any)->None:
        """
        Set one or more transforms in the composition using indexing.

        Args:
            index (int| list[int] | tuple[int]): Index or list/tuple of indices to set transforms at
            value (Any | list[Any] | tuple[Any]): Transform or list/tuple of transforms to set at the specified index (ices)
        Examples:
            >>> compose=Compose([Transform1(), Transform2(), Transform3()])
            >>> compose[1]=NewTransform() # Replace the second transform
            >>> compose[0:2]=[NewTransform1(), NewTransform2()] # Replace first two transforms
        """
        assert isinstance(index, (int, list, tuple)), f'The indices should be either list/tuple or int type but got {type(index)}'
        if isinstance(index, (list, tuple)):
            assert isinstance(value, (list, tuple)), f"Values should be a sequence but got {type(value)}"
            assert len(index)==len(value), f'Values should be the same length as indices, but got {len(value)} values and {len(index)} indices'
        if isinstance(index, int): index, value=[index], [value]
        for i, v in zip(index, value):
            assert i < len(self.transforms), f'list index {i} out of range {len(self.transforms)}'
            self.transforms[i]=v
            
    def tolist(self):
        """
        Convert the transforms to a standard Python list
        Returns:
            (list[callable]): A list containing all transform objects
        """
        return self.transforms
    def __repr__(self):
        """
        Return a string representation
        Returns:
            (str): A string representation, including the list of transforms
        """
        return f"{self.__class__.__name__}({', '.join(f'{t}' for t in self.transforms)})"


class Format:
    """
    A class for formatting image annotations for object detection, instance segmentation, and pose estimation tasks.

    This class standardizes image and instance annotations to be used by the `collate_fn` in PyTorch DataLoader
    """
    def __init__(self, bbox_format:str='xywh', normalize:bool=True, return_mask:bool=False, return_keypoint:bool=False,
                return_obb:bool=False, mask_ratio:int=4, mask_overlap:bool=True, batch_idx:bool=True, bgr:float=0.):
        """
        Initialize the Format class with given parameters for image and instance annotation formatting.
        Args:
            bbox_format (str): Format for bounding boxes. Options are `xywh`, `xyxy`, etc.
            normalize (bool): Whether to normalize bounding boxes to [0,1].
            return_mask (bool): If True, return instance masks for segmentation tasks
            return_keypoint (bool): If True, return keypoints for pose estimation tasks.
            return_obb (bool): If True, return oriented bounding boxes
            mask_ratio (int): Downsample ratio for masks.
            mask_overlap (bool): If True, allows mask overlap
            batch_idx (bool): If True, keep batch indices
            bgr (float): Probability of returning BGR images instead of RGB
        """
        self.bbox_format=bbox_format
        self.normalize=normalize
        self.return_mask=return_mask # set to False when training detection only
        self.return_keypoint=return_keypoint
        self.return_obb=return_obb
        self.mask_ratio=mask_ratio
        self.mask_overlap=mask_overlap
        self.batch_idx=batch_idx
        self.bgr=bgr

    def _format_segments(self, instances:Instances, cls:np.ndarray, w:int, h:int)->tuple[np.ndarray, Instances, np.ndarray]:
        """
        Convert polygon segments to masks
        Args:
            instances (Instances): Object containing segment information
            cls (np.ndarray): Class labels for each instances of size Nx1 for N instances
            w (int): Width of the image
            h (int): Height of the image
        Returns:
            masks (np.ndarray): Masks with shape (N,H,W) or (1,H,W) if mask_overlap is True
            instances (Instances): Updated instances object with sorted segments if mask_overlap is True
            cls (np.ndarray): Updated class labels, sorted if mask_overlap is True
        Notes:
            - If mask_overlap is True, masks are overlapped and sorted by area from large to small
            - If mask_overlap is False, each mask is represented seperately as a binary mask
            - Masks are downsampled according to mask_ratio
        """
        segments=instances.segments
        if self.mask_overlap:
            masks, sorted_idx=polygon2masks_overlap((h,w), segments, downsample_ratio=self.mask_ratio)
            masks=masks[None] # HxW -> 1xHxW
            instances=instances[sorted_idx] # from large to small
            cls=cls[sorted_idx]
        else:
            masks=polygon2masks((h, w), segments, color=1, downsample_ratio=self.mask_ratio)
        return masks, instances, cls

    def _format_img(self, img:np.ndarray)->torch.Tensor:
        """
        Format an image from a numpy array to a PyTorch tensor
        This function performs the following operations:
        1. Ensure that the image has 3 dimensions (add a channel dimension if needed)
        2. Transpose the image from HWC to CHW
        3. Optionally flip the color channels from BGR to RGB
        4. Convert the image to a contiguous array
        5. Convert the Numpy array to a PyTorch tensor
    
        Args:
            img (np.ndarray): Input image as a Numpy array with shape (H,W,C) or (H,W)
        Returns:
            (torch.Tensor): Formatted image as a PyTorch tensor with shape (C,H,W)
        """
        if len(img.shape)<3: img=np.expand_dims(img, -1)
        img=img.transpose(2,0,1) # HxWxC -> CxHxW
        img=np.ascontiguousarray(img[::-1] if random.uniform(0,1)>self.bgr and img.shape[0]==3 else img)
        img=torch.from_numpy(img)
        return img
    
    def __call__(self, labels: dict[str, Any])->dict[str, Any]:
        """
        Format image annotations for object detection, instance segmentation, and pose estimation tasks

        This method standardizes the image and instance annotations to be used by the `collate_fn` in PyTorch DataLoader
        It processes the input labels dict, converting annotations to the specified format and applying normalization if required
        Args:
            labels (dict[str, Any]): A  dict containing image and annotation data with the following keys:
                - `img` (np.ndarray): The input image as a numpy array of size HxWxC 
                - `cls` (np.ndarray): Class labels for instances of size Nx1 where N is the number of objects
                - `instances` (Instances): An Instances object containing bounding boxes, segments, and keypoints
        Returns: 
            (dict[str, Any]): A dict with formatted data, including:
                - `img`: Formatted image tensor
                - `cls` : Class label's tensor
                - `bboxes`: Bounding box tensor in the specified format
                - `masks`: Instance masks tensor (if return_mask is True)
                - `keypoints`: Keypoint tensor (if return_keypoint is True)
                - `batch_idx`: Batch index tensor (if batch_idx is True)
        """
        img=labels.pop('img')
        h, w = img.shape[:2]
        cls=labels.pop('cls')
        instances=labels.pop('instances')
        instances.convert_bbox(format=self.bbox_format)
        instances.denormalize(w, h)
        nl=len(instances) # number of boxes/segmentations

        if self.return_mask:
            if nl:
                masks,instances,cls=self._format_segments(instances, cls, w, h)
                masks=torch.from_numpy(masks)
            else: masks=torch.zeros(1 if self.mask_overlap else nl, *[s//self.mask_ratio for s in img.shape[:2]], dtype=torch.uint8)
            labels['masks']=masks # 1xHxW if overlap else NxHxW
        labels['img']=self._format_img(img) # CxHxW
        labels['cls']=torch.from_numpy(cls) if nl else torch.zeros(nl, 1) # Nx1
        labels['bboxes']=torch.from_numpy(instances.bboxes) if nl else torch.zeros((nl, 4)) # Nx4
        if self.return_keypoint:
            labels['keypoints']=torch.empty(0, 3) if instances.keypoints is None else torch.from_numpy(instances.keypoints)
            if self.normalize:
                labels['keypoints'][...,0]/=w
                labels['keypoints'][...,1]/=h
        if self.return_obb:
            #labels['bboxes']=xyxyxyxy2xywhr(torch.from_numpy(instances.segments)) if len(instances.segments) else torch.zeros((0, 5))
            raise NotImplementedError(f'Please see https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py')
        # Note: normalize (oriented) bounding boxes in xywh(r) format for width-height consistency
        if self.normalize:
            labels['bboxes'][:,[0,2]]/=w
            labels['bboxes'][:,[1,3]]/=h
        if self.batch_idx: labels['batch_idx']=torch.zeros(nl)
        
        return labels

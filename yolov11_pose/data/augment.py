from __future__ import annotations

import math
import random
from copy import deepcopy
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from computer_vision.yolov11_pose.utils.instance import Instances
from computer_vision.yolov11_pose.utils.metrics import bbox_ioa
from computer_vision.yolov11_pose.utils.ops import segment2box
from computer_vision.yolov11_pose.utils import IterableSimpleNamespace

class BaseMixTransform:
    """Base class for mix transformations like CutMix, MixUp, and Mosaic

    This class provides a foundation for implementing transformations on datasets. It handles the probability-based application
    of transformations and manages the mixing of multiple images and labels

    Examples:
        >>> class CustomMixTransform(BaseMixTransform):
        ...      def _mix_transform(self, labels):
        ...          # Implement custom mix logic here
        ...          return labels
        ...
        ...      def get_indexes(self):
        ...          return [random.randint(0, len(self.dataset)-1) for _ in range(3)]
        >>> dataset=YOLODataset()
        >>> transform=CustomMixTransform(dataset, p=0.5)
        >>> mixed_labels=transform(original_labels)
    """
    def __init__(self, dataset, pre_transform=None, p=0.0)->None:
        """Initialize the BaseMixTransform object for mix transformations like CutMix, MixUp and Mosaic.

        This class serves as a base for implementing mix transform in image processing pipelines
        Args:
            dataset (Any): The dataset object containing images and labels for mixing
            pre_transform (Callable | None): Optional transform to apply before mixing
            p (float): Probability of applying the mix transformation. Should be in the range [0,1]
        """
        self.dataset=dataset
        self.pre_transform=pre_transform
        self.p=p
        
    def __call__(self, labels:dict[str, Any])->dict[str, Any]:
        """Apply pre-processing transformation and cutmix/mixup/mosaic transforms to labels data

        This method determines whether to apply the mix transform based on a probability factor. If applied, it selects additional images,
        applies pre-transforms if specified, and then performs the mix transform

        Args:
            labels (dict[str, Any]): A dict containing label data for an image
        Returns:
            (dict[str, Any]): The transformed labels dict, which may include mixed data from other images
        Examples:
            >>> transform=BaseMixTransform(dataset, pre_transform=None, p=0.5)
            >>> result=transform({'image':img, 'bboxes':bboxes, 'cls', classes})
        """
        if random.uniform(0,1)>self.p: return labels

        # Get index of other images
        indexes=self.get_indexes()
        if isinstance(indexes, int): indexes=[indexes]

        # Get image and label information
        mix_labels=[self.dataset.get_image_and_label(i) for i in indexes]

        if self.pre_transform is not None:
            for i, data in enumerate(mix_labels): mix_labels[i]=self.pre_transform(data)
        labels['mix_labels']=mix_labels

        # Update cls and texts
        labels=self._update_label_text(labels)
        # Mosaic, CutMix, or Mixup
        labels=self._mix_transform(labels)
        labels.pop('mix_labels', None)
        return labels

    def _mix_transform(self, labels:dict[str, Any]):
        """Apply CutMix, Mixup, or Mosaic augmentation to the label dict

        This method should be implemented by subclass to perform specific mix transformation like CutMix, Mixup, or Mosaic.
        It modifies the input label dict in-place with the augmented data

        Args:
            labels (dict[str, Any]): A dict containing image and label data. Expect to have a 'mix_labels' key with a list 
                of additional image and label data for mixing
        Returns:
            (dict[str, Any]): The modified labels dict with augmented data after applying the mix transform
        Examples:
            >>> transform=BaseMixTransform(dataset)
            >>> labels = {"image": img, "bboxes": boxes, "mix_labels": [{"image": img2, "bboxes": boxes2}]}
            >>> augmented_labels = transform._mix_transform(labels)
        """
        raise NotImplementError
        
    def get_indexes(self):
        """Get a list of shuffled indexes for mosaic augmentation
        Returns:
            (list[int]): A list of shuffled indexes from the dataset
        Examples:
            >>> transform=BaseMixTransform(dataset)
            >>> indexes=transform.get_indexes()
            >>> print(indexes)  # [3, 18,7,2]
        """
        return random.randint(0, len(self.dataset)-1)

    @staticmethod
    def _update_label_text(labels:dict[str, Any])->dict[str, Any]:
        """Update label text and class IDs for mixed labels in image augmentation

        This method processes the 'texts' and 'cls' fields of the input labels dict and any mixed labels, creating a unified set of text 
        labels and updating class IDs accordingly

        Args:
            labels (dict[str, Any]): A dict containing label information, including 'texts' and 'cls' fields, and optionally a 'mix_labels'
                field with additional label dicts
        Returns:
            (dict[str, Any]): The updated labels dict with unified text labels and updated class IDs

        Examples:
            >>> labels={
            ...   'texts': [['cat'],['dog']],
            ...   'cls': torch.tensor([[0],[1]]),
            ...   'mix_labels': [{'texts':[['bird'],['fish']] 'cls':torch.tensor([[2],[3]])}],
            ... }
            >>> updated_labels=self._updare_label_text(labels)
            >>> print(updated_labels['texts'])
            [['cat'],['dog'],['bird'],['fish']]
            >>> print(updated_labels['mix_labels'][0]['cls'])
            tensor([[2],
                    [3]])
        """
        if 'texts' not in labels: return labels
        mix_texts=[*labels['texts'], *(item for x in labels['mix_labels'] for item in x['texts'])]
        mix_texts=list({tuple(x) for x in mix_texts})
        text2id={text:i for i, text in enumerate(mix_texts)}

        for label in [labels]+labels['mix_labels']:
            for i, cls in enumerate(labels['cls'].squeeze(-1).tolist()):
                text=labels['texts'][int(cls)]
                labels['cls'][i]=text2id[tuple(text)]
            labels['texts']=mix_texts
        return labels

class Mosaic(BaseMixTransform):
    """Mosaic augmentation for image dataset

    This class perform mosaic augmentation by combining multiple (3, 4 or 9) images into a single mosaic image. The augmentation
    is applied to a dataset with a given probability

    Examples:
        >>> dataset=Dataset(...)
        >>> mosaic_aug=Mosaic(dataset, imgsz=640, p=0.5, n=4)
        >>> augmented_labels=mosaic_aug(original_labels)
    """
    def __init__(self, dataset, imgsz:int=640, p:float=1., n:int=4):
        """Initialize the Mosaic augmentation object
        
        This class performs mosaic augmentation by combining multiple (3, 4 or 9) images into a single mosaic image. The 
        augmentation is applied to a dataset with a given probability

        Args:
            dataset (Any): The dataset on which the mosaic augmentation is applied
            imgsz (int): Image size (height and width) after mosaic pipeline of a single image
            p (float): Probability of applying the mosaic augmentation. Must be in the range [0,1]
            n (int): The grid size, either 4 (for 2x2) or 9 (for 3x3)
        """
        assert 0.<=p<=1., f'The probability should be in the range [0, 1], but got {p}'
        assert n in {3,4,9}, 'grid must be equal to 4 or 9'
        super().__init__(dataset=dataset, p=p)
        self.imgsz=imgsz
        self.border=(-imgsz//2, -imgsz//2) # (height, width) but since they are equal, it does not matter
        self.n=n
        self.buffer_enabled=self.dataset.cache != 'ram'
        
    def get_indexes(self):
        """Return a list of random indexes from the dataset for mosaic augmentation
        
        This method selects random image indices either from a buffer or from the entire dataset, depending on the 'buffer' 
        parameter. It is used to choose images for creating mosaic augmentation

        Returns:
            (list[int]): A list of random image indices. The length of the list is n-1, where n is the number of images used in 
                mosaic (either 2, 3 or 8, depending on whether n is 3, 4 or 9)
        """
        #print(f'In data.augment.Mosaic.get_indexes self.buffer_enabled {self.buffer_enabled} len(self.dataset.buffer) {len(self.dataset.buffer)}')
        if self.buffer_enabled and len(self.dataset.buffer)>self.n: # select images from buffer
            return random.choices(list(self.dataset.buffer), k=self.n-1)
        else: # select any images 
            return [random.randint(0, len(self.dataset)-1) for _ in range(self.n-1)]

    @staticmethod
    def _update_labels(labels:dict[str, Any], padw:int, padh:int)->dict[str, Any]:
        """Update label coordinates with padding values

        This method adjusts the bounding box coordinates of object instances in the labels by adding padding values. It also
        denormalizes the coordinates of they were previously normalized.

        Args:
            labels (dict[str, Any]): A dict containing image and instance information
            padw (int): Padding width in pixels to be added to the x-coordinates
            padh (int): Padding height in pixels to be added to the y-coordinates
        Returns:
            (dict[str, Any]): Updated labels dict with adjusted instance coordinates
        Examples:
            >>> labels {'img':np.zeros((100,100,3)), 'instances':Instances(...)}
            >>> padw, padh=50,50 # in pixels
            >>> updated_labels=Mosaic._update_labels(labels, padw, padh) # output boxes are denormalized
        """
        h, w=labels['img'].shape[:2]
        labels['instances'].convert_bbox(format='xyxy')
        labels['instances'].denormalize(w, h)
        labels['instances'].add_padding(padw, padh)
        return labels

    def _cat_labels(self, mosaic_labels:list[dict[str, Any]])->dict[str, Any]:
        """Concatenate and process labels for mosaic augmentation

        This method combines labels from multiple images used in mosaic augmentation, clip instances to the mosaic border, and remove
        zero-area boxes

        Args:
            mosaic_labels (list[dict[str, Any]]): A list of label dicts for each image in the mosaic
        Returns:
            (dict[str, Any]): A dict containing concatenated and processed labels for the mosaic image, including:
                - im_file (str): File path of the first image in the mosaic
                - ori_shape (tuple[int, int]): Original shape of the first image
                - resized_shape (tuple[int, int]): Shape of the mosaic image (imgsz*2, imgsz*2)
                - cls (np.ndarray): Concatenated class labels
                - instances (Instances): Concatenated instance annotation
                - mosaic_border (tuple[int, int]): Mosaic border size
                - texts (list[str], optional): Text labels if present in the original labels
        """
        if not mosaic_labels: return {}
        cls=[]
        instances=[]
        imgsz=self.imgsz*2 # mosaic image size
        for labels in mosaic_labels:
            cls.append(labels['cls'])
            instances.append(labels['instances'])
        # Final labels
        final_labels={
            'im_file': mosaic_labels[0]['im_file'],
            'ori_shape':mosaic_labels[0]['ori_shape'],
            'resized_shape': (imgsz, imgsz),
            'cls': np.concatenate(cls, 0),
            'instances': Instances.concatenate(instances, axis=0),
            'mosaic_border': self.border,
        }
        final_labels['instances'].clip(imgsz, imgsz)
        good=final_labels['instances'].remove_zero_area_boxes()
        final_labels['cls']=final_labels['cls'][good]
        if 'texts' in mosaic_labels[0]: final_labels['texts']=mosaic_labels[0]['texts']
        return final_labels

    def _mosaic4(self, labels:dict[str, Any])->dict[str, Any]:
        """Create a 2x2 image mosaic from four input images
        
        This method combines four images into a single mosaic image by placing them in a 2x2 grid. It also updates the 
        corresponding labels for each image in the mosaic
        
        Args:
            labels (dict[str, Any]): A dict containing image data and labels for the base image (index 0) and three additional
                images (indices 1-3) in the 'mix_labels' key.
        Returns:
            (dict[str, Any]): A dict containing the mosaic image and updated labels. The 'img' key contains the mosaic image as
                a numpy array, and other keys contain the combined and adjusted labels for all four images
        Examples:
            >>> mosaic=Mosaic(dataset, imgsz=640, p=1., n=4)
            >>> labels={"img":np.random.rand(480, 640,3),
            ...         "mix_labels": [{"img":np.random.rand(480,640,3)} for _ in range(3)],
            ... }
            >>> result=mosaic._mosaic4(labels)
            >>> assert result["img"].shape==(1280,1280,3)
        """
        mosaic_labels=[]
        s=self.imgsz
        yc,xc=(int(random.uniform(-x, 2*s+x)) for x in self.border)
        
        for i in range(4):
            labels_patch=labels if i==0 else labels['mix_labels'][i-1]
            # load image
            img=labels_patch['img']
            h, w=labels_patch.pop('resized_shape') # same shape as img, i.e., =img.shape[:2]
            
            # Place img in img4
            if i==0: # top left
                img4=np.full((s*2, s*2, img.shape[2]),114,dtype=np.uint8) # base image with 4 tiles
                x1a,y1a,x2a,y2a=max(xc-w, 0),max(yc-h, 0), xc, yc # xmin, ymin, xmax, ymax (large image) 
                x1b,y1b,x2b,y2b=w-(x2a-x1a), h-(y2a-y1a), w, h # xmin, ymin, xmax, ymax (small image)
            elif i==1: # top right
                x1a,y1a,x2a,y2a=xc, max(yc-h,0), min(xc+w, s*2), yc
                x1b,y1b,x2b,y2b=0, h-(y2a-y1a), min(w, x2a-x1a), h
            elif i==2: # bottom left
                x1a,y1a,x2a,y2a=max(xc-w,0), yc, xc, min(s*2, yc+h)
                x1b,y1b,x2b,y2b=w-(x2a-x1a), 0, w, min(y2a-y1a, h)
            else: # i==3 bottom right
                x1a,y1a,x2a,y2a=xc, yc, min(xc+w, s*2), min(s*2, yc+h)
                x1b,y1b,x2b,y2b=0,0,min(w, x2a-x1a), min(y2a-y1a, h)
            
            img4[y1a:y2a, x1a:x2a]=img[y1b:y2b,x1b:x2b] # img[ymin:ymax, xmin:xmax]
            padw=x1a-x1b
            padh=y1a-y1b
        
            labels_patch=self._update_labels(labels_patch, padw, padh)
            mosaic_labels.append(labels_patch)
        
        final_labels=self._cat_labels(mosaic_labels)
        final_labels['img']=img4
        return final_labels

    def _mosaic3(self, labels:dict[str, Any])->dict[str, Any]:
        """Create a 1x3 image mosaic by combining three images
        
        This method arranges three images in a horizontal layout, with the main image in the center and two additional images on either side.
        It is part of the Mosaic augmentation technique used in object detection
        
        Args:
            labels (dict[str, Any]): A dict containing image and label information for the main (center) image and additional images, including
                'img' key with the image array, and 'mix_labels' key with a list of two dicts containing information for the side images
        Returns:
            (dict[str, Any]): A dict with the mosaic image and updated labels. Key include:
                - 'img' (np.ndarray): The mosaic image array with shape (H, W, C)
                - Other keys from the input labels, updated to reflect the new image dimensions
        """
        mosaic_labels=[]
        s=self.imgsz
        for i in range(3):
            labels_patch=labels if i==0 else labels['mix_labels'][i-1]
            # Load image
            img=labels_patch['img']
            h, w=labels_patch.pop('resized_shape')
        
            # Place img in img3
            if i==0: # center
                img3=np.full((s*3, s*3, img.shape[2]), 114, dtype=np.uint8) # base image with 3 tiles
                h0, w0=h,w
                c=s,s,s+w, s+h # xmin, ymin, xmax, ymax (base) coordinates
            elif i==1: # right
                c=s+w0, s, s+w0+w, s+h
            elif i==2: # left
                c=s-w, s+h0-h, s, s+h0
        
            padw,padh=c[:2]
            x1, y1, x2, y2=(max(x,0) for x in c) # allocate coordinates
            
            img3[y1:y2, x1:x2]=img[(y1-padh):, (x1-padw):] # img3[ymin:ymax, xmin:xmax]
    
            # img3 is of size sx3, sx3 but later we return img3 (cropped by border on left/right and top/bottom) of size sx2, sx2, so
            # annotations are shifted by padw+border, padh+border 
            labels_patch=self._update_labels(labels_patch, padw+self.border[0], padh+self.border[1])
            mosaic_labels.append(labels_patch)
        
        final_labels=self._cat_labels(mosaic_labels)
        final_labels['img']=img3[-self.border[1]:self.border[1], -self.border[0]:self.border[0]]
        return final_labels
    
    def _mosaic9(self, labels:dict[str, Any])->dict[str, Any]:
        """Create a 3x3 image mosaic from the input image and eight additional images
        
        This method combines nine images into a single mosaic image. The input image is placed at the center, and eight additional images
        from the dataset are placed around it in a 3x3 grid pattern
        
        Args:
            labels (dict[str, Any]): A dict containing the input image and its associated labels. It should have the following keys:
                - 'img' (np.ndarray): The input image
                - 'resized_shape' (tuple[int, int]): The shape of the resized image (height, width)
                - 'mix_labels' (list[dict]): A list of dicts containing information for the additional eight images
        Returns:
            (dict[str, Any]): A dict containing the mosaic image and updated labels. It includes the following keys:
                - 'img' (np.ndarray): The final mosaic image
                - Other keys from the input labels, updated to reflect the new mosaic arrangement
        """
        mosaic_labels=[]
        s=self.imgsz
        hp,wp=-1,-1 # previous height, width 
        for i in range(9):
            labels_patch=labels if i==0 else labels['mix_labels'][i-1]
            # Load image
            img=labels_patch['img']
            h, w=labels_patch.pop('resized_shape')
        
            # Place img in img9
            if i==0: # center
                img9=np.full((s*3,s*3,img.shape[2]),114,dtype=np.uint8) # base image with 9 tiles
                h0, w0=h, w
                c=s, s, s+w, s+h # xmin, ymin, xmax, ymax (base) coordinates
            elif i==1: # top
                c=s, s-h, s+w, s
            elif i==2: # top right
                c=s+wp, s-h, s+wp+w, s
            elif i==3: # right
                c=s+w0, s, s+w0+w, s+h
            elif i==4: # bottom right
                c=s+w0, s+hp, s+w0+w, s+hp+h
            elif i==5: # bottom
                c=s+w0-w, s+h0, s+w0, s+h0+h
            elif i==6: # bottom left
                c=s+w0-wp-w, s+h0, s+w0-wp, s+h0+h
            elif i==7: # left
                c=s-w, s+h0-h, s, s+h0
            elif i==8: # top left
                c=s-w, s+h0-hp-h, s, s+h0-hp
            padw,padh=c[:2]
            x1,y1,x2,y2=(max(x, 0) for x in c) # allocate coordinate
        
            # Image
            img9[y1:y2, x1:x2]=img[(y1-padh):, (x1-padw):] # img9[ymin:ymax, xmin:xmax]
            hp, wp=h,w # previous height and width  for the next iteration
        
            # img3 is of size sx3, sx3 but later we return img3 (cropped by border on left/right and top/bottom) of size sx2, sx2, so
            # annotations are shifted by padw+border, padh+border 
            labels_patch=self._update_labels(labels_patch, padw+self.border[0], padh+self.border[1])
            mosaic_labels.append(labels_patch)
        
        final_labels=self._cat_labels(mosaic_labels)
        final_labels['img']=img9[-self.border[1]:self.border[1], -self.border[0]:self.border[0]]
        return final_labels

    def _mix_transform(self, labels:dict[str, Any])->dict[str, Any]:
        """Apply mosaic augmentation to the input image and labels
        
        This method combines multiple images (3, 4, or 9) into a single mosaic image based on the 'n' attribute. It ensures that 
        rectangular annotations are not present and that there are other images available for mosaic
        Args:
            labels (dict[str, Any]): A dict containing image data and annotations. Expecting keys including
                - 'rect_shape': Should be None as rect and mosaic are mutually exclusive
                - 'mix_labes': A list of dicts containing data for other images to be used in mosaic
        Returns:
            (dict[str, Any]): A dict containing the mosaic-augmented image and updated annotation
        """
        assert labels.get('rect_shape') is None, 'rect and mosaic are mutually exclusive'
        assert len(labels.get('mix_labels',[]))==self.n-1, f'There are not sufficient additional images for mosaic augmentation, requiring {n-1} but got {len(labels.get("mix_labels",[]))}'
        return self._mosaic3(labels) if self.n==3 else self._mosaic4(labels) if self.n==4 else self._mosaic9(labels)

class MixUp(BaseMixTransform):
    """Apply MixUp augmentation to image datasets

    This class implements the MixUp augmentation technique as described in the paper [mixup: Beyond Empirical Risk Minimization]
    (https://arxiv.org/abs/1710.09412). MixUp combines two images and their labels using a random weight.

    Examples:
        >>> dataset=YourDataset(...) # Your image dataset
        >>> mixup = MixUp(dataset, p=0.5)
        >>> augmented_labels=mixup(original_labels)
    """

    def __init__(self, dataset, pre_transform=None, p:float=0.0)->None:
        """Initialize the MixUp augmentation object

        MixUp is an image augmentation technique that combines two images by taking a weighted sum of their pixel values and labels. 

        Args:
            dataset (Any): The dataset to which MixUp augmentation will be applied
            pre_transform (Callable | None): Optional transform to apply to images before MixUp
            p (float): Probability of applying MixUp augmentation to an image. Must be in the range [0,1]
        """
        super().__init__(dataset=dataset, pre_transform=pre_transform, p=p)

    def _mix_transform(self, labels: dict[str, Any])->dict[str, Any]:
        """Apply MixUp augmentation to the input labels

        This method implements the MixUp augmentation technique as described in the paper "mixup: Beyond Empirical Risk
        Minimization" (https://arxiv.org/abs/1710.09412). 

        Args:
            labels (dict[str, Any]): A dict containing the original image and label information
        Returns:
            (dict[str, Any]): A dict containing the mixed-up image and combined label information
        """
        r=np.random.beta(32., 32.) # mixup ratio, alpha=beta=32.
        labels2=labels['mix_labels'][0]
        labels['img']=(labels['img']*r +labels2['img']*(1.-r)).astype(np.uint8)
        labels['instances']=Instances.concatenate([labels['instances'], labels2['instances']], axis=0)
        labels['cls']=np.concatenate([labels['cls'], labels2['cls']], 0)
        return labels

class LetterBox:
    """Resize image and padding for detection, instance segmentation, and pose.

    This class resizes and pads images to a specified shape while preserving aspect ratio. It also updates corresponding labels and bounding boxes.

    Examples:
        >>> transform=LetterBox(new_shape=(640,640))
        >>> result=transform(labels)
        >>> resized_img=result['img']
        >>> updated_instances=result['instances']
    """
    def __init__(self, new_shape:tuple[int, int]=(640, 640), auto:bool=False, scale_fill:bool=False, scaleup:bool=True, center:bool=True,
        stride:int=32, padding_value:int=114, interpolation:int=cv2.INTER_LINEAR):
        """Initialize LetterBox object for resizing and padding images

        This class is designed to resize and pad images for object detection, instance segmentation, and pose estimation tasks. It supports
        various resizing modes including auto-sizing, scale-fill, and letterboxing.

        Args:
            new_shape (tuple[int,int]): Target size (height, width) for the resized image
            auto (bool): If True, use minimum rectangle to resize. If False, use new_shape directly
            scale_fill (bool): If True, stretch the image to new shape without padding
            scaleup (bool): If True, allow scaling up. If False, only scale down
            center (bool): If True, center the placed image. If False, place image in top-left corner
            stride (int): Stride of the model (e.g., 32 for YOLO)
            padding_value (int): Value for padding the image. Default is 114
            interpolation (int): Interpolation method for resizing. Default is cv2.INTER_LINEAR
        """
        self.new_shape=new_shape
        self.auto=auto
        self.scale_fill=scale_fill
        self.scaleup=scaleup
        self.stride=stride
        self.center=center # put the image in the middle or top-left
        self.padding_value=padding_value
        self.interpolation=interpolation

    @staticmethod
    def _update_labels(labels:dict[str, Any], ratio:tuple[float, float], padw:float, padh:float)->dict[str, Any]:
        """Update labels after applying letteringbox to an image

        This method modifies the bounding box coordinates of instances in the labels to account for resizing and padding applied during 
        letterboxing.

        Args:
            labels (dict[str, Any]): A dict containing image labels and instances
            ratio (tuple[float, float]): Scaling ratio (width, height) applied to the image
            padw (float): Padding width added to the image
            padh (float): Padding height added to the image
        Returns:
            (dict[str, Any]): Updated labels dict with modified instance coordinates
        """
        labels['instances'].convert_bbox(format='xyxy')
        labels['instances'].denormalize(*labels['img'].shape[:2][::-1]) # width, height
        labels['instances'].scale(*ratio)
        labels['instances'].add_padding(padw, padh)
        return labels

    def __call__(self,labels:dict[str, Any]|None=None, image:np.ndarray=None)->dict[str, Any]|np.ndarray:
        """Resize and pad an image for object detection, instance segmentation, or pose estimation tasks

        This method applies letterboxing to the input image, which involves resizing the image while maintaining its aspect ratio
        and adding padding to fit the new shape. It also updates any associated labels accordingly

        Args:
            labels (dict[str, Any]|None): A dict containing image data and associated labels, or None
            image (np.ndarray|None): The input image as a numpy array. If None, the image is taken from `labels`.
        Returns:
            (dict[str, Any]|np.ndarray): If `labels` is provided, returns an updated dict with the resized and padded image, updated labels,
                and additional metadata. If `labels` is not provided, return the resized and padded image only
        """
        if labels is None: labels={}
        img=labels.get('img') if image is None else image
        shape=img.shape[:2] # current shape (height, width)
        new_shape=labels.pop('rect_shape', self.new_shape)
        if isinstance(new_shape, int): new_shape=(new_shape, new_shape)

        # Scale ratio (new/old)
        r=min(new_shape[0]/shape[0], new_shape[1]/shape[1]) 
        if not self.scaleup: # only scale down, do not scale up (for better validation mAP)
            r=min(r, 1.)

        # Compute padding
        ratio=r,r # width, height ratios
        new_unpad=round(shape[1]*r), round(shape[0]*r) # width, height
        dw, dh=new_shape[1]-new_unpad[0], new_shape[0]-new_unpad[1] # width, height
        #print(f'Before auto dw {dw}, dh {dh} current shape {shape}')
        if self.auto: # minimum rectangle 
            # dw and dh are the smallest number to add to make new_unpad the next modulo of stride
            # let a be new_unpad, r is new_shape, and s is stride, i.e., dw=r-a
            # [a+ ((r−a)%s)]% s=0 
            dw, dh=np.mod(dw, self.stride), np.mod(dh, self.stride)
            #print(f'After auto dw {dw}, dh {dh}')
        elif self.scale_fill: # stretch 
            dw, dh=0., 0.
            new_unpad=(new_shape[1], new_shape[0]) # width, height
            ratio=new_shape[1]/shape[1], new_shape[0]/shape[0] # width, height ratios

        if self.center: # divide padding into 2 sides
            dw/=2
            dh/=2
        if shape[::-1]!=new_unpad: # resize
            img=cv2.resize(img, new_unpad, interpolation=self.interpolation)
            if img.ndim==2: img=img[...,None]

        top, bottom=round(dh-0.1) if self.center else 0, round(dh+0.1)
        left, right=round(dw-0.1) if self.center else 0, round(dw+0.1)
        h,w,c=img.shape
        if c==3:
            img=cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(self.padding_value,)*3)
        else: # multispectral
            pad_img=np.full((h+top+bottom, w+left+right, c), fill_value=self.padding_value, dtype=img.dtype)
            pad_img[top:top+h, left:left+w]=img
            img=pad_img
        if labels.get('ratio_pad'):
            labels['ratio_pad']=(labels['ratio_pad'], (left, top)) # for evaluation
        if len(labels):
            labels=self._update_labels(labels, ratio, left, top)
            labels['img']=img
            labels['resized_shape']=new_shape
            return labels
        return img


class CutMix(BaseMixTransform):
    """Apply CutMix augmentation to image datasets as described in the paper https://arxiv.org/abs/1905.04899.

    CutMix combines two images by replacing a random rectangular region of one image with the corresponding region from another image, and
    adjusts the labels proportionally to the area of the mixed region.

    Examples:
        >>> dataset=YourDataset(...)
        >>> cutmix=CutMix(dataset, p=0.5)
        >>> augmented_labels=cutmix(original_labels)
    """
    def __init__(self, dataset, pre_transform=None, p:float=0., beta:float=1., num_areas:int=3)->None:
        """Initialize the CutMix augmentation object
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
        
    def _rand_bbox(self, width:int, height:int)->tuple[int, int, int, int]:
        """Generate random bounding box coordinates for the cut region
        Args:
            width (int): Width of the image
            height (int): Height of the image
        Returns:
            (tuple[int, int,int,int]): (x1,y1,x2,y2) coordinates of the bounding box
        """
        # Sample mixing ratio from Beta distribution
        lam=np.random.beta(self.beta, self.beta) # controls the area ratio of the patch taken from image B and placed onto image A.

        cut_ratio=np.sqrt(1.0-lam)
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


    def _mix_transform(self, labels:dict[str, Any])->dict[str, Any]:
        """Apply CutMix augmentation to the input labels
        Args:
            labels (dict[str, Any]): A dict containing the original image and label information
        Returns:
            (dict[str, Any]): A dict containing the mixed image and adjusted labels
        """
        h, w=labels['img'].shape[:2]
        
        cut_areas=np.asarray([self._rand_bbox(w, h) for _ in range(self.num_areas)], dtype=np.float32) # NAx4 where NA is num_areas
        ioa1=bbox_ioa(cut_areas, labels['instances'].bboxes) # (self.num_areas, num_boxes)
        idx=np.nonzero(ioa1.sum(axis=1)<=0)[0] # find cut_areas that do not overlap with any ground truth boxes, i.e., sum==0
        if len(idx)==0: return labels # all cut_areas overlap with some ground truth boxes 
        
        labels2=labels.pop("mix_labels")[0]
        area=cut_areas[np.random.choice(idx)] # randomy select one of size (4,)
        ioa2=bbox_ioa(area[None], labels2["instances"].bboxes).squeeze(0) # 1xM -> M where M is the number of boxes in labels2['instances']
        
        # find cut_area that overlap with some mix-label boxes. assuming annotation are normalized
        indexes2=np.nonzero(ioa2>=(0.01 if len(labels2["instances"].segments) else 0.1))[0] # requires large overlap if annotation only contain boxes
        if len(indexes2)==0: return labels
        
        instances2=labels2['instances'][indexes2]
        instances2.convert_bbox('xyxy')
        instances2.denormalize(w,h)
        
        # Apply CutMix
        x1,y1,x2,y2=area.astype(np.int32)
        labels['img'][y1:y2, x1:x2]=labels2['img'][y1:y2, x1:x2]
        
        # Restrain instances2 to the random bounding border
        # shift the labels so its rectangle's top left corner become (0,0), so clipping works correctly
        # i.e., move the cut-out patch so its origin/top-left corner is at (0,0)
        instances2.add_padding(-x1, -y1)
        instances2.clip(x2-x1, y2-y1) # clip to remove annotation lies outside cut-out patch
        instances2.add_padding(x1, y1) # shift the annotation back to the coordinate system of the main image
        
        labels['cls']=np.concatenate([labels['cls'], labels2['cls'][indexes2]], axis=0)
        labels['instances']=Instances.concatenate([labels["instances"], instances2], axis=0)
    
        return labels

class RandomPerspective:
    """Implement random perspective and affine transformations on images and corresponding annotations

    This class applies random rotations, translations, scaling, shearing, and perspective transformations to images and their associated
    bounding boxes, segments, and keypoints. It can be used as part of an augmentation pipeline for object detection and instance 
    segmentation tasks

    Examples:
        >>> transform=RandomPerspective(degrees=10, translate=0.1, scale=0.1, shear=10)
        >>> image=np.random.randint(0, 255,(640,640,3),dtype=np.uint8)
        >>> labels={'img':image, 'cls':np.array([0,1]), 'instances':Instances(...)}
        >>> result=transform(labels)
        >>> transformed_image=result['img']
        >>> transformed_instances=result['instances']
    """
    def __init__(self, degrees:float=0., translate:float=0.1, scale:float=0.5, shear:float=0., perspective:float=0., 
                 border:tuple[int, int]=(0,0), pre_transform=None):
        """Initialize RandomPerspective object with transformation parameters
        Args:
            degrees (float): Degree range for random rotations
            translate (float): Fraction of total width and height for random translation
            scale (float): Scaling factor interval, e.g., a scale factor of 0.5 allows a resize between 50%-150%
            shear (float): Shear intensity (angle in degrees)
            perspective (float): Perspective distortion factor
            border (tuple[int,int]): Tuple specifying mosaic border (top/bottom, left/right)
            pre_transform (Callable|None): Function/transform to apply to the image before starting the random transform
        """
        self.degrees=degrees
        self.translate=translate
        self.scale=scale
        self.shear=shear
        self.perspective=perspective
        self.border=border # mosaic border
        self.pre_transform=pre_transform
        
    def affine_transform(self, img:np.ndarray, border:tuple[int, int],border_value:tuple[int,int,int]=(114, 114, 114))->tuple[np.ndarray, np.ndarray, float]:
        """Apply a sequence of affine transformations centered around the image center

        This function performs a series of geometric transformations on the input image, including translation, perspective
        change, rotation, scaling, and shearing. The transformtions are applied in a specific order to maintain consistency

        Args:
            img (np.ndarray): Input image to be transformed
            border (tuple[int,int]): Border dimensions for the mosaic image
            border_value (tuple[int, int, int]): Padding value, default to (114,114,114)
        Returns:
            img (np.ndarray): Transformed image
            M (np.ndarray): 3x3 transformation matrix
            s (float): Scale factor applied during transformation
        """
        # Center
        C=np.eye(3, dtype=np.float32)
        C[0,2]=-img.shape[1]/2 # x translation (pixels)
        C[1,2]=-img.shape[0]/2 # y translation (pixels)

        # Perspective
        P=np.eye(3, dtype=np.float32)
        P[2,0]=random.uniform(-self.perspective, self.perspective) # x perspective (about y)
        P[2,1]=random.uniform(-self.perspective, self.perspective) # y perspective (about x)

        # Rotation and scale
        R=np.eye(3, dtype=np.float32)
        a=random.uniform(-self.degrees, self.degrees)
        s=random.uniform(1-self.scale, 1+self.scale)
        R[:2]=cv2.getRotationMatrix2D(angle=a, center=(0,0), scale=s)

        # Shear
        S=np.eye(3, dtype=np.float32)
        S[0,1]=math.tan(random.uniform(-self.shear, self.shear)*math.pi/180.) # x shear (degrees)
        S[1,0]=math.tan(random.uniform(-self.shear, self.shear)*math.pi/180.) # y shear (degrees)

        # Translation
        T=np.eye(3, dtype=np.float32)
        T[0,2]=random.uniform(0.5-self.translate, 0.5+self.translate)*self.size[0] # x translation (pixels)
        T[1,2]=random.uniform(0.5-self.translate, 0.5+self.translate)*self.size[1] # y translation (pixels)

        # Combine transformations
        M=T @ S @ R @ P @ C  # order of operations (righ to left) is IMPORTANT
        # Affine image
        if (border[0]!=0) or (border[1]!=0) or (M!=np.eye(3, dtype=np.float32)).any(): # image changed
            if self.perspective: img=cv2.warpPerspective(img, M, dsize=self.size, borderValue=border_value)
            else: img=cv2.warpAffine(img, M[:2], dsize=self.size, borderValue=border_value)
            if img.ndim==2: img=img[...,None]
        return img, M, s

    def apply_bboxes(self, bboxes:np.ndarray, M:np.ndarray)->np.ndarray:
        """Apply affine transformation to bounding boxes
        
        This function applies an affine transformation to a set of bounding boxes using the provided transformation matrix
        Args:
            bboxes (np.ndarray): Bounding boxes in xyxy format with shape (N, 4), where N is the number of bounding boxes
            M  (np.ndarray): Affine transformation matrix with shape (3,3)
        Returns:
            (np.ndarray): Transformed bounding boxes in xyxy format with shape (N,4)
        """
        n=len(bboxes)
        if n==0: return bboxes

        xy=np.ones((n*4, 3), dtype=bboxes.dtype)
        xy[:,:2]=bboxes[:,[0,1,2,3,0,3,2,1]].reshape(n*4, 2) # x1y1,x2y2,x1y2,x2y1 the four corners of the boxes
        xy=xy @ M.T
        xy=(xy[:,:2]/xy[:,2:3] if self.perspective else xy[:,:2]).reshape(n, 8) # perspective rescale or affine

        # Create new boxes
        x=xy[:,[0,2,4,6]]
        y=xy[:,[1,3,5,7]]

        return np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1)),dtype=bboxes.dtype).reshape(4, n).T

    def apply_segments(self, segments:np.ndarray, M:np.ndarray)->tuple[np.ndarray, np.ndarray]:
        """Apply affine transformations to segments and generate new bounding boxes

        This function applies affine transformations to input segments and generates new bounding boxes based on the 
        transformed segments. It clips the transformed segments to fit within the new bounding boxes
        Args:
            segments (np.ndarray): Input segments with shape (N,M,2) where N is the number of segments, M is the number
                of points in each segment and 2 for x, y
            M (np.ndarray): Affine transformation matrix with shape (3,3)
        Returns:
            bboxes (np.ndarray): New bounding boxes with shape (N,4) in xyxy format
            segments (np.ndarray): Transformed and clipped segments with shape (N, M, 2)
        """
        n, num=segments.shape[:2]
        if n==0: return [], segments
        xy=np.ones((n*num, 3), dtype=segments.dtype)
        xy[:,:2]=segments.reshape(-1,2)
        xy = xy @ M.T # transform
        xy = xy[:,:2]/xy[:, 2:3] # if perspective xy[:,2:3] is not 1; otherwise, 1
        segments=xy.reshape(n, -1, 2)
        bboxes=np.stack([segment2box(xy, self.size[0], self.size[1]) for xy in segments], 0)
        segments[...,0]=segments[...,0].clip(bboxes[:, 0:1], bboxes[:, 2:3])
        segments[...,1]=segments[...,1].clip(bboxes[:, 1:2], bboxes[:, 3:4])
        return bboxes, segments

    def apply_keypoints(self, keypoints:np.ndarray, M:np.ndarray)->np.ndarray:
        """Apply affine transformation to keypoints

        This method transforms the input keypoints using the provided affine transformation matrix. It handles
        perspective rescaling if necessary and updates the visibility of keypoints that fall outside the image 
        boundaries after transform

        Args:
            keypoints (np.ndarray): Array of keypoints with shape (N, 17, 3), where N is the number of instances, 17 is the
                number of keypoints per instance, and 3 represent (x,y, visibility)
            M (np.ndarray): 3x3 affine transformation matrix
        Returns:
            (np.ndarray): Transformed keypoints with the same shape as input (N,17,3)
        """
        n, nkpt=keypoints.shape[:2]
        if n==0: return keypoints

        xy=np.ones((n*nkpt, 3), dtype=np.float32)
        visible=keypoints[...,2].reshape(n*nkpt, 1)
        xy[:,:2]=keypoints[...,:2].reshape(n*nkpt, 2)
        xy = xy @ M.T # transform
        xy = xy[:, :2]/xy[:, 2:3] # perspective rescale or affine
        outside_mask=(xy[:,0]<0) | (xy[:,1]<0) | (xy[:,0]>self.size[0]-1) | (xy[:,1]>self.size[1]-1)
        visible[outside_mask]=0
        return np.concatenate([xy, visible], axis=-1).reshape(n, nkpt, 3)

    @staticmethod
    def box_candidates(box1:np.ndarray, box2:np.ndarray, wh_thr:int=2, ar_thr:int=100, area_thr:float=0.1, eps:float=1e-16)->np.ndarray:
        """Compute candidate boxes for further processing based on size and aspect ratio criteria

        This method compares boxes before and after augmentation to determine if they meet specified thresholds for width, height, 
        aspect ratio, and area. It is used to filter out boxes that have been overly distorted or reduced by the augmentation process

        Args:
            box1 (np.ndarray): Original boxes before augmentation, shape (4, N) where N is the number of boxes with the format
                [x1, y1, x2, y2] in absolute coordinates
            box2 (np.ndarray): Augmented boxes after transformation, shape (4, N) with format []x1, y1, x2, y2] in absolute coordinates
            wh_thr (int): Width and height threshold in pixels. Boxes smaller than this in either dimension are rejected
            ar_thr (int): Aspect ratio threshold. Boxes with an aspect ratio greater than this value are rejected
            area_thr (float): Area ratio threshold. Boxes with an area ratio (new/old) less than this value are rejected
            eps (float): Small epsilon value to prevent division by zero
        Returns:
            (np.ndarray): Boolean array of shape (N,) indicating whether boxes are candidates. True values correspond to boxes that
                meet/pass all criteria
        """
        w1, h1=box1[2]-box1[0], box1[3]-box1[1]
        w2, h2=box2[2]-box2[0], box2[3]-box1[1]
        # Aspect ratio
        ar=np.maximum(w2/(h2+eps), h2/(w2+eps))
        # print(f'In data.augment.RandomPerspective.box_candidate w2 {w2}>2 {w2>wh_thr}')
        # print(f'In data.augment.RandomPerspective.box_candidate h2 {h2}>2 {h2>wh_thr}')
        # print(f'In data.augment.RandomPerspective.box_candidate (w2*h2 {w2*h2}/(w1*h1+eps {w1*h1+eps})>0.1 {((w2*h2/(w1*h1+eps)) > area_thr)}')
        # print(f'In data.augment.RandomPerspective.box_candidate ar {ar}<100 {ar<ar_thr}')
        return (w2>wh_thr) & (h2>wh_thr) & ((w2*h2/(w1*h1+eps)) > area_thr) & (ar<ar_thr)

    def __call__(self,labels:dict[str, Any])->dict[str, Any]:
        """Apply random perspective and affine transform to an image and its associated annotation

        This method performs a series of transformations including rotation, translation, scaling, shearing, and
        perspective distortion on the input image and adjusts the corresponding bounding boxes, segments, and keypoints,
        accordingly

        Args:
            labels (dict[str, Any]): A dict containing image data and annotations, including
                -'img' (np.ndarray): The input image
                -'cls' (np.ndarray): Class labels
                -'instances' (Instances): object instances with bounding boxes, segments, and keypoints
                -'mosaic_border' (tuple[int, int], optional): Border size of mosaic augmentation
        Returns:
            (dict[str, Any]): Transformed labels dict containing:
                -'img' (np.ndarray): The transformed image
                -'cls' (np.ndarray): Updated class labels
                -'instances' (Instances): Updated object instances
                -'resized_shape' (tuple[int, int]): New image shape after transformation
        """
        # We only apply pre_transform if labels did NOT previously go through mosaic transform
        if self.pre_transform and 'mosaic_border' not in labels: labels=self.pre_transform(labels)
        labels.pop('ratio_pad', None) # we do not need ratio_pad

        img=labels['img']
        cls=labels['cls']
        instances=labels.pop('instances')
        # Make sure the coordinate formats are the right one
        instances.convert_bbox(format='xyxy')
        instances.denormalize(*img.shape[:2][::-1])

        border=labels.pop('mosaic_border', self.border)
        #print(f'In data.augment.RandomPerspective.__call__ border {border}')
        self.size=img.shape[1]+border[1]*2, img.shape[0]+border[0]*2 # width, height
        # M is an affine matrix
        # scale is to be passed to `box_candidates` function
        img, M, scale=self.affine_transform(img, border)

        bboxes=self.apply_bboxes(instances.bboxes, M)
        segments=instances.segments
        keypoints=instances.keypoints
        # Update bboxes if there are segments
        if len(segments): bboxes, segments=self.apply_segments(segments, M)
        if keypoints is not None: keypoints=self.apply_keypoints(keypoints, M)
        new_instances=Instances(bboxes, segments, keypoints, bbox_format='xyxy', normalized=False)
        # Clip
        new_instances.clip(*self.size)

        # Filter instances
        instances.scale(scale_w=scale, scale_h=scale,bbox_only=True)
        # Make the bboxes have the same scale with new_bboxes
        i=self.box_candidates(box1=instances.bboxes.T, box2=new_instances.bboxes.T, area_thr=0.01 if len(segments) else 0.1)
        labels['instances']=new_instances[i]
        labels['cls']=cls[i]
        labels['img']=img
        labels['resized_shape']=img.shape[:2]
        return labels

class CopyPaste(BaseMixTransform):
    """CopyPaste class for applying Copy-Paste augmentation to image datasets

    This class implements the Copy-Paste augmentation technique as described in the paper "Simple Copy-Paste is a Strong
    Data Augmentation Method for Instance Segmentation" (https://arxiv.org/abs/2012.07177). It combines objects from different
    images to create new training samples

    Examples:
        >>> dataset=YourDataset(...) 
        >>> copypaste=CopyPaste(dataset, p=0.5)
        >>> augmented_labels=copypaste(original_labels)
    """
    def __init__(self, dataset=None, pre_transform=None, p:float=0.5, mode:str='flip')->None:
        """Initialize CopyPaste object with dataset, pre_transform, and probability of applying CopyPaste"""
        super().__init__(dataset=dataset, pre_transform=pre_transform, p=p)
        assert mode in {'flip', 'mixup'}, f'Expected `mode` to be `flip` or `mixup`, but got {mode}'
        self.mode=mode
    def _mix_transform(self, labels:dict[str, Any])->dict[str, Any]:
        """Apply CopyPaste augmentation to combine objects from another image into the current image"""
        labels2=labels['mix_labels'][0]
        return self._transform(labels, labels2)
    def __call__(self, labels:dict[str, Any])->dict[str, Any]:
        """Apply CopyPaste augmentatioon to an image and its labels"""
        if len(labels['instances'].segments)==0 or self.p==0: return labels
        if self.mode=='flip': return self._transform(labels)

        # Get index of aother image
        indexes=self.get_indexes()
        if isinstance(indexes, int): indexes=[indexes]

        # Get image information will be used for copy-paste
        mix_labels=[self.dataset.get_image_and_label(i) for i in indexes]

        if self.pre_transform is not None:
            for i, data in enumerate(mix_labels): mix_labels[i]=self.pre_transform(data)
        labels['mix_labels']=mix_labels

        # Update cls and texts
        labels=self._update_label_text(labels)
        # Copypaste
        labels=self._mix_transform(labels)
        labels.pop('mix_labels', None)
        return labels

    def _transform(self, labels1:dict[str, Any], labels2:dict[str, Any]={})->dict[str, Any]:
        """Apply CopyPaste augmentation to combine objects from another image into the current image"""
        
        im=labels1['img']
        if 'mosaic_border' not in labels1: im=im.copy() # avoid modifying original non-mosaic image
        cls=labels1['cls'] # (M,1)
        h, w=im.shape[:2]
        instances=labels1.pop('instances')
        instances.convert_bbox(format='xyxy')
        instances.denormalize(w, h)
        
        instances2=labels2.pop('instances', None)
        if instances2 is None:
            instances2=deepcopy(instances)
            instances2.fliplr(w)
        # Find boxes in instances2 that have small overlap with instances1.bboxes
        ioa=bbox_ioa(instances2.bboxes, instances.bboxes) # intersect over area of instances.bboxes (N,M)
        indexes=np.nonzero((ioa<0.3).all(1))[0] #(N,) overlap over area < 0.3
        n=len(indexes)
        # (N,M) for each N, find max M. Then get max-ioa of indexes
        # Then get softed-indices from small ioa to large
        sorted_idx=np.argsort(ioa.max(1)[indexes]) # get the index of instances2.bboxes that least overlap with instances1.bboxes
        indexes=indexes[sorted_idx]
        ins_mask=np.zeros(im.shape, np.uint8) # mask of instances2 annotation, i.e., area to write on labels1['img']
        for j in indexes[:round(self.p*n)]:# number of pastes controlled by probability
            cls=np.concatenate((cls, labels2.get('cls',cls)[[j]]), axis=0) # advanced/fancy indexing to preserve original dimension
            instances = Instances.concatenate((instances, instances2[[j]]), axis=0)
            # instances2.segments is NxPx2 so instances2.segments[[j]] is 1xPx2
            cv2.drawContours(ins_mask, instances2.segments[[j]].astype(np.int32), -1, (1,1,1), cv2.FILLED)
        result=labels2.get('img', cv2.flip(im, 1)) # flip horizontally around y axis, i.e., flip left/right
        if result.ndim==2: # cv2.flip would eliminate the last dimension for grayscale images
            result=result[...,None]
        i=ins_mask.astype(bool)
        im[i]=result[i]
        labels1['img']=im
        labels1['cls']=cls
        labels1['instances']=instances
        return labels1

class RandomFlip:
    """Apply a random horizontal or vertical flip to an image with a given probability

    This class performs random image flipping and updates corresponding instance annotations such as bounding boxes and keypoints

    Examples:
        >>> transform=RandomFlip(p=0.5, direction='horizontal')
        >>> result=transform({'img':image, 'instances':instances})
        >>> flipped_image=result['img']
        >>> flipped_instances=result['instances']
    """
    def __init__(self, p:float=0.5, direction:str='horizontal', flip_idx:list[int]|None=None)->None:
        """Initialize the RandomFlip class with probability and direction

        This class applies a random horizontal or vertical flip to an image with a given probability. It also updates any instances (bounding boxes,
        keypoints, etc) accordingly

        Args:
            p (float): The probability of applying the flip. Must be between 0 and 1
            direction (str): The direction to apply the flip. Must be 'horizontal' or 'vertical'
            flip_idx (list[int]|None): Index mapping for flipping keypoints, if any
        """
        assert direction in {'horizontal', 'vertical'}, f'Support direction "horizontal" or "vertical, got {direction}"'
        assert 0<=p<=1., f'The probability should be in range [0,1], but got {p}'
        self.p=p
        self.direction=direction
        self.flip_idx=flip_idx
        
    def __call__(self, labels:dict[str, Any])->dict[str, Any]:
        """Apply random flip to an image and update any instances like bounding boxes or keypoints accordingly

        This method randomly flips the input image either horizontally or vertically based on the initialized probability and direction
        It also updates teh corresponding instances (bounding boxes,keypoints) to match the flipped image
        
        Args:
            labels (dict[str, Any]): A dict containing the following keys:
                - 'img' (np.ndarray): The image to be flipped
                - 'instances' (Instances): Object containing boxes and optionally keypoints
        Returns:
            (dict[str, Any]): The dict with the flipped image and updated instances:
                - 'img' (np.ndarray): The flipped image
                - 'instances' (Instances): Updated instances matching the flipped image
        Examples:
            >>> labels={"img":np.random.rand(640,640,3), "instances":Instances(...)}
            >>> random_flip=RandomFlip(p=0.5,direction="horizontal")
            >>> flipped_labels=random_flip(labels)
        """
        img=labels['img']
        instances=labels.pop("instances")
        instances.convert_bbox(format='xywh')
        h, w=img.shape[:2]
        if instances.normalized: h=w=1
        
        # WARNING: two separate if and calls to random.random() intentional for reproducibility with older versions
        if self.direction=='vertical' and random.random()<self.p:
            img=np.flipud(img)
            instances.flipud(h)
            # I do not think we need to reordering the keypoints after flipud
            # if self.flip_idx is not None and instances.keypoints is not None: 
            #     instances.keypoints=np.ascontiguousarray(instances.keypoints[:,self.flip_idx,:])
        if self.direction=='horizontal' and random.random()<self.p:
            img=np.fliplr(img)
            instances.fliplr(w)
            if self.flip_idx is not None and instances.keypoints is not None:
                instances.keypoints=np.ascontiguousarray(instances.keypoints[:, self.flip_idx,:])
        labels['img']=np.ascontiguousarray(img)
        labels['instances']=instances
        return labels

class RandomHSV:
    """Randomly adjust the Hue, Saturation, and Value (HSV) channels of an image

    This class applies random HSV augmentation to images within predefined limits set by hgain, sgain, vgain

    Examples:
        >>> augmenter=RandomHSV(hgain=0.5, sgain=0.5, vgain=0.5)
        >>> image=np.random.randint(0, 255, (100,100,3), dtype=np.uint8)
        >>> labels={'img':image}
        >>> augmenter(labels)
    """
    def __init__(self, hgain:float=0.5, sgain:float=0.5, vgain:float=0.5)->None:
        """Initialize the RandomHSV object for random HSV (Hue, Saturation, Value) augmentation

        This class applies random adjustments to the HSV channels of an image within specified limits
        Args:
            hgain (float): Maximum variation for hue. Should be in the range [0,1]
            sgain (float): Maximum variation for saturation. Should be in the range [0,1]
            vgain (float): Maximum variation for value. Should be in the range [0,1]
        """
        self.hgain=hgain
        self.sgain=sgain
        self.vgain=vgain

    def __call__(self, labels:dict[str, Any])->dict[str, Any]:
        """Apply random HSV augmentation to an image within predefined limits
        
        This method modifies the input image by randomly adjusting its Hue, Saturation, and Value (HSV) channels. The adjustment
        are made within th elimits set by the hgain, sgain, and vgain during initialization

        Args:
            labels (dict[str, Any]): A dict containing image data and metadata. Must include an 'img' key with the image as a numpy
                array
        Returns:
            (dict[str, Any]): A dict containing the mixed image and adjusted labels
        """
        img=labels['img']
        if img.shape[-1]!=3: return labels # only apply to image with 3 pixel-value channels

        if self.hgain or self.sgain or self.vgain:
            dtype=img.dtype # uint8

            r=np.random.uniform(-1, 1, 3)*[self.hgain, self.sgain, self.vgain] # (3,) ndarray of random gains
            x=np.arange(0, 256, dtype=r.dtype)
            # lut_hue=(( x* (r[0]+1)) %180).astype(dtype) # original hue implementatioon from ultralytics<=8.3.78
            lut_hue=((x+r[0]*180)%180).astype(dtype)
            lut_sat=np.clip(x*(r[1]+1), 0, 255).astype(dtype)
            lut_val=np.clip(x*(r[2]+1), 0, 255).astype(dtype)
            lut_sat[0]=0 # prevent pure white changing color, introduced in 8.3.79

            hue, sat, val=cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)) # each of size (H,W)
            im_hsv=cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            cv2.cvtColor(im_hsv, cv2.COLOR_HLS2BGR, dst=img) # no return needed
        return labels

class Compose:
    """A class for composing multiple image augmentation transformation

    Examples:
        >>> transforms=[RandomFlip(), RandomPerspective()]
        >>> compose=Compose(transforms)
        >>> transformed_data=compose(data)
        >>> compose.append(RandomHSV())
        >>> compose.insert(0, LetterBox())
    """
    def __init__(self, transforms):
        """Initialize the Compose object with a list of transforms

        Args:
            transforms (list[Callable]): A list of callable transform objects to be applied sequentially
        """
        self.transforms=transforms if isinstance(transforms, list) else [transforms]

    def __call__(self, data):
        """Apply a series of transformations to input data

        This method sequentially applies each transformation in the Compose object to the input data

        Args:
            data (Any): The input data to be transformed. The type depends on the required input of the transformations in the list
        Returns:
            (Any): The transformed data after applying all transformation in the sequence
        Examples:
            >>> compose=Compose([Transform1(), Transform2(), Transform3()])
            >>> transformed_data=compose(input_data)
        """
        for t in self.transforms: data=t(data)
        return data
    
    def append(self, transform):
        """Append a new transform to the existing list of transforms

        Args:
            transform (callable): The transformation to be added to the composition
        """
        self.transforms.append(transform)

    def insert(self, index, transform):
        """Insert a new transform at a specified index in the existing list of transformations

        Args:
            index (int): The index at which to insert the new transform
            transform (callable): The transform to be inserted
        """
        self.transforms.insert(index, transform)

    def __getitem__(self, index:list[int]|int)->Compose:
        """Retrieve a specific transform or a set of transforms using indexing
        Args:
            index (list[int]|int): Index or list of indices of the transforms to retrieve
        Returns:
            (Compose): A new Compose object containing the selected transform(s)
        Examples:
            >>> compose=Compose([RandomFlip(), RandomPerspective(), RandomHSV()])
            >>> single_transform=compose[1] # Return a Compose object with only RandomPerspective 
            >>> multiple_transforms=compose[0:2] # Returns a Compose object with RandomFlip and RandomPerspective
        """
        assert isinstance(index, (int, list)), f'The indices should be either list or int type but got {type(index)}'
        return Compose([self.transforms[i] for i in index]) if isinstance(index,list) else self.transforms[index]

    def __setitem__(self, index:list[int]|int, value:list[Any]|Any)->None:
        """Set one or more transforms in the composition using indexing

        Args:
            index (list[int]|int): Index or list of indices to set transforms at
            value (list[Any]|Any): Transform or list of transforms to set at the specified index(es)
        Examples:
            >>> compose=Compose([Transform1(), Transform2(), Transform3()])
            >>> compose[1]=NewTransform() # Replace second transform
            >>> compose[0:2]=[NewTransform1(), NewTransform2()] # Replace first two transforms
        """
        assert isinstance(index, (int, list)), f'The indices should be either list or int type but got {type(index)}'
        if isinstance(index, list):
            assert isinstance(value, list), f'The indices should be the same type as values, but got {type(index)} and {type(value)}'
        if isinstance(index, int): index, value=[index],[value]
        for i, v in zip(index, value):
            assert i<len(self.transforms), f'list index {i} out of range {len(self.transforms)}'
            self.transforms[i]=v
            
    def tolist(self):
        """Convert the list of transforms to a standard Python list
        Returns:
            (list): A list containing all the transform objects in the Compose instance
        Examples:
            >>> compose=Compose([RandomFlip(), RandomPerspective(), RandomHSV()])
            >>> transform_list=compose.tolist()
            >>> print(len(transform_list))
            3
        """
        return self.transforms

    def __repr__(self):
        """Return a string representation of the Compose object
        Returns:
            (str): A string representation of the Compose object, including the list of transforms
        Examples:
            >>> transforms=[RandomFlip(), RandomPerspective(degrees=10, translate=0.1, scale=0.1)]
            >>> compose=Compose(transforms)
            >>> print(compose)
            Compose([
                RandomFlip(),
                RandomPerspective(degrees=10, translate=0.1, scale=0.1)
            ])
        """
        return f"{self.__class__.__name__}({', '.join(f'{t}' for t in self.transforms)})"


def v8_transforms(dataset, imgsz:int, hyp:IterableSimpleNamespace, stretch:bool=False):
    """Apply a series of image transformations for training
    
    This function creates a composition of image augmentation techniques to prepare images for YOLO training. It includes
    operations such as mosaic, copy-paste, random perspective, mixup and HSV adjustment
    
    Args:
        dataset (Dataset): The dataset object containing image data and annotations
        imgsz (int): The target image size for resizing
        hyp (IterableSimpleNamespace): A dict of hyperparameters controlling various aspects of the transformation
        stretch (bool): If True, applies stretching to the image. If False, use LetterBox resizing
    Returns:
        (Compose): A composition of image transformations to be applied to the dataset
    Examples:
        >>> dataset=YOLODataset(img_path='path/to/images', imgsz=640)
        >>> hype=IterableSimpleNamespace(mosaic=1., copy_paste=0.5, degrees=10, translate=0.2, scale=0.9)
        >>> transforms=v8_transforms(dataset, imgsz=640, hyp=hyp)
        >>> augmented_data=transforms(dataset[0])
    """
    mosaic=Mosaic(dataset, imgsz=imgsz,p=hyp.mosaic)
    # If not stretch, LetterBox will only be applied if mosaic is not used, i.e., hyp.mosaic=0
    affine=RandomPerspective(degrees=hyp.degrees, translate=hyp.translate, scale=hyp.scale, shear=hyp.shear, 
                             perspective=hyp.perspective, pre_transform=None if stretch else LetterBox(new_shape=(imgsz, imgsz)))
    pre_transform=Compose([mosaic, affine])
    if hyp.copy_paste_mode=='flip':
        pre_transform.insert(1, CopyPaste(p=hyp.copy_paste, mode=hyp.copy_paste_mode))
    else:
        pre_transform.append(
            CopyPaste(dataset, pre_transform=Compose([Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic), affine]),
                      p=hyp.copy_paste, mode=hyp.copy_paste_mode
                     )
        )
    
    flip_idx=dataset.data.get('flip_idx', []) # for keypoints augmentation
    if dataset.use_keypoints:
        kpt_shape=dataset.data.get('kpt_shape',None)
        if len(flip_idx)==0 and hyp.fliplr>0.:
            hyp.fliplr=0. # need flip_idx to flip keypoint annotation labels from left to right
            warnings.warn("No 'flip_idx' array definedin data.yaml, disabling 'fliplr' augmentation")
        elif flip_idx and (len(flip_idx)!=kpt_shape[0]):
            raise ValueError(f'data.yaml flip_idx={flip_idx} length must be equal to kpt_shape[0]={kpt_shape[0]}')
    return Compose(
        [pre_transform,
         MixUp(dataset, pre_transform=pre_transform, p=hyp.mixup),
         CutMix(dataset, pre_transform=pre_transform, p=hyp.cutmix),
         RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
         RandomFlip(direction='vertical', p=hyp.flipud, flip_idx=None),
         RandomFlip(direction='horizontal', p=hyp.fliplr, flip_idx=flip_idx),
        ]
    ) # transforms

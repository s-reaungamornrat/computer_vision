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
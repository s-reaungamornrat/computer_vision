from __future__ import annotations

from typing import Any

import cv2
import random
import numbers
import numpy as np

from computer_vision.yolov11.instance.instance import Instances
from computer_vision.yolov11.utils.metrics import bbox_ioa

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
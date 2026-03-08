from __future__ import annotations
from typing import Optional, Union

import numpy as np
from torch.nn.modules.utils import _pair

from computer_vision.slowfast.mmengine.utils.misc import is_tuple_of
from computer_vision.slowfast.mmcv.image.photometric import iminvert
from computer_vision.slowfast.mmcv.image.geometric import rescale_size, imresize, imflip


def _init_lazy_of_proper(results, lazy):
    """Initialize lazy operation properly

    Make sure that a lazy operation is properly initialized, and avoid a non-lazy operation accidentally getting mixed in.

    Required keys in results are 'imgs' if 'img_shape' not in results; otherwise, required keys are 'img_shape' and add or 
    modified keys are 'img_shape' and 'lazy'
    Added or modified keys in 'lazy' are 'original_shape', 'crop_bbox', 'flip', 'flip_direction', 'interpolation'

    Args:
        results (dict): A dict storing data pipeline results
        lazy (bool): Whether to apply lazy operation. Default to False
    """
    if 'img_shape' not in results: results['img_shape']=results['imgs'][0].shape[:2] # (H,W)
    if lazy:
        img_h, img_w=results['img_shape']
        lazyop={'original_shape':results['img_shape']}
        lazyop['crop_bbox']=np.array([0,0,img_w, img_h], dtype=np.float32)
        lazyop['flip']=False
        lazyop['flip_direction']=None
        lazyop['interpolation']=None
        results['lazy']=lazyop
    else: assert 'lazy' not in results


class Resize:
    """Resize images to a specific size

    Required Keys: 'img_shape', 'modality', 'imgs' (optional), 'keypoint' (optional)
    Added or Modified Keys: 'imgs', 'img_shape', 'keep_ratio', 'scale_factor', 'lazy', 'resize_size'. Required keys in 'lazy' is None,
        added or modified keys is 'interpolation'

    Args:
        scale (tuple[number]): If keep_ratio is True, it serves as a maximum size along width and height dimension; often, defined as 
            (-1, min_size) or (min_size, -1) to scale an image such that its shortest size is min_size. Otherwise, it serves as (w,h) 
            of output size
        keep_ratio (bool): If True, image will be resized without changing the aspect ratio. Otherwise, it will be resized to the given size
            Default: True
        interpolation (str): Algorithm used for interpolation with options of 'nearest', 'bilinear', 'bicubic', 'area' and 'lanczos'. 
            Default: 'bilinear'
        lazy (bool): Whether to apply lazy operation. Default: False
    """
    def __init__(self, scale, keep_ratio=True, interpolation='bilinear', lazy=False):
        
        if isinstance(scale, float): assert scale>0, f'Invalid scale value {scale}, must be positive'
        elif isinstance(scale, tuple):
            max_long_edge=max(scale)
            max_short_edge=min(scale)
            if max_short_edge==-1: scale=(np.inf, max_long_edge) 
        else: raise TypeError(f"scale must be float or tuple of int, but got {type(scale)}")

        self.scale=scale
        self.keep_ratio=keep_ratio
        self.interpolation=interpolation
        self.lazy=lazy

    def __call__(self, results:dict)->Optional[Union[dict, tuple[list,list]]]:
        return self.transform(results)

    def _resize_imgs(self, imgs, new_w, new_h):
        """
        Args:
            imgs (list[np.ndarray]): List of images, each is of size (H,W,C)
             new_w (int): Desired image width
             new_h (int): Desired image height 
        """
        return [
            imresize(img, (new_w, new_h), interpolation=self.interpolation) for img in imgs
        ]

    @staticmethod
    def _resize_kps(kps, scale_factor):
        """
        Args:
            kps (np.ndarray): Keypoint probaby of size (M,T,V,2) where 2 is for x, y and M is the number of persons (instances) in the frame
                T is the number of frames, V is the number of keypoints (e.g., 17 for COCO)
            scale_factor (np.ndarray): Scale along width and height, of size (2,)
        """
        return kps*scale_factor
        
    @staticmethod
    def _box_resize(box, scale_factor):
        """ Rescale bounding boxes according to scale_factor
        Args:
            box (np.ndarray): The bounding boxes
            scale_factor (np.ndarray): The scale factor along width and height dimension of size (2,)
        """
        assert len(scale_factor)==2
        scale_factor=np.concatenate([scale_factor, scale_factor]) # [scale_w, scale_h, scale_w, scale_h]
        return box*scale_factor

    def transform(self, results):
        """Perform the resize augmentation
        Args:
            results (dict): The resulting dict to be modified and passed to the next transform in the pipeline
        """
        _init_lazy_of_proper(results, self.lazy)
        
        if any(x in results for x in ['gt_bboxes','keypoint']): 
            assert not resize_op.lazy, "Keypoint/Bounding box augmentation are not compatible with lazy==True"
            
        if 'scale_factor' not in results: results['scale_factor']=np.array([1,1], dtype=np.float32)
            
        img_h, img_w=results['img_shape']
        if self.keep_ratio:
            new_w, new_h=rescale_size((img_w, img_h), self.scale)
        else: new_w, new_h=self.scale
        
        self.scale_factor=np.array([new_w/img_w, new_h/img_h], dtype=np.float32)
        
        results['img_shape']=(new_h, new_w)
        results['keep_ratio']=self.keep_ratio
        results['scale_factor']=results['scale_factor']*self.scale_factor
        
        if not self.lazy:
            if 'imgs' in results:
                results['imgs']=self._resize_imgs(results['imgs'], new_w, new_h)
            if 'keypoint' in results: 
                results['keypoint']=self._resize_kps(results['keypoint'], self.scale_factor)
        else:
            lazyop=results['lazy']
            if lazyop['flip']: raise NotImplementedError('Put Flip at last for now')
            lazyop['interpolation']=self.interpolation
            
        if 'gt_bboxes' in results:
            results['gt_bboxes']=self._box_resize(results['gt_bboxes'], self.scale_factor)
            if 'proposals' in results and results['proposals'] is not None:
                assert results['proposals'].shape[1]==4
                results['proposals']=self._box_resize(results['proposals'], self.scale_factor)
                
        return results

    def __repr__(self)->str:
        return f"{self.__class__.__name__}(scale={self.scale}, keep_ratio={self.keep_ratio}, interpolation={self.interpolation}, lazy={self.lazy})"

class RandomResizedCrop:
    """Random crop based on area and heigh-width ratio range

    Required keys are 'img_shape', 'crop_bbox', 'imgs' (optional), 'keypoint' (optional), added or modified keys are 'imgs', 'keypoint',
    'crop_bbox', and 'lazy'. Required keys in 'lazy' are 'flip', 'crop_bbox', and added or modified key is 'crop_bbox'

    Args:
        area_range (tuple[float]): The candidate area scale range of output cropped images. Default: (0.08, 1.)
        aspect_ratio (tuple[float]): The candidate aspect ratio range of the output cropped images, where aspect ratio is width/height. 
            Default: (3/4, 4/3)
        lazy (bool): Whether to apply lazy operation. Default: False
    """
    def __init__(self, area_range=(0.08, 1.), aspect_ratio_range=(3/4, 4/3), lazy=False):
        
        self.area_range=area_range
        self.aspect_ratio_range=aspect_ratio_range
        self.lazy=lazy

        assert is_tuple_of(self.area_range, float), f"Area range must be a tuple of float, but got {type(area_range)}"
        assert is_tuple_of(self.aspect_ratio_range, float), f"Aspect ratio range must be a tuple of type float, but got {type(aspect_ratio_range)}"


    def __call__(self, results:dict)->Optional[Union[dict, tuple[list,list]]]:
        return self.transform(results)

    @staticmethod
    def get_crop_bbox(img_shape, area_range, aspect_ratio_range,max_attempts=10):
        """Get a crop bbox given the area range and aspect ratio range
        Args:
            img_shape (tuple[int]): Image size (height, width)
            area_range (tuple[float]): The area scale range of output cropped images. 
            aspect_ratio_range (tuple[float]): The aspect ratio range between width and height, i.e., width/height
            max_attempts (int): The maximum number of times attempted to generate random candidate bounding box. If it does not qualified one,
                the center bounding box will be used
        Returns:
            (list[int]): A random crop bbox within the area range and aspect ratio range
        """
        assert 0 < area_range[0] <= area_range[1] <= 1
        assert 0 < aspect_ratio_range[0]<=aspect_ratio_range[1]
        
        img_h, img_w=img_shape
        area=img_h*img_w
        
        min_ar, max_ar=aspect_ratio_range
        # by sampling in log-space, we treat "tallness" and "wideness" as equal opposites.e.g., 
        # ensuring that a 1:3 box and a 3:1 box have an equal probability of being selected.
        aspect_ratios=np.exp(np.random.uniform(np.log(min_ar), np.log(max_ar), size=max_attempts))
        target_areas=np.random.uniform(*area_range, size=max_attempts)*area
        candidate_crop_w=np.round(np.sqrt(target_areas*aspect_ratios)).astype(np.int32)
        candidate_crop_h=np.round(np.sqrt(target_areas/aspect_ratios)).astype(np.int32)
        
        for i in range(max_attempts):
            crop_w=candidate_crop_w[i]
            crop_h=candidate_crop_h[i]
            #print(f"{i=}, {crop_w=}, {crop_h=}, {img_w=}, {img_h=}")
            if crop_h<=img_h and crop_w<=img_w:
                x_offset=np.random.randint(0, img_w-crop_w)
                y_offset=np.random.randint(0, img_h-crop_h)
                return x_offset, y_offset, x_offset+crop_w, y_offset+crop_h
        
        # fall back
        crop_size=min(img_h, img_w)
        x_offset=(img_w-crop_size)//2
        y_offset=(img_h-crop_size)//2
        return x_offset, y_offset, x_offset+crop_size, y_offset+crop_size

    @staticmethod
    def _crop_kps(kps, crop_bbox):
        """Crop keypoints
        Args:
            kps (np.ndarray): Keypoint probaby of size (M,T,V,2) where 2 is for x, y and M is the number of persons (instances) in the frame
                T is the number of frames, V is the number of keypoints (e.g., 17 for COCO)
            crop_bbox (np.ndarray): 1D bounding box array of x1,y1,x2,y2 in pixel units, where (x1,y1) is the top left corner and (x2,y2) is the
                bottom right corner
        """
        return kps-crop_bbox[:2] # subtract x1,y1
        
    @staticmethod
    def _crop_imgs(imgs, crop_bbox):
        """Crop each frame
        Args:
            imgs (list[np.ndarray]): List of video frames, each is of size (H,W,C)
            crop_bbox (np.ndarray): 1D bounding box array of x1,y1,x2,y2 in pixel units, where (x1,y1) is the top left corner and (x2,y2) is the
                bottom right corner
        """
        x1,y1,x2,y2=crop_bbox
        return [img[y1:y2, x1:x2] for img in imgs]

    @staticmethod
    def _box_crop(box, crop_bbox):
        """Crop the bounding boxes according to the crop_bbox
        Args:
            box (np.ndarray): The bounding boxes of size (*,4) where * can be N (number of boxes) or more dimensions (such as (B,N) where B is
                the batch size), and 4 is for x1,y1,x2,y2 in pixel units where (x1,y1) is the top left corner and (x2,y2) is the
                bottom right corner
            crop_bbox (np.ndarray): 1D bounding box array of x1,y1,x2,y2 in pixel units, where (x1,y1) is the top left corner and (x2,y2) is the
                bottom right corner
        Returns:
            (np.ndarray): The bounding boxes after cropping with same size as the input box
        """
        x1,y1,x2,y2=crop_bbox
        width, height=x2-x1, y2-y1
        box_=box.copy()
        box_[...,0::2]=np.clip(box[...,0::2]-x1, 0, width-1)
        box_[...,1::2]=np.clip(box[...,1::2]-y1, 0, height-1)
        return box_
        
    def _all_box_crop(self, results, crop_bbox):
        """Crop the gt_bboxes and proposals in results according to crop_bbox
        Args:
            results (dict): All information about the sample, which contain 'gt_bboxes' and 'proposals' (optional)
            crop_bbox (np.ndarray): 1D bounding box array of x1,y1,x2,y2 in pixel units, where (x1,y1) is the top left corner and (x2,y2) is the
                bottom right corner
        """
        results['gt_bboxes']=self._box_crop(results['gt_bboxes'], crop_bbox)
        if 'proposals' in results and results['proposals'] is not None:
            assert results['proposals'].shape[-1]==4
            results['proposals']=self._box_crop(results['proposals'], crop_bbox)
        return results
    
    def transform(self, results):
        """Perform random resized crop augmentation
        Args:
            results (dict): The resulting dict to be modified and passed to the next transform in the pipeline
        """
        _init_lazy_of_proper(results, self.lazy)
        if any(x in results for x in ['keypoint', 'gt_bboxes']):
            assert not self.lazy, "Keypoint/Bounding box augmentations are not compatible with lazy==True"
        
        img_h, img_w=results['img_shape']
        left,top,right,bottom=self.get_crop_bbox((img_h, img_w), self.area_range, self.aspect_ratio_range) # x1,y1,x2,y2
        new_h, new_w=bottom-top, right-left
        
        # crop_quadruple is a 4-element tuple or list that defines the cropping window using the format [x,y,w,h] where x,y is the yop pixel coordinates
        if 'crop_quadruple' not in results: results['crop_quadruple']=np.array([0,0,1,1],dtype=np.float32) # x,y,w,h
        # normalize x1,y1,w,h by image size
        x_ratio, y_ratio=left/img_w, top/img_h
        w_ratio, h_ratio=new_w/img_w, new_h/img_h
        
        old_crop_quadruple=results['crop_quadruple']
        old_x_ratio, old_y_ratio=old_crop_quadruple[0], old_crop_quadruple[1]
        old_w_ratio, old_h_ratio=old_crop_quadruple[2], old_crop_quadruple[3]
        
        # normalized crop ratio relative to the original image (taking into account the previous crop) 
        new_crop_quadruple=[old_x_ratio+x_ratio*old_w_ratio, 
                            old_y_ratio+y_ratio*old_h_ratio,
                            w_ratio*old_w_ratio,
                            h_ratio*old_h_ratio]
        results['crop_quadruple']=np.array(new_crop_quadruple, dtype=np.float32)
        
        crop_bbox=np.array([left, top, right, bottom]) # x1, y1, x2, y2
        results['crop_bbox']=crop_bbox
        results['img_shape']=(new_h, new_w)
        
        if not self.lazy:
            if 'keypoint' in results: results['keypoint']=self._crop_kps(results['keypoint'], crop_bbox)
            if 'imgs' in results: results['imgs']=self._crop_imgs(results['imgs'], crop_bbox)
        else:
            lazyop=results['lazy']
            if 'flip' in lazyop and lazyop['flip']: raise NotImplementedError('Please put flip as the last transformation for now')
            # record crop_bbox in lazyop dict to ensure only crop once in Fuse
            # here we get existing previous crop
            lazy_left, lazy_top, lazy_right, lazy_bottom=lazyop['crop_bbox']
            # below we convert new crop coordinates into the coordinate system of the previous crop
            left=left*(lazy_right-lazy_left)/img_w # scale current left based on previos crop
            right=right*(lazy_right-lazy_left)/img_w
            top=top*(lazy_bottom-lazy_top)/img_h # scale current top based on previos crop
            bottom=bottom*(lazy_bottom-lazy_top)/img_h
            # finally, we shift it into the original image coordinate, i.e,.,
            # offset the rescaled crop by the previous crop's top-left corner to produce a new absolute crop box in the original image coordinates
            lazyop['crop_bbox']=np.array([(lazy_left+left), (lazy_top+top), (lazy_left+right), (lazy_top+bottom)], dtype=np.float32)
        
        if 'gt_bboxes' in results: results=self._all_box_crop(results, results['crop_bbox'])
        
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size}, lazy={self.lazy})"

class RandomCrop:
    """Vanilla square random crop 

    Required keys are 'img_shape', 'keypoint' (optional), 'imgs' (optional), added or modified keys are 'keypoint', 'imgs', 'lazy'; 
    Required keys in 'lazy' are 'flip', 'crop_bbox', added or modified key os 'crop_bbox'

    Args:
        size (int): The output size of the images
        lazy (bool): Whether to apply lazy operation. Default: False
    """
    def __init__(self, size, lazy=False):
        assert isinstance(size, int), f"Size must be an int, but got {type(size)}"
        self.size=size
        self.lazy=lazy

    def __call__(self, results:dict)->Optional[Union[dict, tuple[list,list]]]:
        return self.transform(results)
        
    @staticmethod
    def _crop_kps(kps, crop_bbox):
        """Crop keypoints
        Args:
            kps (np.ndarray): Keypoint probaby of size (M,T,V,2) where 2 is for x, y and M is the number of persons (instances) in the frame
                T is the number of frames, V is the number of keypoints (e.g., 17 for COCO)
            crop_bbox (np.ndarray): 1D bounding box array of x1,y1,x2,y2 in pixel units, where (x1,y1) is the top left corner and
                (x2,y2) is the bottom right corner
        """
        return kps-crop_bbox[:2] # subtract x1, y1

    @staticmethod
    def _crop_imgs(imgs, crop_bbox):
        """Crop each frame
        Args:
            imgs (list[np.ndarray]): List of video frames, each is of size (H,W,C)
            crop_bbox (np.ndarray): 1D bounding box array of x1,y1,x2,y2 in pixel units, where (x1,y1) is the top left corner and (x2,y2) is the
                bottom right corner
        """
        x1,y1,x2,y2=crop_bbox
        return [img[y1:y2,x1:x2] for img in imgs]

    @staticmethod
    def _box_crop(box, crop_bbox):
        """Crop the bounding boxes according to the crop_bbox
        Args:
            box (np.ndarray): The bounding boxes of size (*,4) where * can be N (number of boxes) or more dimensions (such as (B,N) where B is
                the batch size), and 4 is for x1,y1,x2,y2 in pixel units where (x1,y1) is the top left corner and (x2,y2) is the
                bottom right corner
            crop_bbox (np.ndarray): 1D bounding box array of x1,y1,x2,y2 in pixel units, where (x1,y1) is the top left corner and (x2,y2) is the
                bottom right corner
        Returns:
            (np.ndarray): The bounding boxes after cropping with same size as the input box
        """
        x1,y1,x2,y2=crop_bbox
        width, height=x2-x1, y2-y1

        box_=box.copy()
        box_[...,0::2]=np.clip(box[...,0::2]-x1, 0, width-1)
        box_[...,1::2]=np.clip(box[...,1::2]-y1, 0, height-1)
        return box_

    def _all_box_crop(self, results, crop_bbox):
        """Crop the gt_bboxes and proposals in results according to crop_bbox
        Args:
            results (dict): All information about the sample, which contain 'gt_bboxes' and 'proposals' (optional)
            crop_bbox (np.ndarray): 1D bounding box array of x1,y1,x2,y2 in pixel units, where (x1,y1) is the top left corner and (x2,y2) is the
                bottom right corner
        """
        results['gt_bboxes']=self._box_crop(results['gt_bboxes'], crop_bbox)
        if 'proposals' in results and results['proposals'] is not None:
            assert results['proposals'].shape[-1]==4
            results['proposals']=self._box_crop(results['proposals'], crop_bbox)
        return results

    def transform(self, results):
        """ Perform RandomCrop augmentation
        Args:
            results (dict): The resulting dict to be modified and passed to the next transform in pipeline
        """
        _init_lazy_of_proper(results, self.lazy)
        if any(x in results for x in ['gt_bboxes', 'keypoint']):
            assert not self.lazy, f'Kypoint/bounding box augmentation are not compatible with lazy==True'

        img_h, img_w=results['img_shape']
        assert self.size<=img_h and self.size<=img_w

        y_offset=x_offset=0
        if img_h>self.size: y_offset=int(np.random.randint(0, img_h-self.size))
        if img_w>self.size: x_offset=int(np.random.randint(0, img_w-self.size))

        if 'crop_quadruple' not in results: results['crop_quadruple']=np.array([0,0,1,1], dtype=np.float32) # x, y, w, h

        # normalize coordinate of crop coordinates
        x_ratio, y_ratio=x_offset/img_w, y_offset/img_h # normalized top left corner, i.e., normalized x1, y1
        w_ratio, h_ratio=self.size/img_w, self.size/img_h # normalized image size, i.e., normalized w, h

        old_crop_quadruple=results['crop_quadruple']
        old_x_ratio, old_y_ratio=old_crop_quadruple[0], old_crop_quadruple[1]
        old_w_ratio, old_h_ratio=old_crop_quadruple[2], old_crop_quadruple[3]
        # new crop location relative to/in the old crop coordinate system
        new_crop_quadruple=[old_x_ratio+x_ratio*old_w_ratio,
                            old_y_ratio+y_ratio*old_h_ratio,
                            w_ratio*old_w_ratio,
                            h_ratio*old_h_ratio]
        results['crop_quadruple']=np.array(new_crop_quadruple, dtype=np.float32)
        
        new_h, new_w=self.size, self.size # in pixel units
        crop_bbox=np.array([x_offset, y_offset, x_offset+new_w, y_offset+new_h]) # x1, y1, x2, y2 in pixel units
        results['crop_bbox']=crop_bbox
        results['img_shape']=(new_h, new_w)

        if not self.lazy:
            if 'keypoint' in results: results['keypoint']=self._crop_kps(results['keypoint'], crop_bbox)
            if 'imgs' in results: results['imgs']=self._crop_imgs(results['imgs'], crop_bbox)
        else:
            lazyop=results['lazy']
            if 'flip' in lazyop and lazyop['flip']: raise NotImplementedError("Please put Flip as the last transform for now")

            # record crop_bbox in lazyop dict to ensure only crop once in Fuse
            lazy_left, lazy_top, lazy_right, lazy_bottom=lazyop['crop_bbox']
            left=x_offset*(lazy_right-lazy_left)/img_w
            right=(x_offset+new_w)*(lazy_right-lazy_left)/img_w
            top=y_offset*(lazy_bottom-lazy_top)/img_h
            bottom=(y_offset+new_h)*(lazy_bottom-lazy_top)/img_h
            lazyop['crop_bbox']=np.array([(lazy_left+left), (lazy_top+top), (lazy_left+right), (lazy_top+bottom)], dtype=np.float32)
            
        # Process boxes
        if 'gt_bboxes' in results: results=self._all_box_crop(results, results['crop_bbox'])

        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size}, lazy={self.lazy})"

class CenterCrop(RandomCrop):
    """Crop the center area from images

    Required keys are 'img_shape', 'imgs' (optional), 'keypoint' (optional), added or modified keys are 'imgs', 'keypoints', 'crop_bbox',
    'lazy' and 'img_shape'. Required keys in 'lazy' is 'crop_bbox', added or modified key is 'crop_bbox'

    Args:
        crop_size (int | tuple[int]): (width, height) of crop size
        lazy (bool): Whether to apply lazy operation. Default: False
    """

    def __init__(self, crop_size, lazy=False):
        
        self.crop_size=_pair(crop_size)
        self.lazy=lazy

        assert is_tuple_of(self.crop_size, int), f"Crop size must be int or tuple of int, but got {type(crop_size)}"

    def transform(self, results):
        """Perform the center crop augmentation
        Args:
            results (dict): The resulting dict to be modified and passed to the next transform in pipeline
        """
        _init_lazy_of_proper(results, self.lazy)
        
        if any(x in results for x in ['keypoint', 'gt_bboxes']):
            assert not self.lazy, "Keypoint/bounding box augmentations are not compatible with lazy==True"
            
        img_h, img_w=results['img_shape']
        crop_w, crop_h=self.crop_size
        
        left=(img_w-crop_w)//2
        top=(img_h-crop_h)//2
        right=left+crop_w
        bottom=top+crop_h
        new_h, new_w=bottom-top, right-left
        
        crop_bbox=np.array([left, top, right, bottom])
        results['crop_bbox']=crop_bbox
        results['img_shape']=(new_h, new_w)
        
        if 'crop_quadruple' not in results: results['crop_quadruple']=np.array([0,0,1,1], dtype=np.float32) # normalized x, y, w, h
        
        x_ratio, y_ratio=left/img_w, top/img_h
        w_ratio, h_ratio=new_w/img_w, new_h/img_h
        
        old_crop_quadruple=results['crop_quadruple']
        old_x_ratio, old_y_ratio=old_crop_quadruple[0], old_crop_quadruple[1]
        old_w_ratio, old_h_ratio=old_crop_quadruple[2], old_crop_quadruple[3]
        
        new_crop_quadruple=[old_x_ratio+x_ratio*old_w_ratio,
                            old_y_ratio+y_ratio*old_h_ratio,
                            w_ratio*old_w_ratio,
                            h_ratio*old_h_ratio]
        results['crop_quadruple']=np.array(new_crop_quadruple, dtype=np.float32)
        if not self.lazy:
            if 'keypoint' in results: results['keypoint']=self._crop_kps(results['keypoint'], crop_bbox)
            if 'imgs' in results: results['imgs']=self._crop_imgs(results['imgs'], crop_bbox)
        else:
            lazyop=results['lazy']
            if lazyop['flip']: raise NotImplementedError("Please put Flip at last for now")
        
            # record crop_bbox in lazyop dict to ensure only crop once in Fuse
            lazy_left, lazy_top, lazy_right, lazy_bottom=lazyop['crop_bbox']
            left=left*(lazy_right-lazy_left)/img_w
            right-right*(lazy_right-lazy_left)/img_w
            top=top*(lazy_bottom-lazy_top)/img_h
            bottom=bottom*(lazy_bottom-lazy_top)/img_h
            lazyop['crop_bbox']=np.array([(lazy_left+left), (lazy_top+top), (lazy_left+right), (lazy_top+bottom)], dtype=np.float32)
        
        if 'gt_bboxes' in results: results=self._all_box_crop(results, results['crop_bbox'])
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(crop_size={self.crop_size}, lazy={self.lazy})"

class ThreeCrop:
    """Crop images into three crops

    Crop images equally into three crops with equal intervals along the shorter side

    Required keys are 'imgs', 'img_shape', added or modified keys are 'imgs', 'crop_bbox', and 'img_shape'

    Args:
        crop_size (int|tuple[int]): (width, height) of crop size
    """
    def __init__(self, crop_size):
        
        self.crop_size=_pair(crop_size)
        assert is_tuple_of(self.crop_size, int), f"Crop size must be int or tuple of int, but got {type(crop_size)}"

    def __call__(self, results:dict)->Optional[Union[dict, tuple[list,list]]]:
        return self.transform(results)

    def transform(self, results):
        """Perform the ThreeCrop augmentation
        
        Args:
            results (dict): The resulting dict to be modified and passed to the next transform in pipeline
        """
        _init_lazy_of_proper(results, False)
        if any(x in results for x in ['gt_bboxes', 'proposals']): warnings.warn("ThreeCrop cannot process bounding boxes")
        
        imgs=results['imgs']
        img_h, img_w=results['imgs'][0].shape[:2] # (H,W,C)
    
        crop_w, crop_h=self.crop_size
        assert crop_h==img_h or crop_w==img_w, ("Crop width or height must be equal to image width or height, i.e., crop_h==img_h or crop_w==img_w "
                                                f"but got {crop_h=} vs {img_h=} and {crop_w=} vs {img_w}")
        if crop_h==img_h:
            w_step=(img_w-crop_w)//2
            offsets=[(0,0), (2*w_step, 0), (w_step, 0)] # for left, right and middle
        elif crop_w==img_w:
            h_step=(img_h-crop_h)//2
            offsets=[(0,0), (0,2*h_step), (0, h_step)] # for top, bottom, and middle
        
        cropped=[]
        crop_bboxes=[]
        for x_offset, y_offset in offsets:
            bbox=[x_offset, y_offset, x_offset+crop_w, y_offset+crop_h]
            crop=[img[y_offset:(y_offset+crop_h), x_offset:(x_offset+crop_w)] for img in imgs]
            cropped.extend(crop)
            crop_bboxes.extend([bbox for _ in range(len(crop))])
        
        crop_bboxes=np.asarray(crop_bboxes) # (3xlen(imgs), 4)
        results['imgs']=cropped # list of 3xlen(imgs), each is (H,W,C) ndarray
        results['crop_bbox']=crop_bboxes 
        results['img_shape']=results['imgs'][0].shape[:2]
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(crop_size={self.crop_size})"

class Flip:
    """Flip the input images with a probability
    
    Reverse the order of elements in the given images with a specific direction. The shape of the images is preserved, but the elements 
    are reordered

    Required keys are 'img_shape', 'modality', 'imgs' (optional), 'keypoint (optional), added or modified keys are 'imgs', 'keypoint', 'lazy'
    and 'flip_direction'. Required keys in 'lazy' is None, added or modified keys are 'flip' and 'flip_direction'. The Flip augmentation
    should be placed after any cropping/reshaping augmentations, to make sure crop_quadruple is calculated properly.

    Args:
        flip_ratio (float): Probability of implementing flip. Default: 0.5
        direction (str): Flip images horizontally or vertically. Options are 'horizontal' or 'vertical'. Default: 'horizontal'
        flip_label_map (dict[int, int], optional): Transform the label of the flipped image with specific label. Default:None
        left_kp (list[int]): Indices of left keypoints in the order corresponding to right keypoints, used to flip keypoints. Default: None
        right_kp (list[int]): Indices of right keypoints in the order corresponding to left keypoints, used to flip keypoints. Default: None
        lazy (bool): Whether to apply lazy operation. Default: False
    """
    _directions=['horizontal', 'vertical']

    def __init__(self, flip_ratio=0.5, direction='horizontal', flip_label_map=None, left_kp=None, right_kp=None, lazy=False):
        
        assert direction in self._directions, f"Direction {direction} is not supported"
        self.flip_ratio=flip_ratio
        self.direction=direction
        self.flip_label_map=flip_label_map
        self.left_kp=left_kp
        self.right_kp=right_kp
        self.lazy=lazy

    def __call__(self, results:dict)->Optional[Union[dict, tuple[list,list]]]:
        return self.transform(results)

    def _flip_imgs(self, imgs, modality):
        """Utility function for flipping image
        Args:
            imgs (list[np.ndarray]): List of (H,W,C) video frames
            modality (str): Image modalities such as RGB, Flow, etc
        """
        imgs=[imflip(img, self.direction) for img in imgs]
        lt=len(imgs)
        if modality=='Flow': # The 1st frame of each 2 frames is flow-x
            for i in range(0, lt, 2): img[i]=iminvert(img[i]) # only flip x element of flow
        return imgs

    def _flip_kps(self, kps, kpscore, img_width):
        """Utility function for flipping keypoint
        Args:
            kps (np.ndarray): Keypoint probaby of size (M,T,V,2) where 2 is for x, y and M is the number of persons (instances) in the frame
                T is the number of frames, V is the number of keypoints (e.g., 17 for COCO)
            kpscore (np.ndarray): Keypoint probaby of size (M,T,V) where M is the number of persons (instances) in the frame
                T is the number of frames, V is the number of keypoints (e.g., 17 for COCO)
            img_width (int): Width of image
        """
        kp_x=kps[...,0]
        kp_x[kp_x!=0]=img_width-1-kp_x[kp_x!=0]
        new_order=list(range(kps.shape[2])) # I do not know dimension of kps why 2 here
        if all(x is not None for x in [self.left_kp , self.right_kp]):
            for left, right in zip(self.left_kp, self.right_kp):
                new_order[left]=right
                new_order[right]=left
        kps=kps[:,:,new_order]
        if kpscore is not None: kpscore=kpscore[:,:,new_order]
        return kps, kpscore

    @staticmethod
    def _box_flip(box, img_width):
        """Flip the bounding boxes given the width of the image
        Args:
            box (np.ndarray): The bounding boxes with shape (M,T,4) or (M*T, 4) in xyxy format where M is the number of persons or objects(instances)
                and T is the number of frames
            img_width (int): Image width
        """
        box_=box.copy()
        box_[...,0]=img_width-1-box[...,0]
        box_[...,2]=img_width-1-box[...,2]
        return box_

    def transform(self, results):
        """Perform the Flip augmentation
        Args:
            results (dict): The resulting dict to be modified and passed to the next transform in the pipeline
        """
        _init_lazy_of_proper(results, self.lazy)
        if any(x in results for x in ['keypoint', 'gt_bboxes']): 
            assert not self.lazy and self.direction=='horizontal', ("Keypoint/Bounding box augmentation are not compatible with lazy==True"
                                                                    " and only support horizontal flip")
        modality=results['modality']
        if modality=='Flow': assert self.direction=='horizontal'
        
        flip=np.random.rand()<self.flip_ratio
        if not flip: return results
            
        results['flip']=flip
        results['flip_direction']=self.direction
        img_width=results['img_shape'][1]
        
        if self.flip_label_map is not None:
            results['label']=self.flip_label_map.get(results['label'], results['label']) #???
        
        if not self.lazy:
            if 'imgs' in results: results['imgs']=self._flip_imgs(results['imgs'], modality)
            if 'keypoint' in results:
                kp=results['keypoint']
                kpscore=results.get('keypoint_score', None)
                kp, kpscore=self._flip_kps(kp, kpscore, img_width)
                results['keypoint']=kp
                if 'keypoint_score' in results: results['keypoint_score']=kpscore
        else:
            lazyop=results['lazy']
            if lazyop['flip']: assert NotImplementedError('Only support a single flip')
            lazyop['flip']=flip
            lazyop['flip_direction']=self.direction
        
        if 'gt_bboxes' in results:
            results['gt_bboxes']=self._box_flip(results['gt_bboxes'], img_width)
            if 'proposals' in results and results['proposals'] is not None:
                assert results['proposals'].shape[-1]==4
                results['proposals']=self._box_flip(results['proposals'],img_width)
        
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(flip_ratio={self.flip_ratio}, direction={self.direction}, flip_label_map={self.flip_label_map}, lazy={self.lazy})"
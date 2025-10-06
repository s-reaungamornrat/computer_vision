import cv2
import copy
import random
import numpy as np

def instance2mask(image, instance):
    '''
    Args:
        image (np.ndaray): HxWxC image
        instance (Instances)
    Returns:
        mask (np.ndaray): HxWxC image
    '''
    inst=copy.deepcopy(instance)
    inst.convert_bbox(format='xyxy')
    inst.denormalize(*image.shape[:2][::-1])
    segments=inst.segments # MxNx2
    mask=np.zeros(image.shape, np.uint8) 
    for j, segs in enumerate(segments):
        cv2.drawContours(mask,segs[None].astype(np.int32), -1, (random.randint(0,255),80+10*j,100+10*j), cv2.FILLED)
    return mask

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, DWConv
from .blocks import DFL
from computer_vision.yolov11_pose.utils.tal import make_anchors, dist2bbox

class Detect(nn.Module):
    """
    YOLO detection head for object detection models
    This class implements the detection head used in YOLO models for predicting bounding boxes and class probabilities
    Examples:
        Create a detection head for 80 classes
        >>> detect=Detect(nc=80,ch=(256,512,1024))
        >>> x=[torch.randn(1,256,80,80), torch.randn(1,512,40,40),torch.randn(1,1024,20,20)]
        >>> outputs=detect(x)
    """
    dynamic=False # force grid reconstruction
    export=False # export mode
    format=None # export format
    end2end=False # end2end
    max_det=300
    shape=None
    anchors=torch.empty(0)
    strides=torch.empty(0)
    legacy=False # backward compatibility for v3-v9
    xyxy=False # xyxy or xywh output
    def __init__(self,nc:int=80, ch:tuple=()):
        """
        Initialize the YOLO detection layer with the specified number of classes and channels
        Args:
            nc (int): Number of classes
            ch (tuple): Tuple of channel sizes from the backbone feature maps
        """
        super().__init__()
        self.nc=nc # number of classes
        self.nl=len(ch) # number of detection layers
        self.reg_max=16 # DFL channels or DFL number of discrete bins
        self.no=nc+self.reg_max*4 # number of outputs per anchor where bins are for each box dimension
        self.stride=torch.zeros(self.nl) # strides computed during build
        c2,c3=max(16, ch[0]//4, self.reg_max*4), max(ch[0], min(self.nc, 100)) # channels
        self.cv2=nn.ModuleList(
            nn.Sequential(Conv(x,c2,3), Conv(c2,c2,3),nn.Conv2d(c2, 4*self.reg_max, 1)) for x in ch
        )
        self.cv3=(
            nn.ModuleList(nn.Sequential(Conv(x,c3,3), Conv(c3,c3,3), nn.Conv2d(c3, self.nc,1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x,x,3), Conv(x,c3,1)),
                    nn.Sequential(DWConv(c3,c3,3), Conv(c3,c3,1)),
                    nn.Conv2d(c3,self.nc,1),
                )
                for x in ch
            )
        )
        self.dfl=DFL(self.reg_max) if self.reg_max>1 else nn.Identity()
        if self.end2end:
            self.one2one_cv2=copy.deepcopy(self.cv2)
            self.one2one_cv3=copy.deepcopy(self.cv3)
            
    def forward(self, x:list[torch.Tensor])->list[torch.Tensor]|tuple:
        """
        Concatenate and return predicted bounding boxes and class probabilities
        """
        if self.end2end:
            raise NotImplementedError('Please implement me see https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L26')
        for i in range(self.nl):
            x[i]=torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training: # training path
            return x
        y=self._inference(x)
        return y if self.export else (y,x)

    def _inference(self,x:list[torch.Tensor])->torch.Tensor:
        """
        Decode predicted bounding boxes and class probabilities based on multiple-level feature maps
        Args:
            x (list[torch.Tensor]): List of feature maps from different detection layers
        Returns:
            (torch.Tensor): Concatenated tensor of decoded bounding boxes and class probabilities
        """
        # Inference path
        shape=x[0].shape # BCHW
        x_cat=torch.cat([xi.view(shape[0], self.no,-1) for xi in x], 2)
        if self.dynamic or self.shape!=shape:
            self.anchors, self.strides=(x.transpose(0,1) for x in make_anchors(x, self.stride, 0.5))
            self.shape=shape
        box, cls=x_cat.split((self.reg_max*4, self.nc),1)
        dbox=self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0))*self.strides
        return torch.cat((dbox, cls.sigmoid()), 1)

    def decode_bboxes(self, bboxes:torch.Tensor, anchors:torch.Tensors, xywh:bool=True)->torch.Tensor:
        """
        Decode bounding boxes from predictions
        """
        return dist2bbox(bboxes, anchors,xywh=xywh and not self.end2end and not self.xyxy, dim=1)
        
    def bias_init(self):
        """
        Initialize Detect() biases, WARNING: require stride availability
        """
        m=self
        for a,b,s in zip(m.cv2,m.cv3,m.stride):
            a[-1].bias.data[:]=1.0 # box
            b[-1].bias.data[:m.nc]=math.log(5/m.nc/(640/s)**2) # cls (.01 objects, 80 classes, 640 img)
        if self.end2end:
            for a,b,s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):
                a[-1].bias.data[:]=1.0 # box
                b[-1].bias.data[:m.nc]=math.log(5/m.nc/(640/s)**2) # cls (.01 objects, 80 classes, 640 img)

class Pose(Detect):
    """
    YOLO Pose head for keypoints models
    Examples:
        Create a pose detection head
        >>> pose=Pose(nc=80,kpt_shape=(17,3),ch=(256,512,1024))
        >>> x=[torch.randn(1,256,80,80),torch.randn(1,512,40,40),torch.randn(1,1024,20,20)]
        >>> outputs=pose(x)
    """
    def __init__(self,nc:int=80,kpt_shape:tuple=(17,3),ch:tuple=()):
        """
        Initialize YOLO pose detection head
        Args:
            nc (int): Number of classes
            kpt_shape (tuple): Number of keypoints, number of dimension (2 for x,y and 3 for x,y,visibility)
            ch (tuple): Tuple of channel sizes from backbone feature maps
        """
        super().__init__(nc, ch)
        self.kpt_shape=kpt_shape
        self.nk=kpt_shape[0]*kpt_shape[1] # number of total keypoint dimensions
        c4=max(ch[0]//4, self.nk)
        self.cv4=nn.ModuleList(nn.Sequential(Conv(x,c4,3), Conv(c4,c4,3), nn.Conv2d(c4,self.nk,1)) for x in ch)

    def forward(self, x:list[torch.Tensor])->torch.Tensor|tuple:
        """
        Perform forward pass through YOLO model and return prediction
        """
        bs=x[0].shape[0] # batch size
        kpt=torch.cat([self.cv4[i](x[i]).view(bs,self.nk,-1) for i in range(self.nl)],-1) # (bs,17*3,h*w)
        x=Detect.forward(self,x)
        if self.training: return x, kpt
        pred_kpt=self.kpts_decode(bs, kpt)
        return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0],pred_kpt],1),(x[1],kpt))

    def kpts_decode(self,bs:int, kpts:torch.Tensor)->torch.Tensor:
        """
        Decode keypoints from predictions
        """
        ndim=self.kpt_shape[1]
        if self.export: 
            raise NotImplementedError('Please implement me, see https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L319')
        else:
            y=kpts.clone()
            if ndim==3:
                y[:,2::ndim].sigmoid_()
                # for MAC, y[:,2::ndim].=y[:,2::ndim].sigmoid()
            y[:,0::ndim]=(y[:,0::ndim]*2. + (self.anchors[0]-0.5))*self.strides
            y[:,1::ndim]=(y[:,1::ndim]*2. + (self.anchors[1]-0.5))*self.strides
        return y
    
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import re
import yaml
import types
import contextlib

import torch
import torch.nn as nn

from computer_vision.yolov11_pose.utils.ops import make_divisible
from computer_vision.yolov11_pose.nn.modules import Conv, DWConv, C3k2, SPPF, C2PSA, Concat, Pose, Detect
from computer_vision.yolov11_pose.utils.torch_utils import initialize_weights, model_info, intersect_dicts, fuse_conv_and_bn
from computer_vision.yolov11_pose.utils import IterableSimpleNamespace

class DetectionModel(nn.Module):
    """
    YOLO detection model
    Examples:
        >>> model=DetectionModel('yolo11.yaml', ch=3, nc=80)
        >>> results=model.predict(image_tensor)
    """
    def __init__(self, cfg='yolo11n.yaml', ch=3, nc=None, verbose=True):
        """
        Initialize the YOLO detection model 
        Args:
            cfg (str|dict): Model configuration file path or dictionary
            ch (int): Number of input channels
            nc (int, optional): Number of classes
            verbose (bool): Whether to display model information
        """
        super().__init__()
        # Read yaml
        if isinstance(cfg, str): self.yaml_file=cfg
        elif isinstance(cfg, dict): self.yaml=cfg
        else: self.yaml=yaml_model_load(cfg)

        self.yaml['channel']=ch # save channels
        if nc and nc !=self.yaml['nc']: 
            print(f"In nn.tasks.DetectionModel.__init__ input {nc} is not equal to nc in yaml {self.yaml['nc']}->overwrite yaml by input")
            self.yaml['nc']=nc
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.names={i:f'{i}' for i in range(self.yaml['nc'])} # default names dict
        self.inplace=self.yaml.get('inplace', True)
        self.end2end=getattr(self.model[-1], "end2end", False)

        # Build strides
        m=self.model[-1] # Detect
        if isinstance(m, Detect): # includes all Detect subclasses like Segment, Pose, OBB, etc
            s=256 # example image size
            m.inplace=self.inplace
            def _forward(x):
                """Perform a forward pass through the model, handling different Detect subclass types accordingly"""
                if self.end2end: return self.forward(x)['one2many']
                return self.forward(x)[0] if isinstance(m, (Pose)) else self.forward(x)
                
            self.model.eval() # Avoid changing batch statistics until training begins
            m.training=True # Setting it to True to properly return strides
            m.stride=torch.tensor([s/x.shape[-2] for x in _forward(torch.zeros(1,ch,s,s))])
            self.stride=m.stride
            self.model.train() # Set model back to training (default) mode
            m.bias_init() # only run once
            
        # Init weights and biases
        initialize_weights(self)
        if verbose: self.info()
        
    def info(self, detailed=False, verbose=True, imgsz=640):
        """
        Print model information
        Args:
            detailed (bool): If True, print out detailed information about the model
            verbose (bool): If True, print out the model information
            imgsz (int): The size of the image that the model will be trained on
        """
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)

    def fuse(self, verbose=True):
        """
        Fuse `Conv2d` and `BatchNorm2d` layers into a single layer for improved computational efficiency
        Returns:
            (torch.nn.Module): The fused model
        """
        if self.is_fused(): return self
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, "bn"):
                m.conv=fuse_conv_and_bn(m.conv, m.bn) # update conv
                delattr(m, 'bn') # remove batchnorm
                m.forward=m.forward_fuse # update forward
        self.info(verbose=verbose)
        return self

    def is_fused(self, thresh=10):
        """
        Check if the model has less than a certain threshold of BatchNorm layers
        Args:
            thresh (int, optional): The threshold number of BatchNorm layers
        Returns:
            (bool): True if the number of BatchNorm layers in the model is less than the threshold, False otherwise
        """
        bn=tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k) # normalization layers, i.e., BatchNorm2d()
        return sum(isinstance(v, bn) for v in self.modules()) < thresh # True if < `thresh` BatchNorm layers in the model
            
    def forward(self, x, *args, **kwargs):
        """
        Perform forward pass of the model for either training or inference
        If x is a dict, calculates and returns the loss for training. Otherwise, return predictions for inference
        Args:
            x (torch.Tensor | dict): Input tensor for inference or dict with image tensor and labels for training
            *args (Any): Variable length argument list
            **kwargs (Any): Arbitrary keyword arguments
        Returns:
            (torch.Tensor): Loss if x is a dict (training) or network prediction (inference)
        """
        if isinstance(x, dict): # for cases of training and validating while training
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        """
        Perform a forward pass through the network
        Args:
            x (torch.Tensor): The input tensor to the model
            profile (bool): Print the computation time of each layer if True
            visualize (bool): Save the feature maps of the model if True
            augment (bool): Augment image during prediction
            embed (list, optional): A list of feature vectors/embeddings to return
        Returns:
            (torch.Tensor): The last output of the model
        """
        if augment: raise NotImplementedError('Please implement me, see https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py')
        return self._predict_once(x, profile, visualize, embed)

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        """
        Perform a forward pass through the network
        Args:
            x (torch.Tensor): The input tensor to the model
            profile (bool): Print the calculation time of each layer if True
            visualize (bool): Save the feature maps of the model if True
            embed (list, optional): A list of feature vectors/embeddings to return
        Returns:
            (torch.Tensor): The last output of the model
        """
        y, dt, embeddings=[],[],[] # outputs
        embed=frozenset(embed) if embed is not None else {-1}
        max_idx=max(embed)
        for m in self.model:
            if m.f!=-1: # if not from previous layer
                x=y[m.f] if isinstance(m.f, int) else [x if j==-1 else y[j] for j in m.f] # from earlier layers
            if profile:
                raise NotImplementedError('Please implement me, see https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py')
            x=m(x) # run
            y.append(x if m.i in self.save else None) # save output
            if visualize:
                raise NotImplementedError('Please implement me, see https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py')
            if m.i in embed:
                raise NotImplementedError('Please implement me, see https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py')
        #print(f'In nn.tasks._predict_once x {x.shape}')
        return x

    def loss(self, batch, preds=None):
        """
        Compute loss
        Args:
            batch (dict): Batch to compute loss on
            preds (torch.Tensor | list[torch.Tensor], optional): Prediction
        """
        if getattr(self, "criterion", None) is None: self.criterion=self.init_criterion()
        if preds is None: preds=self.forward(batch['img'])
        return self.criterion(preds, batch)

    def load(self, weights, verbose=True):
        """
        Load weights into the model, why making sure to adjust the number of input channels of the first convolutional layer
        Args:
            weights (dict| torch.nn.Module): The pre-trained weights to be loaded
            verbose (bool, optional): Whether to log the transfer progress
        """
        model=weights['model'] if isinstance(weights, dict) else weights # torchvision models are not dicts
        csd=model.float().state_dict() # checkpoint state_dict
        updated_csd=intersect_dicts(csd, self.state_dict()) # intersect
        self.load_state_dict(updated_csd, strict=False)
        len_updated_csd=len(updated_csd)
        first_conv='model.0.conv.weight' # hard-coded to yolo models for now
        # mostly used to boost multi-channel training
        state_dict=self.state_dict()
        if first_conv not in updated_csd and first_conv in state_dict:
            c1, c2, h, w=state_dict[first_conv].shape
            cc1,cc2,ch,cw=csd[first_conv].shape
            if ch==h and cw==w:
                c1,c2=min(c1,cc1), min(c2,cc2)
                state_dict[first_conv][:c1,:c2]=csd[first_conv][:c1,:c2]
                len_updated_csd+=1
        if verbose:
            print(f"Transferred {len_updated_csd}/{len(self.model.state_dict())} items from pretrained weights")


class PoseModel(DetectionModel):
    """
    YOLO Pose model
    This class extends DetectionModel to handle human pose estimation, providing specialized loss computation for keypoint detection 
    and pose estimation
    Examples:
        Initialize a pose model
        >>> model=PoseModel('yolo11n-pose.yaml', ch=3, nc=1, data_kpt_shape=(17,3))
        >>> results=model.predict(image_tensor)
    """
    def __init__(self, cfg="yolo11n-pose.yaml", ch=3, nc=None, data_kpt_shape=(None,None), verbose=True):
        """
        Initialize YOLO pose model
        Args:
            cfg (str|dict): Model configuration file path or dict
            ch (int): Number of input channels
            nc (int, optional): Number of classes
            data_kpt_shape (tuple[int,int]): Shape of keypoint data
            verbose (bool,optional): Whether to display model information
        """
        print(f'In nn.tasks.PoseModel.__init__ type(cfg) {type(cfg)}, cfg {cfg}' )
        if isinstance(cfg, IterableSimpleNamespace): cfg=vars(cfg)
        elif isinstance(cfg, (str, Path)): cfg=yaml_model_load(cfg)
        assert isinstance(cfg, dict), f'cfg must be dict, but got {type(cfg)}'
        if any(data_kpt_shape):
            if isinstance(cfg, IterableSimpleNamespace) and not hasattr(cfg, 'kpt_shape'): cfg.kpt_shape=data_kpt_shape
            elif isinstance(cfg, dict) and ('kpt_shape' not in cfg or list(data_kpt_shape)!=list(cfg["kpt_shape"])):
                print(f'Override model.yaml kpt_shape={cfg["kpt_shape"] if "kpt_shape" in cfg else " "} with kpt_shape={data_kpt_shape}')
                cfg["kpt_shape"]=data_kpt_shape
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """
        Initialize the loss criterion for PoseModel
        """
        #return v8PoseLoss()
        raise NotImplementedError('Please implement me, see https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L551')

def yaml_model_load(path):
    """
    Load model configuration from yaml file
    Args:
        path (str|Path): Path to the yaml file
    Returns:
        (dict): Model dict
    """
    if isinstance(path, str): path=Path(path)
    assert isinstance(path, Path), f'{path} must be path to yaml'
    if isinstance(path, Path):
        assert path.is_file(), f'{path} does not exist'
        with open(path) as f: config=yaml.load(f, Loader=yaml.SafeLoader)
    # guess model scale
    try: config['scale']=re.search(r"yolo(e-)?[v]?\d+([nslmx])", path.stem).group(2)
    except AttributeError: config['scale']=''
    config["yaml_file"] = str(path)
    return config
    
def cfg2task(cfg):
    """
    Guess task from YAML dict
    Args:
        cfg (dict): Configuration as read from YAML
    Returns:
        task (str): Model task type
    """
    m=cfg["head"][-1][-2].lower()
    if m in {'classify', 'classifier','cls','fc'}: return 'classify'
    if 'detect' in m: return 'detect'
    if 'segment' in m: return 'segment'
    if m=='pose': return 'pose'
    if m=='obb': return 'obb'

def parse_model(d, ch, verbose=True):
    """
    Parse a YOLO model.yaml dict into a PyTorch model
    Args:
        d (dict): Model dict
        ch (int): Input channel
        verbose (bool): Whether to print model details
    Returns:
        model (torch.nn.Sequential): PyTorch model
        save (list): Sorted list of output layers
    """
    import ast
    print(f'In nn.tasks.parse_model d {d}')
    legacy=True # backward compatibility for v3-v9
    max_channels=float('inf')
    nc, act, scales=(d.get(x) for x in ('nc','activation', 'scales'))
    if verbose: print(f'In nn.tasks.parse_model nc {nc}, act {act}, scales {scales}')
    depth,width,kpt_shape=(d.get(x, 1.) for x in ('depth_multiple', 'width_multiple', 'kpt_shape'))
    if verbose: print(f'In nn.tasks.parse_model depth {depth}, width {width}, kpt_shape {kpt_shape}')
    scale=d.get('scale')
    if scales:
        if not scale:
            scale=next(iter(scales.keys()))
            print(f'In nn.tasks.parse_model no model scale passed. Assuming scale={scale}.')
            depth,width,max_channels=scales[scale]
    if verbose: print(f'In nn.tasks.parse_model depth {depth}, width {width}, max_channels {max_channels}')
    
    if verbose:print(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch=[ch]
    layers, save, c2=[],[],ch[-1] # layers, savelist, ch out
    base_modules = frozenset({Conv, C3k2, SPPF, C2PSA})
    # modules with 'repeat' arguments
    repeat_modules = frozenset({C3k2, C2PSA})
    for i, (f, n, m, args) in enumerate(d['backbone']+d['head']): # from, number, module, args
        m=(getattr(torch.nn,m[3:]) if 'nn.' in m else getattr(__import__('torchvision').ops,m[16:]) if 'torchvision.ops.' in m else globals()[m])
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j]=locals()[a] if a in locals() else ast.literal_eval(a)
        n=n_=max(round(n*depth),1) if n>1 else n # depth gain
        if m in base_modules:
            c1,c2=ch[f],args[0]
            if c2!=nc: # if c2 not equal the number of class
                c2=make_divisible(min(c2, max_channels)*width, 8)
            args=[c1,c2,*args[1:]]
            if m in repeat_modules:
                args.insert(2,n) # number of repeats
                n=1
            if m is C3k2: 
                print(f'In nn.tasks.parse_model m is C3k2 scale {scale}')
                legacy=False
                if scale in 'mlx': args[3]=True
        elif m is torch.nn.BatchNorm2d: args=[ch[f]]
        elif m is Concat: c2=sum(ch[x] for x in f)
        elif m in frozenset({Pose}):
            args.append([ch[x] for x in f])
            if m in {Pose}: m.legacy=legacy
        else: c2=ch[f]
    
        m_=torch.nn.Sequential(*(m(*args) for _ in range(n))) if n>1 else m(*args) # module
        t=(str(m)[str(m).rfind('.')+1:]).strip("'>") # module type
        m_.np=sum(x.numel() for x in m_.parameters()) # number of parameters
        m_.i, m_.f, m_.type=i, f,t # attach index, `from` index, type
        if verbose: print(f"{i:>3}{f!s:>20}{n_:>3}{m_.np:10.0f}  {t:<45}{args!s:<30}")  # print
        save.extend(x%i for x in ([f] if isinstance(f, int) else f) if x!=-1) # append to savelist
        layers.append(m_)
        if i==0: ch=[]
        ch.append(c2)
        
    return torch.nn.Sequential(*layers), sorted(save)

def load_checkpoint(weight):
    """Load a single model weights

    Args:
        weight (str|Path): Model weight path
    Returns:
        ckpt (dict): Model checkpoint dict
    """
    ckpt=torch.load(weight, map_location="cpu", weights_only=False)

    return ckpt
import yaml
import copy
from pathlib import Path

import torch

from .head import Detect
from .module import parse_model
from .utils import initialize_weights, intersect_dicts


class DetectionModel(torch.nn.Module):
    def __init__(self, cfg='yolo11n.yaml', ch=3, verbose=True):
        """
        Args:
            cfg (str| dict): Model configuration file path or dictionary
            ch (int): Number of input channels
            verbose (bool): Whether to display model information
        """
        super().__init__()
        if isinstance(cfg, str):
            cfg=Path(cfg)
            assert cfg.is_file(), f'{cfg} does not exist'
            with open(cfg) as f: self.yaml=yaml.load(f, Loader=yaml.SafeLoader)
        elif isinstance(cfg, dict): self.yaml=cfg
        else: raise TypeError(f'cfg must be dict/str but got {type(cfg)}')

        self.yaml['channels']=ch
        self.model, self.save=parse_model(d=copy.deepcopy(self.yaml), ch=ch, verbose=False)
        self.names={i:f'{i}' for i in range(self.yaml['nc'])} # default names dict
        self.inplace=self.yaml.get('inplace', True)
        self.end2end=getattr(self.model[-1], 'end2end', False)

        # Build strides
        m=self.model[-1] # detect
        if isinstance(m, Detect):
            s=256 
            m.inplace = self.inplace
            self.model.eval() # Avoid chaning batch statistics until training begins 
            m.training=True # setting it to True to properly return strides
            # Square image so H=W
            m.stride=torch.tensor([s/x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))]) # forward
            self.stride=m.stride
            self.model.train() # Set model bacl to training (default) mode
            m.bias_init() # only run once
        else: self.stride=torch.tensor([32]) # default stride for, i.e., RTDETR

        # Init weights, biases
        initialize_weights(self)
        
    def forward(self, x, *args, **kwargs):
        '''
        Perform forward pass for training or inference. If x is a dict, return the loss for training;
        otherwise, return predictions for inference
        Args:
            x (torch.Tensor | dict): Input tensor for inference or dict with image tensor and labels for training
            *arg (Any): Variable length argument list
            **kwargs (Any): Arbitrary keyword arguments
        Returns:
            torch.Tensor: loss if x is a dict and predictions otherwise
        '''
        if isinstance(x, dict): # for cases of training and validating while training
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)
        
    def loss(self, x, *args, **kwargs):
        pass
        
    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        '''
        Perform a forward pass through the network
        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool): Print the computation time of each layer if True
            visualize (bool): Save the feature maps of the model if True
            augment (bool): Augment image during prediction
            embed (list, optional): A list of feature vectors/embeddings to return
        Returns:
            (torch.Tensor): The last output of the model
        '''
        return self._predict_once(x, embed)
        
    def _predict_once(self, x, embed=None):
        '''
        Perform a forward pass through the network
        Args:
            x (torch.Tensor): The input tensor to the model
            embed (list, optional): A list of feature vectors/embeddings to return
        Returns:
            (torch.Tensor): The last output of the model
        '''
        y=[] # outputs
        embed=frozenset(embed) if embed is not None else {-1}
        max_idx=max(embed)
        # print(f'In BaseModel._predict_once max_idx {max_idx} embed {embed}')
        for m in self.model:
            if m.f!=-1: # if not from previous layer
                x=y[m.f] if isinstance(m.f, int) else [x if j==-1 else y[j] for j in m.f] # from earlier layers
            x=m(x) # run
            y.append(x if m.i in self.save else None)
        return x
        
    def load(self, weights):
        """
        Load weights into the model
        Args:
            weights (dict|torch.nn.Module): The pre-trained weights to be loaded
            verbose (bool, optional): Whether to log the transfer progress
        """
        model=weights['model'] if isinstance(weights, dict) else weights # torchvision models are not dicts
        csd=model.float().state_dict() # checkpoint state_dict as FP32
        updated_csd=intersect_dicts(csd, self.state_dict()) 
        self.load_state_dict(updated_csd)
        first_conv='model.0.conv.weight' # hard-coded to yolo for now
        state_dict=self.state_dict()
        if first_conv not in updated_csd and first_conv in state_dict:
            c1,c2,h,w=state_dict[first_conv].shape
            cc1,cc2,ch,cw=csd[first_conv].shape # checkpoint
            if ch==h and cw==w:
                c1,c2=min(c1,cc1),min(c2,cc2)
                state_dict[first_conv][:c1,:c2]=csd[first_conv][:c1,:c2]
                
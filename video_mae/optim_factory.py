import json
import torch

def get_parameter_groups(model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None):
    """Divide parameters into a set with weight_decay imposed and a set without. The set that will not have weight decay includes those
    that are 1D parameters, bias, scale and those in skip_list
    Args:
        model (nn.Module): Model
        weight_decay (float): Weight decay to be imposed
        skip_list (sequence): Sequence of parameter names that will not be constrained through weight decay
    Returns:
        (list[dict]): List of parameter groups, each group is a dict having format {'weight_decay':float, "param":[], 'lr_scale':float}
    Reference: https://github.com/OpenGVLab/VideoMAEv2/blob/master/optim_factory.py#L56
    """
    parameter_group_names={}
    parameter_group_vars={}
    
    for name, param in model.named_parameters():
        if not param.requires_grad: continue # frozen weights
        if len(param.shape)==1 or name.endswith(".bias") or name.endswith('.scale') or name in skip_list:
            # print(f"shape1, bias, scale, skip: {name=}, {param.shape=}")
            group_name='no_decay'
            this_weight_decay=0.
        else:
            group_name='decay'
            this_weight_decay=weight_decay
        layer_id=None
        if get_num_layer is not None: 
            layer_id=get_num_layer(name)
            group_name=f"layer_{layer_id}_{group_name}"
    
        if group_name not in parameter_group_names:
            scale=1.
            if get_layer_scale is not None: scale=get_layer_scale(layer_id)
    
            parameter_group_names[group_name]={"weight_decay":this_weight_decay, "params":[], 'lr_scale':scale}
            parameter_group_vars[group_name]={"weight_decay":this_weight_decay, "params":[], "lr_scale":scale}
    
        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    print(f"Param group {json.dumps(parameter_group_names, indent=2)}")
    return list(parameter_group_vars.values())

def create_optimizer(args, model, get_num_layer=None, get_layer_scale=None, filter_bias_and_bn=True, skip_list=None):
    """
    Build and return a PyTorch optimizer, supporting layer-wise learning rate scaling and weight decay filtering

    It supports Layer-wise Learning Rate Decay (LLRD) during fine tuning. The core idea is the earlier layers (closer to the input) should have
    a smaller learning rate than later layers (closer to the output) during fine tuning to preserve the foundational features learned during 
    pretraining, while allowing the head to adapt to the new task. LLRD is implemented based on `get_num_layer` and `get_layer_scale` see 
    `get_num_layer_vit` and `LayerDecayValueAssigner` as examples respectively.
    Args:
        args (Namespace/Config): A configuration object containing optimization hyperparameters. Key expected attributes include:
            - args.opt (str): Name of the optimizer (e.g., 'adamw', 'sgd')
            - args.weight_decay (float): Global weight decay coefficient
            - args.lr (float): Base learning rate
            - args.opt_eps (float, optional): Epsilon for numerical stability
            - args.opt_betas (tuple[float], optional): Betas for Adam-based optimizers
        model (nn.Module): The neural network whose parameters will be optimized
        get_num_layer (callable, optional): A function mapping a parameter name to its layer index. Used for applying Layer-wise Learning Rate
            Decay (LLRD)
        get_layer_scale (callable, optional): A function returning the specific learning rate scaling factor for a given layer index
        filter_bias_and_bn (bool): Whether to disable weight decay for all bias parameters and Normalization layer weights.
        skip_list (sequence, optional): A collection of parameter names that should be explicitly excluded from weight decay. If None, the function
            attempts to call model.no_weight_decay() to retrieve this list
    Returns:
        (torch.optim.Optimizer): A configured optimizer with parameters partitioned into specific groups for decay and scaling
    """
    opt_lower=args.opt.lower()
    weight_decay=args.weight_decay
    if weight_decay and filter_bias_and_bn:
        skip=set()
        if skip_list is not None: skip|={skip_list}
        if hasattr(model, 'no_weight_decay'): skip|=model.no_weight_decay()
        parameters=get_parameter_groups(model, weight_decay, skip, get_num_layer, get_layer_scale)
        weight_decay=0.
    else: parameters=model.parameters()
    
    opt_args=dict(lr=args.lr, weight_decay=weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None: opt_args['eps']=args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None: opt_args['betas']=args.opt_betas
    print(f"Optimizer settings: {opt_args}")
    
    opt_split=opt_lower.split('_')
    opt_lower=opt_split[-1]
    if opt_lower=='sgd' or opt_lower=='nesterov':
        opt_args.pop('eps',None)
        optimizer=torch.optim.SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower=='momentum':
        opt_args.pop('eps',None)
        optimizer=torch.optim.SGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower=='adamw':
        optimizer=torch.optim.AdamW(parameters, **opt_args)
    else: raise ValueError(f"{args.opt} is not supported")
    return optimizer

def get_num_layer_vit(param_name, num_max_layer):
    """Part of Layer-wise Learning Rate Decay (LLRD), assigning a layer ID to a parameter name. 
    Note this code is written by Google Gemini"""
    if param_name in ('cls_token', 'mask_token', 'pos_embed'): return 0
    elif param_name.startswith('patch_embed'): return 0
    elif param_name.startswith('blocks'):
        layer_id=int(param_name.split('.')[1]) # e.g., blocks.12.norm1.weight
        return layer_id+1
    else:
        # The head or global norm gets the highest index
        return num_max_layer-1

class LayerDecayValueAssigner:
    """Part of Layer-wise Learning Rate Decay (LLRD), to compute scale applied to learning rate per parameter/layer.  
    Note this code is written by Google Gemini"""
    def __init__(self, decay_rate, num_layers):
        self.decay_rate=decay_rate # e.g., 0.75
        self.num_layers=num_layers
    def get_layer_scale(self, layer_id):
        """ Compute decay factor/scale that used to scale learning rate so that the learning rate decreases as we go towards the input. In other words,
        the factor starts from 1.0 for the last layer and decreases as we move toward the input
            lr_layer=lr_base * factor where factor=`decay_rate`**(num_layer-layer-1)
        """
        return self.decay_rate**(self.num_layers-1-layer_id)
        
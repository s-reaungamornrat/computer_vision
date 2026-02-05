import os
import random

import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
        
def initialize_weights(model):
    """Initialize model weights to random values"""
    for m in model.modules():
        t=type(m)
        if t is nn.Conv3d: pass
        elif t is nn.BatchNorm3d:
            m.eps=1e-3 # from default of 1e-05
            m.momentum=0.03 # from default of 0.1
        elif t in {nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU}:
            m.inplace=True
            
def is_dist_avail_and_initialized():
    if not dist.is_available(): return False
    if not dist.is_initialized(): return False
    return True
    
def reduce_across_processes(val, op=dist.ReduceOp.SUM):
    if not is_dist_avail_and_initialized():
        # nothing to sync, but we still convert to tensor for consistency with the distributed case
        return torch.tensor(val) if not isinstance(val, torch.Tensor) else val
    t=torch.tensor(val, device='cuda')
    dist.barrier()
    dist.all_reduce(t, op=op)
    return t
    
def seed_worker(worker_id:int)->None:
    """Set dataloader worker seed for reproducibility across worker processes"""
    worker_seed=torch.initial_seed()%(2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def init_seeds(seed=0, deterministic=False):
    """Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    Args:
        seed (int, optional): Random seed
        deterministic (bool): Whether to set deterministic algorithm
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # for multi-gpu, exception safe
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True) # warn if deterministic is not possible
        torch.backends.cudnn.deterministic=True
        os.environ['CUBLAS_WORKSPACE_CONFIG']=":4096:8"
        os.environ['PYTHONHASHSEED']=str(seed)
    else: unset_deterministic()

def unset_deterministic():
    """Unset all the configurations applied for deterministic training"""
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic=False
    os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
    os.environ.pop('PYTHONHASHSEED',None)

def setup_for_distributed(is_master):
    """This function disables printing when not in master process"""
    import builtins as __builtin__
    
    builtin_print=__builtin__.print

    def print(*args, **kwargs):
        force=kwargs.pop("force", False)
        if is_master or force: builtin_print(*args, **kwargs)
            
    __builtin__.print=print
    
def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank=int(os.environ["RANK"])
        args.world_size=int(os.environ["WORLD_SIZE"])
        args.gpu=int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank=int(os.environ["SLURM_PROCID"])
        args.gpu=args.rank%torch.cuda.device_count()
    elif hasattr(args,"rank"): pass
    else:
        print("Not using distributed mode")
        args.distributed=False
        return

    args.distributed=True
    torch.cuda.set_device(args.gpu)
    args.dist_backend='nccl'
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size,
                                        rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank==0)

def save_checkpoint(fpath, model, optimizer, lr_scheduler, epoch, best_acc, scaler=None):
    """Save checkpoint 
    Args:
        fpath (str): Path to save file
        model (torch.nn.Module): Deep learning model
        optimizer (torch.optim): Optimizer
        lr_scheduler (torch.optim): Scheduler
        epoch (int): Current training epoch
        best_acc (float): Best top1 accuracy so far
        scaler (torch.amp.GradScaler)
    """
    checkpoint={
        "model":model.state_dict(),
        "optimizer":optimizer.state_dict(),
        "lr_scheduler":lr_scheduler.state_dict(),
        "epoch":epoch,
        "best_acc":best_acc
    }
    if scaler is not None: checkpoint['scaler']=scaler.state_dict()
    torch.save(checkpoint, fpath)


def load_checkpoint(fpath, model, optimizer, lr_scheduler, scaler):
    """Load checkpoint. The state_dicts of model,optimizer, lr_scheduler, and/or scaler will get updated via pass by reference
    so we do not need to return the updated model, optimizer, lr_scheduler, scaler
    Args:
        fpath (str): Path to save file
        model (torch.nn.Module): Deep learning model
        optimizer (torch.optim): Optimizer
        lr_scheduler (torch.optim): Scheduler
        scaler (torch.amp.GradScaler)
    Returns:
        start_epoch (int):  Next taining epoch 
        best_acc (float): Best top1 accuracy so far
    """
    checkpoint=torch.load(fpath, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    if scaler is not None: scaler.load_state_dict(checkpoint['scaler'])
    print(f"Resume training from epoch {checkpoint['epoch']+1} with best_acc at {checkpoint['best_acc']} based on checkpoint {fpath}")
    return checkpoint['epoch']+1, checkpoint['best_acc']
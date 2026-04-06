import math
import time
import random
import numbers
import datetime
from collections import defaultdict, deque

import torch
import numpy as np
from torch.utils.data._utils.collate import default_collate
import torch.distributed as dist

def is_dist_avail_and_initialized(): return dist.is_available() and dist.is_initialized()

def get_world_size():
    return 1 if not is_dist_avail_and_initialized() else dist.get_world_size()
    
def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0, warmup_steps=-1):
    """
    Construct cosine schedule for learninging rate and weight decay
    Args:
        base_value (float): Base learning rate value
        final_value (float): Final learning rate value
        epochs (int): Maximum number of epochs
        niter_per_ep (int): Number of steps/iterations per epoch
        warmup_epochs (int): Number of warm-up epochs
        start_warmup_value (float): Learning rate at the start of warmup period
        warmup_steps (int): Number of warmup iterations/steps
    Returns:
        (np.ndarray): Learning rate schedule in iterations/steps
    Reference: https://github.com/OpenGVLab/VideoMAEv2/blob/master/utils.py#L433
    """
    warmup_schedule=np.array([])
    warmup_iters=warmup_epochs*niter_per_ep
    if warmup_steps>0: warmup_iters=warmup_steps
    print(f"Set warmup steps= {warmup_iters}")
    if warmup_epochs>0: warmup_schedule=np.linspace(start_warmup_value, base_value, warmup_iters)
    iters=np.arange(epochs*niter_per_ep-warmup_iters)
    schedule=np.array([
        final_value+0.5*(base_value-final_value)*
        (1+math.cos(math.pi*i/len(iters))) for i in iters
    ])
    schedule=np.concatenate((warmup_schedule, schedule))
    assert len(schedule)==epochs*niter_per_ep, f"len(schedule)!=epochs*niter_per_ep: {len(schedule)}!={epochs*niter_per_ep}"
    
    return schedule

def seed_worker(worker_id: int) -> None:
    """Set dataloader worker seed for reproducibility across worker processes."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def multiple_samples_collate(batch, fold=False):
    """Collate function for repeated augmentation. Each instance in the batch has more than one sample
    Args:
        batch (tuple | list): Data batch to collate
    Returns:
        (tuple): Collated data batch
    """
    inputs, labels, video_idx, extra_data=zip(*batch)
    inputs=[item for sublist in inputs for item in sublist] # convert list[list[(C,T,H,W) tensors]] to list[(C,T,H,W) tensors]
    labels=[item for sublist in labels for item in sublist]
    video_idx=[item for sublist in video_idx for item in sublist]
    inputs, labels, video_idx, extra_data=(
        default_collate(inputs),
        default_collate(labels),
        default_collate(video_idx),
        default_collate(extra_data),
    )
    if fold: return [inputs], labels, video_idx, extra_data
    return inputs, labels, video_idx, extra_data
    
def multiple_pretrain_samples_collate(batch, fold=False):
    """Collate function for repeated augmentation. Each instance in the batch has more than one sample
    Args:
        batch (tuple|list): Data batch to collate
    Returns:
        (tuple): Collated data batch
    """
    process_data, encoder_mask, decoder_mask=zip(*batch)
    process_data=[item for sublist in process_data for item in sublist]
    encoder_mask=[item for sublist in encoder_mask for item in sublist]
    decoder_mask=[item for sublist in decoder_mask for item in sublist]
    process_data, encoder_mask, decoder_mask=(default_collate(process_data), default_collate(encoder_mask), default_collate(decoder_mask))
    if fold: return [process_data], encoder_mask, decoder_mask
    return process_data, encoder_mask, decoder_mask


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a window or the global series average"""
    def __init__(self, window_size=20, fmt=None):
        if fmt is None: fmt="{median:.4f} ({global_avg:.4f})"
        self.deque=deque(maxlen=window_size)
        self.total=0.
        self.count=0
        self.fmt=fmt
    def update(self, value, n=1):
        self.deque.append(value)
        self.count+=1
        self.total+=value*n
    @property
    def median(self):
        d=torch.tensor(list(self.deque))
        return d.median().item()
    @property
    def avg(self):
        d=torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()
    @property
    def global_avg(self):
        return self.total/self.count
    @property
    def max(self):
        return max(self.deque)
    @property
    def min(self):
        return min(self.deque)
    @property
    def value(self):
        return self.deque[-1]
    def __str__(self):
        return self.fmt.format(median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, min=self.min, value=self.value)

class MetricLogger(object):
    def __init__(self, delimiter='\t'):
        self.meters=defaultdict(SmoothedValue)
        self.delimiter=delimiter
        
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None: continue
            if isinstance(v, torch.Tensor): v=v.item()
            assert isinstance(v, numbers.Number)
            self.meters[k].update(v)

    def __getattr__(self,attr):
        if attr in self.meters: return self.meters[attr]
        if attr in self.__dict__: return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str=[f"{name}:{meter}" for name, meter in self.meters.items()]
        return self.delimiter.join(loss_str)

    def add_meter(self,name, meter): self.meters[name]=meter

    def log_every(self, iterable, print_freq, header=None):
        i=0
        if not header: header=''
        start_time=time.time()
        end=time.time()
        iter_time=SmoothedValue(fmt="{avg:.4f} ({min:.4f} -- {max:.4f})")
        data_time=SmoothedValue(fmt="{avg:.4f} ({min:.4f} -- {max:.4f})")
        space_fmt=":"+str(len(str(len(iterable))))+"d"
        log_msg=[header, '[{0'+space_fmt+ '}/{1}]', 'eta:{eta}', '{meters}', 'time: {time}', 'data: {data}']

        if torch.cuda.is_available(): log_msg.append("max mem: {memory:.0f}")
        log_msg=self.delimiter.join(log_msg)
        MB=1024.0**2
        for obj in iterable:
            data_time.update(time.time()-end)
            yield obj
            iter_time.update(time.time()-end)
            if i%print_freq==0 or i==len(iterable)-1:
                eta_seconds=iter_time.global_avg*(len(iterable)-i) # len(iterable)-i how many iterations left
                eta_string=str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time), 
                                       memory=torch.cuda.max_memory_allocated()/MB)
                    )
                else:
                    print(
                        log_msg.format(i, len(iterable), eta=eta_string, meters=str(self),time=str(iter_time), data=str(data_time))
                    )
            i+=1
            end=time.time()
        total_time=time.time()-start_time
        total_time_str=str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str} ({total_time/len(iterable):.4f} s/it)")


def get_grad_norm(parameters, norm_type:float=2.)->torch.Tensor:
    """Compute gradient norm of parameters
    Args:
        parameters (sequence): Sequence of model parameters
        norm_type (float): Vector norm type
    Return:
        (torch.Tensor): Scalar gradient norm
    """

    if isinstance(parameters, torch.Tensor): parameters=[parameters]
    parameters=[p for p in parameters if p.grad is not None]
    
    norm_type=float(norm_type)
    if len(parameters)==0: return torch.tensor(0.)
    
    if norm_type==float('inf'): return max(p.grad.detach().abs().max() for p in parameters)
    
    return torch.norm(
            torch.stack([
                torch.norm(p.grad.detach(), norm_type) for p in parameters
            ]), norm_type)

def save_checkpoint(fpath, model, optimizer, epoch, best_loss=None, best_acc1=None, scaler=None):
    """Save checkpoint 
    Args:
        fpath (str): Path to save file
        model (torch.nn.Module): Deep learning model
        optimizer (torch.optim): Optimizer
        epoch (int): Current training epoch
        best_loss (float): Best loss so far
        best_acc1 (float): Best top1 accuracy so far
        scaler (torch.amp.GradScaler)
    """
    checkpoint={
        "model":model.state_dict(),
        "optimizer":optimizer.state_dict(),
        "epoch":epoch,
    }
    if best_loss is not None: checkpoint['best_loss']=best_loss
    if best_acc1 is not None: checkpoint['best_acc1']=best_acc1
    if scaler is not None: checkpoint['scaler']=scaler.state_dict()
    torch.save(checkpoint, fpath)

def form_stats(metric_logger, mode='train'):
    """Form a stat dict ready to print to csv as a line that is easily to plot to monitor training
    Args:
        metric_logger (MetricLogger):
        mode (str): Mode of data, including 'train' or 'val'
    """
    assert mode in ('train', 'val')
    stats=dict()
    for k, v in metric_logger.meters.items():
        if k in ('lr', 'min_lr', 'weight_decay' ): v=v.value
        else: v=v.global_avg
        if 'acc' in k: k=f'metric-{k}'
        if any(x in k for x in ['acc', 'loss']): k=f'{mode}/{k}'
        stats[k]=v
    return stats

def load_state_dict(model, state_dict, prefix='', ignore_missing='relative_position_index', verbose=False):
    """Load state dict to model
    Args:
        model (nn.Module): Model
        state_dict (collections.OrderedDict): Pretrained weights
        prefix (str): Prefix of parameter names, often indicates parameter levels in the model parameter tree
        verbose (bool): Whether to print detailed unexpected keys
    """
    missing_keys, unexpected_keys, error_msgs=[],[],[]
    metadata=getattr(state_dict, '_metadata', None)
    state_dict=state_dict.copy() # copy state_dict so _load_from_state_dict can modify it
    if metadata is not None: state_dict._metadata=metadata
    
    def load(module, prefix=''):
        """"Recursively mapp the weights from a flat state_dict onto the hierarchical structure of nn.Module, or bridge the model tree structure and
        the dictionary of weights"""
        # extract `local_metadata` specific to the current module level (using `prefix`)
        local_metadata={} if metadata is None else metadata.get(prefix[:-1], {})
        # torch built-in function https://github.com/pytorch/pytorch/blob/4a8f5e752beb5a6809ba866c83f32dd464a47bfd/torch/nn/modules/module.py#L2333
        # weight injection: `module._load_from_state_dict` looks into the `state_dict`, finds keys that start with the current `prefix`, and copies
        # those tensors into the module's parameters (i.e., weights and biases)
        # missing_keys, unexpected_keys, error_msgs help debug and check whether the checkpoint match the model architecture
        module._load_from_state_dict(state_dict=state_dict, prefix=prefix, local_metadata=local_metadata, strict=True, missing_keys=missing_keys, 
                                     unexpected_keys=unexpected_keys, error_msgs=error_msgs) 
        for name, child in module._modules.items():
            # call `load` recursively on each child since parameter values will be set only `prefix` matches model parameter names, for example
            # `blocks.0.attn.qkv.weight`
            if child is not None: load(child, prefix+name+".") # for each child calculate a new prefix
    
    load(model, prefix=prefix)
    
    warn_missing_keys, ignore_missing_keys=[],[]
    for key in missing_keys:
        keep_flag=True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key: keep_flag=False; break
        if keep_flag: warn_missing_keys.append(key)
        else: ignore_missing_keys.append(key)
    
    missing_keys=warn_missing_keys
    if len(missing_keys)>0: print(f"\nWeights of {model.__class__.__name__} not initialized from pretrained model: {missing_keys}")
    if len(unexpected_keys)>0 and verbose: print(f"\nWeights from pretrained model not used in {model.__class__.__name__}: {unexpected_keys}")
    if len(ignore_missing_keys)>0: print(f"\nIgnored weights of {model.__class__.__name__} not initialized from pretrained model: {ignore_missing_keys}")
    if len(error_msgs)>0: print('\n'.join(error_msgs))

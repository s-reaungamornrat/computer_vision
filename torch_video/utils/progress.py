import time
import datetime
from collections import defaultdict, deque
from pathlib import Path

import torch

from .torch_utils import reduce_across_processes

class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a window or the global series average"""
    def __init__(self, window_size=20, fmt=None):
        if fmt is None: fmt="{median:.4f} ({global_avg:.4f})"

        self.deque=deque(maxlen=window_size)
        self.total=0.
        self.count=0
        self.fmt=fmt
    def update(self, value, n=1):
        self.deque.append(value)
        self.count+=n
        self.total+=value*n
    def synchronize_between_processes(self):
        """Warning: does not synchronize the deque"""
        t=reduce_across_processes([self.count, self.total])
        t=t.tolist()
        self.count=int(t[0])
        self.total=t[1]
    @property
    def median(self):
        d=torch.tensor(list(self.deque), dtype=torch.float32)
        return d.median().item()
    @property
    def global_avg(self): return self.total/self.count

    @property
    def avg(self): return sum(self.deque)/self.deque.maxlen
        
    @property
    def max(self): return max(self.deque)
    
    @property
    def value(self): return self.deque[-1]

    def __str__(self):
        return self.fmt.format(median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value)
        
class MetricLogger:
    def __init__(self, delimiter='\t'):
        self.meters=defaultdict(SmoothedValue)
        self.delimiter=delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor): v=v.item()
            if not isinstance(v, (float, int)): 
                raise TypeError(f"This method expects the value of the input arguments to be of type float or int, but got {type(v)}")
            self.meters[k].update(v)
            
    def __getattr__(self, attr):
        if attr in self.meters: return self.meters[attr]
        if attr in self.__dict__: return self.__dict__[attr]
        raise AttributError(f"'{type(self).__name__}' object has no attribute '{attr}'")
        
    def __str__(self):
        loss_str=[]
        for name, meter in self.meters.items(): loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values(): meter.synchronize_between_processes()

    def add_meter(self, name, meter): self.meters[name]=meter

    def log_every(self, iterable, print_freq, header=None, device=None):
        """Progress logger for loops. It wraps around an iterable (like dataset loader) to provide real-time performance statistics during training or 
        inference. In other words, it passes each object through while tracking the time taken between each step.
        Example:
            >>> for data in logger.log_every(dataloader, 10, 'Training'):
            >>>     # training logics here
        """
        i=0 # batch/iteration index
        if not header: header=''
        start_time=time.time()
        end=time.time()
        iter_time=SmoothedValue(fmt="{avg:.4f}")
        data_time=SmoothedValue(fmt="{avg:.4f}")
        space_fmt=':'+str(len(str(len(iterable)))) + "d" # padding based on the amount of data, e.g., in data loader
        if device.type=='cuda':
            log_msg=self.delimiter.join(
                [
                    header,
                    "[{0"+space_fmt+"}/{1}]", # for current-iteration/total
                    "eta: {eta}", # estimate remaining time for the epoch
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}"
                ]
            )
        else:
            log_msg=self.delimiter.join(
                [header, "[{0"+space_fmt+"}/{1}]", "eta:{eta}", "{meters}", "time:{time}", "data:{data}"]
            )
        MB=1024.*1024.
        for obj in iterable:
            data_time.update(time.time()-end) # time taken to get data for each iteration. 
            yield obj
            iter_time.update(time.time()-end) # time taken for 1 iteration loop
            if i%print_freq==0:
                # estimate time of arrival: how much time left in the epoch based on the avg speed of previous iterations
                eta_seconds=iter_time.global_avg*(len(iterable)-i) 
                eta_string=str(datetime.timedelta(seconds=int(eta_seconds)))
                if device.type=='cuda':
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,meters=str(self), time=str(iter_time), data=str(data_time), 
                        memory=torch.cuda.max_memory_allocated()/MB)
                    )
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time)
                    ))
            i+=1
            end=time.time()
        total_time=time.time()-start_time
        total_time_str=str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str}")

def save_metrics(fpath:str|Path, metrics:dict[str,float|int], epoch:int, start_epoch_time:float):
    """Save metrics and loss to csv file after each epoch
    Args:
        fpath (str|Path): Path to save file
        metrics (dict[str,float|int]): Metric values, e.g., {'loss':0.8, 'acc1':0.6, 'acc5':0.4, ...}
        epoch (int): Current epoch
        start_epoch_time (float): Time at the start of this epoch in seconds
    """
    if isinstance(fpath, str): fpath=Path(fpath)
    fpath.parent.mkdir(parents=True, exist_ok=True)
    
    keys,vals=list(metrics.keys()),list(metrics.values())
    n=len(metrics)+2 # number of columns
    t=time.time()-start_epoch_time # in seconds
    s='' if fpath.exists() else ('%s,' *n %('epoch', 'time', *keys)).rstrip(',')+"\n"
    with open(fpath,'a',encoding='utf-8') as f:
        f.write(s+('%.6g,'*n % (epoch+1, t, *vals)).rstrip(',')+"\n")

def form_stats(metric_logger, mode='train'):
    """Form a stat dict ready to print to csv as a line that is easily to plot to monitor training
    Args:
        metric_logger (MetricLogger):
        mode (str): Mode of data, including 'train' or 'val'
    """
    assert mode in ('train', 'val')
    stats=dict()
    for k, v in metric_logger.meters.items():
        if k in ('lr','clips/s'): v=v.value
        else: v=v.global_avg
        if 'acc' in k: k=f'metric-{k}'
        if any(x in k for x in ['acc', 'loss']): k=f'{mode}/{k}'
        stats[k]=v
    return stats
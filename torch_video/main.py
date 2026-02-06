import time
import warnings
import datetime
from pathlib import Path

import torch
import numpy as np

from computer_vision.torch_video.parameter_parser import parser
from computer_vision.torch_video.utils.train_utils import create_dataloader, setup_training, train
from computer_vision.torch_video.utils.torch_utils import init_seeds, init_distributed_mode, load_checkpoint, clear_memory
from computer_vision.torch_video.utils.plotting import plot_results
from computer_vision.torch_video.data.dataset import NUM_CLASSES

def main(args):
    
    args.start_training_time=time.time() # time at the start of training program
    args.ouput_path=Path(args.ouput_path)
    args.ouput_path.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dirpath=args.ouput_path/"checkpoint"
    args.checkpoint_dirpath.mkdir(parents=True, exist_ok=True)
    
    init_distributed_mode(args)
    
    device=(torch.device(args.device) 
            if (args.device=='cuda' and torch.cuda.is_available() and torch.cuda.device_count()>0) 
            else torch.device('cpu') )
    
    init_seeds(args.seed+1,deterministic=args.use_deterministic_algorithms)
    
    train_loader, val_loader=create_dataloader(args)
    model, optimizer, criterion, lr_scheduler, scaler=setup_training(args, iters_per_epoch=len(train_loader), n_classes=NUM_CLASSES, device=device)
    
    model_without_ddp=model
    if args.distributed:
        model=torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp=model.module
    
    args.start_epoch=0
    best_acc=-np.inf
    if args.resume and (args.checkpoint_dirpath/args.last).is_file():
        args.start_epoch,best_acc=load_checkpoint(args.checkpoint_dirpath/args.last, model_without_ddp, optimizer, lr_scheduler, scaler)
    
    start_time=time.time()
    print("Start training")
    
    train(args, device, model, model_without_ddp, criterion, optimizer, lr_scheduler, train_loader, val_loader, best_acc, train_sampler=None)

    clear_memory()
    
    if (args.ouput_path/"result.csv").is_file: plot_results(args.ouput_path/"result.csv")
        
    total_time=time.time()-start_time
    total_time_str=str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")

if __name__ == "__main__":
    
    args=parser.parse_args()
    
    main(args)
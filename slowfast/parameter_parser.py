import argparse
from mmengine.config import DictAction

parser=argparse.ArgumentParser(description='Train an action recognizer')
parser.add_argument('config', help='train config file path')
parser.add_argument('--work-dir', help='the directory to sabe logs and models')
parser.add_argument('--resume', nargs='?', type=str, const='auto', help='if specify checkpoint path, resume from it, otherwise, specify try to auto resume from the latest checkpoint')
parser.add_argument('--amp', action='store_true', help='enable automatic mix precision training')
parser.add_argument('--no-validate', action='store_true', help='whether not to evaluate the checkpoint during training')
parser.add_argument('--auto-scale-lr', action='store_true', help='whether to auto scale the learning rate according to the actual batch size and the original batch size')
parser.add_argument('--seed', type=int, default=None, help='random seed')
parser.add_argument('--diff-rank-seed', action='store_true', help='whether to set different seeds for different ranks')
parser.add_argument('--deterministic', action='store_true', help='whether to set deterministic options for cudnn backend')
parser.add_argument('--cfg-options', nargs='+', action=DictAction, help=('override some settings in the used config, the key-value pairs'
                                                                         ' in xxx=yyy format will be merged into config file. If the value to '
                                                                         ' overwritten is a list, it should be like key="[a,b]" or key=a,b '
                                                                         'It also allows nested list/tuple values, e.g., key="[(a,b),(c,d)]" '
                                                                         'Note that the quotation marks are nescessary and that no white space '
                                                                         ' is allowed') )
parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
parser.add_argument('--local_rank', '--local-rank', type=int, default=0)

parser.add_argument('-a', '--ann-file', type=str, default=None, help='path to annotation file')
parser.add_argument('-d', '--data-root', type=str, default=None, help='directory contain data')
parser.add_argument('--data-prefix', type=str, default=None, help='relative/sub directories to data')

def merge_args(cfg, args):
    """Merge CLI arguments to config"""
    for key, val in cfg.items():
        if hasattr(args, key): continue
        setattr(args, key, val)
            
    if args.resume=='auto': 
        args.load_from=None
        args.resume=True
    elif isinstance(args.resume,str): 
        args.load_from=args.resume
        args.resume=True
        
    if cfg.get('randomness', None) is None:
        args.randomness=dict(seed=args.seed, diff_rank_seed=args.diff_rank_seed, deterministic=args.deterministic)
    
    if args.no_validate: args.val_cfg=args.val_dataloader=args.val_evaluator=None
    
    # enable automatic mixed precision training
    if args.amp:
        optim_wrapper=cfg.optim_wrapper.get('type', 'OptimWrapper')
        assert optim_wrapper in ['OptimWrapper', 'AmpOptimWrapper'], f"--amp is not supported custom optimizer wrapper type f{optim_wrapper}"
        args.optim_wrapper.type='AmpOptimWrapper'
        args.optim_wrapper.setdefault('loss_scale', 'dynamic')

    return args
    # assert args.work_dir is not None, "Please provide work-dir"
    
    # if args.no_validate: cfg.val_cfg=cfg.val_dataloader=cfg.val_evaluator=None
    # cfg.launcher=args.launcher
    # cfg.work_dir=args.work_dir

    # # enable automatic mixed precision training
    # if args.amp:
    #     optim_wrapper=cfg.optim_wrapper.get('type', 'OptimWrapper')
    #     assert optim_wrapper in ['OptimWrapper', 'AmpOptimWrapper'], f"--amp is not supported custom optimizer wrapper type f{optim_wrapper}"
    #     cfg.optim_wrapper.type='AmpOptimWrapper'
    #     cfg.optim_wrapper.setdefault('loss_scale', 'dynamic')

    # # resume training
    # if args.resume=='auto':
    #     cfg.resume=True
    #     cfg.load_from=None
    # elif args.resume is not None:
    #     cfg.resume=True
    #     cfg.load_from=args.resume

    # # enable auto scale learning rate
    # if args.auto_scale_lr:
    #     if 'auto_scale_lr' in cfg: cfg.auto_scale_lr.enable=True
    #     else: cfg.auto_scale_lr=dict(enable=True)# , base_batch_size=N)

    # # set random seed
    # if cfg.get('randomness', None) is None:
    #     cfg.randomness=dict(seed=args.seed, diff_rank_seed=args.diff_rank_seed, deterministic=args.deterministic)

    # return cfg
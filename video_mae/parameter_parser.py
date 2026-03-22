import argparse

parser=argparse.ArgumentParser('VideoMAE v2 pretraining script')
parser.add_argument('--batch-size', default=16, type=int)
parser.add_argument('--epochs', default=300, type=int)
parser.add_argument('--print_freq', default=20, type=int, help='how often in steps to print training progress')
parser.add_argument('--plot_freq', default=10, type=int, help='how often in epochs to plot training progress')
parser.add_argument('--n_steps', default=None, type=int, help='for debugging and developing code only')
parser.add_argument('--n_epochs', default=None, type=int, help='for debugging and developing code only')
parser.add_argument('--time', default=None, type=float, help='hours to train the model before terminate the training')

# Model parameters
parser.add_argument('--model', default='pretrain_videomae_tiny_patch16_224', type=str, metavar='MODEL', help='name of model to train')
parser.add_argument('--tubelet_size', type=int, default=2)
parser.add_argument('--with_checkpoint', action='store_true', default=False)
parser.add_argument('--cos_attn', action='store_true', default=False)
parser.add_argument('--decoder_depth', default=4, type=int, help='depth of decoder')
parser.add_argument('--mask_type', default='tube', type=str, choices=['random', 'tube'], help='encoder masked strategy')
parser.add_argument('--decoder_mask_type', default='run_cell', choices=['run_cell', 'random'], type=str, help='decoder masked strategy')
parser.add_argument('--mask_ratio', default=0.9, type=float, help='mask ratio of encoder')
parser.add_argument('--decoder_mask_ratio', default=0.5, type=float, help='mask ratio of decoder')
parser.add_argument('--input_size', default=224, type=int, help='images input size for backbone')
parser.add_argument('--drop_path', type=float, default=0., metavar='PCT', help='Drop path rate (default: 0.1)')
parser.add_argument('--normalize_target', default=True, type=bool, help='normalized the target patch pixels')

# Optimizer parameters
parser.add_argument('--opt', default='adamw', type=str, metavar="OPTIMIZER", help='Optimizer (default: "adamw")')
parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer epsilon (default: 1e-8)')
parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA', help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM', help='clip gradient norm (default: None, no clipping)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default:0.9)')
parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
parser.add_argument('--weight_decay_end', type=float, default=None, help=('final value of the weight decay. We use cosine schedule for weight decay.'
                                                                          'Set the same value to args.weight_decay for a fixed/constant weight decay' ) )
parser.add_argument('--lr', type=float, default=1.5e-4, metavar='LR', help='learning rate (default: 1.5e-4)')
parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR', help='warm up learning rate (default: 1e-6)')
parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0 (default: 1e-5)')
parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N', help='epochs to warmup LR if scheduler supports')
parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N', help='iterations to warmup LR if scheduler supports')

# Augmentation parameters
parser.add_argument('--color_jitter', type=float, default=0., metavar='PCT', help='color jitter factor (default: 0.4)')
parser.add_argument('--train-interpolation', type=str, default='bicubic', choices=['random', 'bilinear', 'bicubic'], help='training interpolation')

# Finetuning parameters
parser.add_argument('--finetune', default='', help='finetune from checkpoint')

# Dataset parameters
parser.add_argument('--data_path', default='your/data/annotation/path', type=str, help='dataset path')
parser.add_argument('--data_root', default='', type=str, help='dataset path root')
parser.add_argument('--fname_tmpl', default='img_{:05}.jpg', type=str, help='filename template for rawframe data')
parser.add_argument('--imgenet_default_mean_and_std', default=True, action='store_true')
parser.add_argument('--num_frames', type=int, default=16)
parser.add_argument('--sampling_rate', type=int, default=4)
parser.add_argument('--num_sample', type=int, default=1)
parser.add_argument('--output_dir', default='', help='path to save, empty for no saving')
parser.add_argument('--log_dir', default=None, help='path to save tensorboard log')
parser.add_argument('--device', default='cuda', help='device to use for training and testing')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
parser.add_argument('--auto_resume', action='store_true')
parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
parser.set_defaults(auto_resume=True)
parser.add_argument('--last', default='last.pth', type=str, help='name of latest checkpoint file')
parser.add_argument('--best', default='best.pth', type=str, help='name of best checkpoint file')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
parser.add_argument('--num_workers', default=1, type=int)
parser.add_argument('--pin_mem', action='store_true', help='pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU')
parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
parser.set_defaults(pin_mem=True)

# Distributed training parameters
parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
parser.add_argument('--local_rank', default=-1, type=int)
parser.add_argument('--dist_on_itp', action='store_true')
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
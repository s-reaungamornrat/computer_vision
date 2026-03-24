import argparse

parser=argparse.ArgumentParser('VideoMAE fine-tuning and evaluation script for action classification', add_help=False)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--epochs', default=30, type=int)
parser.add_argument('--update_freq', default=1, type=int)
parser.add_argument('--save_ckpt_freq', default=100, type=int)
parser.add_argument('--print_freq', default=20, type=int, help='how often in steps to print training progress')
parser.add_argument('--plot_freq', default=10, type=int, help='how often in epochs to plot training progress')
parser.add_argument('--n_steps', default=None, type=int, help='for debugging and developing code only')
parser.add_argument('--n_epochs', default=None, type=int, help='for debugging and developing code only')
parser.add_argument('--time', default=None, type=float, help='hours to train the model before terminate the training')

# model parameters
parser.add_argument('--model', default='vit_base_patch16_224', type=str, metavar='MODEL', help='Name of model to train')
parser.add_argument('--tubelet_size', default=2, type=int)
parser.add_argument('--input_size', default=224, type=int, help='input image size')
parser.add_argument('--with_checkpoint', action='store_true', default=False)
parser.add_argument('--drop', type=float, default=0., metavar='PCT', help='Dropout rate (default:0.)')
parser.add_argument('--attn_drop_rate', type=float, default=0., metavar='PCT', help='Attention dropout rate (default:0.)')
parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT', help='Drop path rate (default:0.1)')
parser.add_argument('--head_drop_rate', type=float, default=0., metavar='PCT', help='cls head dropout rate (default:0.)')
parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)
parser.add_argument('--model_ema', action='store_true', default=False)
parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

# optimizer parameters
parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER', help='optimizer (default: "adamw")')
parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON', help='optimizer epsilon (default:1e-8)')
parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='optimizer betas (default: None, use opt default)')
parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM', help='clip gradient norm (default: None, no clipping)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default:0.9)')
parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
parser.add_argument('--weight_decay_end', type=float, default=None, help=('final value of the weight decay. We use cosine schedule for WD '
                                                                          'and using a larger decay by the end of training improves ' 
                                                                          'performance for ViTs'))
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate (default: 1e-3)')
parser.add_argument('--layer_decay', type=float, default=0.75)
parser.add_argument('--warmup_lr', type=float, default=1e-8, metavar='LR', help='warmup learning rate (default:1e-8)')
parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N', help='epochs to warmup LR if scheduler supports')
parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N', help='number of steps to warmup LR, will overard warmup_epochs if set >0')
# augmentation parameters
parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT', help='color jitter factor (default: 0.4)')
parser.add_argument('--num_sample', type=int, default=2, help='repeated_aug (default:2)')
parser.add_argument('--aa', type=str, default='rand-m7-n4-mstd0.5-inc1', metavar='NAME', help=('use autoaugment policy. "v0" or "original"'
                                                                                               ' (default: rand-m7-n4-mstd0.5-inc1)'))
parser.add_argument('--smoothing', type=float, default=0.1, help='label smoothing (default:0.1)')
parser.add_argument('--train_interpolation', type=str, default='bicubic', choices=['random', 'bilinear', 'bicubic'],
                    help='training interpolation (default: bicubic)')
# evaluation parameters
parser.add_argument('--crop_pct', type=float, default=None)
parser.add_argument('--short_side_size', type=int, default=224)
parser.add_argument('--test_num_segment', type=int, default=10)
parser.add_argument('--test_num_crop', type=int, default=3)

# random erase parameters
parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT', help='random erase probability (default: 0.25)')
parser.add_argument('--remode', type=str, default='pixel', help='random erase mode (default: "pixel")')
parser.add_argument('--recount', type=int, default=1, help='random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False, help='do not random erase first (clean) augmentation split')

# mixup parameters
parser.add_argument('--mixup', type=float, default=0.8, help='mixup alpha, mixup enabled if >0.')
parser.add_argument('--cutmix', default=1.0, type=float, help='cutmix alpha, cutmix enabled if >0.')
parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None, help='cutmix min/max ratio, overrides alpha and enables cutmix if set')
parser.add_argument('--mixup_prob', type=float, default=1., help='probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup_mode', type=str, default='batch',  choices=['batch', 'pair', 'elem'], 
                    help='how to apply mixup/cutmix parameters.')

# finetuning parameters
parser.add_argument('--finetune', default='', help='finetune from checkpoint')
parser.add_argument('--model_key', default='model|module', type=str)
parser.add_argument('--model_prefix', default='', type=str)
parser.add_argument('--init_scale', default=0.001, type=float)
parser.add_argument('--use_mean_pooling', action='store_true')
parser.set_defaults(use_mean_pooling=True)
parser.add_argument('--use_cls', action='store_true', dest='use_mean_pooling')

# dataset parameters
parser.add_argument('--train_data_path', default='/your/data/path', type=str, help='dataset path')
parser.add_argument('--val_data_path', default='/your/data/path', type=str, help='dataset path')
parser.add_argument('--data_root', default='', type=str, help='dataset path root')
parser.add_argument('--eval_data_path', default=None, type=str, help='dataste path for evaluation')
parser.add_argument('--nb_classes', default=101, type=int, help='number of the classification types')
parser.add_argument('--imagenet_default_mean_and_std', action='store_true', default=True)
parser.add_argument('--num_segments', type=int, default=1)
parser.add_argument('--num_frames', type=int, default=16)
parser.add_argument('--sampling_rate', type=int, default=4)
parser.add_argument('--sparse_sample', default=False, action='store_true')
parser.add_argument('--data_set', default='UCF101', help='dataset')
parser.add_argument('--fname_tmpl', default='img_{:05}.jpg', type=str, help='filename template for rawframe datasets')
parser.add_argument('--start_idx', default=1, type=int, help='start_idx for rawframe dataset')
parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
parser.add_argument('--log_dir', default=None, help='path where to sabe tensorboard log')
parser.add_argument('--device', default='cuda', type=str, help='device to use for training/testing')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--auto_resume', action='store_true')
parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
parser.set_defaults(auto_resume=True)
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
parser.add_argument('--eval', action='store_true', help='perform evaluation only')
parser.add_argument('--validation', action='store_true', help='perform validation only')
parser.add_argument('--dist_eval', action='store_true', default=False, help='enabling distributed evaluation')
parser.add_argument('--num_workers', default=0, type=int)
parser.add_argument('--pin_mem', action='store_true', help='pin cpu memory in dataloader for more efficient (sometimes) transfer to gpu')
parser.add_argument('--world_size', default=1, type=int, help='number of distrinited processes')
parser.add_argument('--local_rank', default=-1, type=int)
parser.add_argument('--dist_on_itp', action='store_true')
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
parser.add_argument('--enable_deepspeed', action='store_true', default=False)

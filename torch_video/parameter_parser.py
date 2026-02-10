import argparse

parser=argparse.ArgumentParser(description="Video classification training")
parser.add_argument("-d", "--data-path", type=str, default="UCF-101", 
                    help="path to the main video folder containing subfolders of videos from the same category")
parser.add_argument("-a", "--annotation-path", type=str, default="UCF101TrainTestSplits-RecognitionTask", 
                    help="path to annotation folder")
parser.add_argument("-m", "--metadata-path", type=str, default="metadata.pt", 
                    help="path to video metadata including pts for each clip, fps, etc")
parser.add_argument("-o", "--ouput-path", type=str, default=None, help="path to output folder")
parser.add_argument("--data-fold", type=int, default=1, choices=[1,2,3], help="train and test data fold to use")
parser.add_argument("--frame-rate", type=float, default=8, help="desired video frame rate")
parser.add_argument("--clip-duration", type=float, default=2, help="length of extracted video clips in seconds")
parser.add_argument("--step-duration", type=float, default=1.7, help="step between each video clips in seconds")
parser.add_argument("--max-n-clips-per-video", type=int, default=5, help="maximum number of clips per video")
# Augmentation
parser.add_argument("--val-resize-size", default=(180,180), nargs="+", type=int, help="the resize size for validation (default:(112, 112))")
parser.add_argument("--val-crop-size", default=(226, 226), nargs="+", type=int, help="the central crop size for validation (default:(226, 226))")
parser.add_argument("--train-resize-size", default=(180,180), nargs="+", type=int, help="the resize size for training (default:(112, 112))")
parser.add_argument("--train-crop-size", default=(226, 226), nargs="+", type=int, help="the random crop size for training (default:(226, 226))")
parser.add_argument("--hflip-prob", default=0.5, type=float, help="probability of flipping training data horizontally")
# for CutMix and Mixup
# 1  (uniform) for equally-likely blending factors for all factors (e.g., 50/50, 10/90, etc)
# <1 (U-shaped) blending using mostly one or the other (e.g., 95/5). Safer option for small datasets
# >1 (bell-shaped) highly blending factors will be 50/50 resulting in messy images
parser.add_argument("--cutmix-alpha", default=1., type=float, help="controlling blending factor of cutmix")
parser.add_argument("--mixup-alpha", default=1., type=float, help="controlling blending factor of mixup")
parser.add_argument('--use-cutmix-mixup', action='store_true', help='whether to use cutmix and mixup augmentation')

# Model
parser.add_argument('--model', type=str, default='r2plus1d_18', help='model name')

# Training
parser.add_argument('--device', type=str, default='cuda', choices=['cpu','cuda'],help='computing device')
parser.add_argument('--use-deterministic-algorithms', action='store_true', help='force the use of deterministic algorithms only')
parser.add_argument('--seed', type=int, default=0,help='random seed used to initialize random generators')
parser.add_argument('--sync-bn', dest='sync_bn', help='use sync batch norm', action='store_true')
parser.add_argument('--batch-size', type=int, default=24, help='clips per GPU, the total batch size is $NGPU x batch_size')
parser.add_argument('--num-workers', type=int, default=0, help='number of data loading workers. VideoDecoder does not work with num_workers>0')
parser.add_argument('--lr', default=0.0005, type=float, help='initial learning rate. In torchvision.references, this was set to 0.64')
parser.add_argument('--lrf', default=0.01, type=float, help='final LR fraction; final LR = lr0 * lrf')
parser.add_argument('--momentum', default=0.85, type=float, help='momentum', metavar='M')
parser.add_argument("--wd", "--weight-decay", default=0.0005, type=float, metavar='W', help='weight decay (default:1e-4)', dest='weight_decay')
# parser.add_argument('--lr-milestones', nargs='+', default=[20,30,40], type=int, help='decreae lr on milestones')
# parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
parser.add_argument('--warmup-epochs', default=8, type=int, help='the number of epochs to warmup (default:10)')
parser.add_argument('--warmup_momentum', default=0.8, type=float, help='initial momentum during warmup')
parser.add_argument('--warmup_bias_lr', default=0.1, type=float, help='bias learning rate during warmup')
parser.add_argument('--nbs', default=64, type=int, help='nominal batch size used for loss normalization')

# parser.add_argument('--lr-warmup-method',default='linear', choices=['linear', 'constant'], type=str, help='the warm up method (default: linear)')
# parser.add_argument('--lr-warmup-decay', default=0.001, type=float, help='decay for lr')
parser.add_argument('--print-freq', default=100, type=int, help='print frequency in batch-number/iteration')
parser.add_argument('--plot-freq', default=10, type=int, help='plot frequency in epoch')
parser.add_argument('--resume', default=True, type=bool, help='whether to resume training')
parser.add_argument('--last', default='last.pth', type=str, help='name of latest model checkpoint')
parser.add_argument('--best', default='best.pth', type=str, help='name of best model checkpoint')
parser.add_argument('--epochs', default=200, type=int, help='maximum training epoch')
parser.add_argument('--time', default=None, type=float, help='hours to train the model before terminate the training')
parser.add_argument('--n-batches', default=None, type=int, help='number of batches to run before stopping--for debugging purposes')
# Mixed precision training parameters
parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")
parser.add_argument('--num-ffmpeg-threads', default=0, type=int, help='number of threads. default 0 for letting FFmpeg decide on the number of threads')

# distributed training
parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
parser.add_argument("--dist-url", default="env://", type=str, help='url used to set up distributed training')
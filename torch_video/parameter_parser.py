import argparse

parser=argparse.ArgumentParser(description="Video classification training")
parser.add_argument("-d", "--data-path", type=str, default="UCF-101", 
                    help="path to the main video folder containing subfolders of videos from the same category")
parser.add_argument("-a", "--annotation-path", type=str, default="UCF101TrainTestSplits-RecognitionTask", 
                    help="path to annotation folder")
parser.add_argument("-m", "--metadata-path", type=str, default="metadata.pt", 
                    help="path to video metadata including pts for each clip, fps, etc")
parser.add_argument("-o", "--ouput-path", type=str, default=None, help="path to output folder")
parser.add_argument("-c","--class-id", type=str, default="classInd.txt", help="path to a txt file listing pairs of class indices and class names")
parser.add_argument("--data-fold", type=int, default=1, choices=[1,2,3], help="train and test data fold to use")
parser.add_argument("--frame-rate", type=float, default=8, help="desired video frame rate")
parser.add_argument("--clip-duration", type=float, default=2, help="length of extracted video clips in seconds")
parser.add_argument("--step-duration", type=float, default=1.7, help="step between each video clips in seconds")
parser.add_argument("--max-n-clips-per-video", type=int, default=5, help="maximum number of clips per video")
# Augmentation
parser.add_argument("--val-resize-size", default=(112,112), nargs="+", type=int, help="the resize size for validation (default:(128, 171))")
parser.add_argument("--val-crop-size", default=(240, 240), nargs="+", type=int, help="the central crop size for validation (default:(112, 112))")
parser.add_argument("--train-resize-size", default=(112,112), nargs="+", type=int, help="the resize size for training (default:(128, 171))")
parser.add_argument("--train-crop-size", default=(240, 240), nargs="+", type=int, help="the random crop size for training (default:(128, 171))")
parser.add_argument("--hflip-prob", default=0.5, type=float, help="probability of flipping training data horizontally")
# for CutMix and Mixup
# 1  (uniform) for equally-likely blending factors for all factors (e.g., 50/50, 10/90, etc)
# <1 (U-shaped) blending using mostly one or the other (e.g., 95/5). Safer option for small datasets
# >1 (bell-shaped) highly blending factors will be 50/50 resulting in messy images
parser.add_argument("--cutmix-alpha", default=1., type=float, help="controlling blending factor of cutmix")
parser.add_argument("--mixup-alpha", default=1., type=float, help="controlling blending factor of mixup")
parser.add_argument('--use-cutmix-mixup', action='store_true', help='whether to use cutmix and mixup augmentation')

# Training
parser.add_argument('--device', type=str, default='cuda', choices=['cpu','cuda'],help='computing device')
parser.add_argument('--use-deterministic-algorithms', action='store_true', help='force the use of deterministic algorithms only')
parser.add_argument('--seed', type=int, default=0,help='random seed used to initialize random generators')
parser.add_argument('--batch-size', type=int, default=24, help='clips per GPU, the total batch size is $NGPU x batch_size')
parser.add_argument('--num-workers', type=int, default=0, help='number of data loading workers. VideoDecoder does not work with num_workers>0')

# distributed training
parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
parser.add_argument("--dist-url", default="env://", type=str, help='url used to set up distributed training')
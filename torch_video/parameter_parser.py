import argparse

parser=argparse.ArgumentParser(description="Video classification training")
parser.add_argument("-d", "--data-path", type=str, default="UCF-101", 
                    help="path to the main video folder containing subfolders of videos from the same category")
parser.add_argument("-a", "--annotation-path", type=str, default="UCF101TrainTestSplits-RecognitionTask", 
                    help="path to annotation folder")
parser.add_argument("-m", "--metadata-path", type=str, default="metadata.pt", 
                    help="path to video metadata including pts for each clip, fps, etc")
parser.add_argument("-c","--class-id", type=str, default="classInd.txt", help="path to a txt file listing pairs of class indices and class names")
parser.add_argument("--data-fold", type=int, default=1, choices=[1,2,3], help="train and test data fold to use")
parser.add_argument("--frame-rate", type=float, default=8, help="desired video frame rate")
parser.add_argument("--clip-duration", type=float, default=2, help="length of extracted video clips in seconds")
parser.add_argument("--step-duration", type=float, default=1.7, help="step between each video clips in seconds")

parser.add_argument("--val-resize-size", default=(112,112), nargs="+", type=int, help="the resize size for validation (default:(128, 171))")
parser.add_argument("--val-crop-size", default=(240, 240), nargs="+", type=int, help="the central crop size for validation (default:(112, 112))")
parser.add_argument("--train-resize-size", default=(112,112), nargs="+", type=int, help="the resize size for training (default:(128, 171))")
parser.add_argument("--train-crop-size", default=(240, 240), nargs="+", type=int, help="the random crop size for training (default:(128, 171))")
parser.add_argument("--hflip-prob", default=0.5, type=float, help="probability of flipping training data horizontally")
# For CutMix and Mixup
# 1  (uniform) for equally-likely blending factors for all factors (e.g., 50/50, 10/90, etc)
# <1 (U-shaped) blending using mostly one or the other (e.g., 95/5). Safer option for small datasets
# >1 (bell-shaped) highly blending factors will be 50/50 resulting in messy images
parser.add_argument("--cutmix-alpha", default=1., type=float, help="controlling blending factor of cutmix")
parser.add_argument("--mixup-alpha", default=1., type=float, help="controlling blending factor of mixup")
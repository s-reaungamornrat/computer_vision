import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--root', type=str, default=None, help='path to data directory')
parser.add_argument('--image-dirname', type=str, default='images', help='name of image directory')
parser.add_argument('--label-dirname', type=str, default='labels', help='name of label directory')
parser.add_argument('--hyperparam', type=str, default='../default.yaml', help='hyperparameter settings in .yaml file')
parser.add_argument('--data-cfg', type=str, default='../coco128.yaml', help='data configuration in .yaml file')
parser.add_argument('--imgsz', type=int, default=640, help='input image size., i.e., squared images')


# args=parser.parse_args(argument.split())
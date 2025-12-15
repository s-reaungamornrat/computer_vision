import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--save-dir', type=str, default=None, help='path to save prediction')
parser.add_argument('--imgsz', type=int, default=640, help='input image size')
parser.add_argument('--batch-size', type=int, default=16, help='batch size')
parser.add_argument('--deterministic', action='store_true', help='whether to use deterministic training')
parser.add_argument('--data', type=str, default='coco8-pose.yaml', help='data configuration file')
# args=parser.parse_args(argument.split())
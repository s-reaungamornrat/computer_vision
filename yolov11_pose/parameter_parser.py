import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--save-dir', type=str, default=None, help='path to save prediction')
parser.add_argument('--imgsz', type=int, default=640, help='input image size')

# args=parser.parse_args(argument.split())
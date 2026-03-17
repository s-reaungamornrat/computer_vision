# Loading data fast from remote server, see `fsspec + s3fs` and `MosaicML Streaming` 
# pip install fsspec s3fs
# pip install mosaicml-streaming
# Reference https://github.com/OpenGVLab/VideoMAEv2/blob/master/dataset/loader.py#L32

import io
import cv2
from decord import VideoReader, cpu

try: 
    import fsspec
    fsspec_imported=True
except (ImportError, ModuleNotFoundError): fsspec_imported=False

def get_video_loader():
    """ A universal video loader that works with local paths anc cloud URIs (s3://, gs://, etc) using fsspec"""
    # fsspec handles s3://, gs://, and local paths automatically
    def _loader(video_path):
        if fsspec_imported and '://' in video_path: # open remote file as a byte system
            with fsspec.open(video_path, 'rb') as f: video_bytes=io.BytesIO(f.read())
            vr=VideoReader(video_bytes, num_threads=1, ctx=cpu(0))
        else: # standard local path
            vr=VideoReader(video_path, num_threads=1, ctx=cpu(0))
        return vr
    return _loader

def get_image_loader():
    """ A universal image loader that works with local paths anc cloud URIs (s3://, gs://, etc) using fsspec"""
    
    def _loader(frame_path):
        # fsspec.open handles local paths or s3:// seamlessly
        if fsspec_imported:
            with fsspec.open(frame_path, 'rb') as f: img_bytes=f.read()
        else:
            with open(frame_path, 'rb') as f: img_bytes=f.read()
        # convert bytes to numpy array for opencv decoding
        img_np=np.frombuffer(img_bytes, np.uint8)
        img=cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        if img is None: raise ValueError(f'Failed to decode image at {frame_path}')
        # convert BGR (opencv default) to RGB (pytorch/transformer default)
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        return img
        
    return _loader
import torchvision
from torch import Tensor

class UCF101WithvideoId(torchvision.datasets.UCF101):
    """
    This class is based on https://github.com/pytorch/vision/blob/main/torchvision/datasets/ucf101.py and
    https://github.com/pytorch/vision/blob/main/references/video_classification/datasets.py
    """
    def __getitme__(self,idx:int)->tuple[Tensor, Tensor, int]:
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        label = self.samples[self.indices[video_idx]][1]
        if self.transform is not None:
            video = self.transform(video)
        return video, audio, label, video_idx
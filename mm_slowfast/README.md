## Prerequisite

Make sure that `pytorch` version is compatible with `mmcv`. Check `pytorch` and `mmcv` [compatibility](https://mmcv.readthedocs.io/en/latest/get_started/installation.html). `mmaction2` must be installed via **build from source**. 

- `conda create -n openmmlab -c conda-forge python=3.8 -y`
- `pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121` 
- `pip install -U openmim`
- `mim install mmengine` test `python -c "from mmengine.utils.dl_utils import collect_env;print(collect_env())"`
- `pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html` [see](https://mmcv.readthedocs.io/en/latest/get_started/installation.html)
-`mmaction2` [build from source](https://mmaction2.readthedocs.io/en/latest/get_started/installation.html#prerequisites)
- `pip install ipykernel matplotlib`
- `pip install moviepy` # to create and save movie files

## Start with PyTorch

[SlowFast](https://pytorch.org/hub/facebookresearch_pytorchvideo_slowfast/)
[Slow](https://pytorch.org/hub/facebookresearch_pytorchvideo_resnet/)
[X3D](https://pytorch.org/hub/facebookresearch_pytorchvideo_x3d/)

## Start with MMAction2
From the root of `MMAction2`
```
python demo/demo_inferencer.py  demo/demo.mp4 \
    --rec tsn --print-result \
    --label-file tools/data/kinetics/label_map_k400.txt \
	--pred-out-file output.rxr
```

To train
```
python tools/train.py configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py
```

To test
```
python tools/test.py configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py \
    work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb/best_acc/top1_epoch_6.pth
```

To check video
```
https://github.com/open-mmlab/mmaction2/blob/main/tools/analysis_tools/check_videos.py
```


## Model computational complexity and performance

[PytorchVideo](https://github.com/facebookresearch/SlowFast/tree/main/projects/pytorchvideo)

## Commands
- [train](https://github.com/facebookresearch/SlowFast/blob/main/GETTING_STARTED.md)
```
python tools/run_net.py \
  --cfg configs/Kinetics/C2D_8x8_R50.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_dataset \
  NUM_GPUS 2 \
  TRAIN.BATCH_SIZE 16 \
```

- [test](https://github.com/facebookresearch/SlowFast/blob/main/GETTING_STARTED.md)
```
python tools/run_net.py \
  --cfg configs/Kinetics/C2D_8x8_R50.yaml \
  DATA.PATH_TO_DATA_DIR path_to_your_dataset \
  TEST.CHECKPOINT_FILE_PATH path_to_your_checkpoint \
  TRAIN.ENABLE False \
```

## Google Colab

[mmaction2_tutorial.ipynb](https://colab.research.google.com/github/open-mmlab/mmaction2/blob/master/demo/mmaction2_tutorial.ipynb#scrollTo=No_zZAFpWC-a)
## Prerequisite

- `conda create -n slowfast -c conda-forge python=3.10`
- `pip3 install torch torchvision`
- `pip install -U fvcore`
- `pip install ipykernel matplotlib`
- `pip install av`

## Source
[repository](https://github.com/facebookresearch/SlowFast)

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
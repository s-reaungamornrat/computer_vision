Ultralytics API is

```
model=YOLO('yolo11n.pt')
results=model.train(data='coco8.yaml', epochs=3)
```
involving
- [BaseModel](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/model.py#L738)
- [YOLO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/model.py)
- [BaseTrainer](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/trainer.py)
- [DetectionTrainer](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/train.py)

We note that [YOLO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/model.py) calls a real model class with real base model in [ultralytics.nn.task](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py)
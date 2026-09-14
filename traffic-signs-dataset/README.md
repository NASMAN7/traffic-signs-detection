# Traffic Signs Dataset

Labeled image dataset for traffic sign detection and classification, structured for use with YOLO-based models.

## Structure
```
train/   ← training images + labels
valid/   ← validation images + labels
test/    ← test images + labels
```

## Usage
Used in conjunction with [projet_appr](https://github.com/AstA6XD9/projet_appr) for training a YOLO traffic sign detection model.
Configure the dataset path in `data.yaml` before training.
# Midterm Project – CNN Image Classification on CIFAR-10
**AI 100 – Penn State University**

## Problem
Multi-class image classification (10 categories) using a Convolutional Neural Network trained on the CIFAR-10 dataset.

## Dataset
- **CIFAR-10**: 60,000 color images (32×32 px), 10 classes
- 50,000 training samples / 10,000 test samples
- Classes: airplane, car, bird, cat, deer, dog, frog, horse, ship, truck

## Model
Custom CNN with 3 convolutional blocks + BatchNorm + Dropout + fully connected classifier.

## Requirements
```
torch
torchvision
matplotlib
numpy
```
Install with:
```bash
pip install torch torchvision matplotlib numpy
```

## Run
```bash
python train_cifar10.py
```

## Results
See `training_curves.png` and `metrics.json` after training.

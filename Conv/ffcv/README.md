# FFCV ImageNet Training Pipeline

A high-performance ImageNet training system using FFCV and PyTorch Lightning, supporting various advanced training techniques and loss functions.

## Features
- 🚀 4x Faster training with FFCV dataloading
- 🧠 Supports multiple loss functions:
  - Cross Entropy (CE)
  - Label Smoothing (LS)
  - Logit Penalty (LP)
  - Max Suppression (MS)
  - Online Label Smoothing (OLS)
- 🖥️ Mixed-precision training (BF16)
- 🧩 Modular architecture for easy customization
- 📊 TensorBoard & Neptune logging

## Installation
```bash
# Install pixi package manager
curl -fsSL https://pixi.sh/install.sh | bash

# Initialize environment (automatically installs dependencies)
pixi install
```

## Data Preparation
1. Convert ImageNet to FFCV format:
```bash
# Example: 500px images with 50% compression probability and quality 90
./create_data/write_imagenet.sh 500 0.5 90
```

## Configuration
Modify `config.toml` for training parameters:
```toml
[model]
model = "resnet50"
num_classes = 1000
use_blurpool = 1

[optimizer]
lr = 0.8
momentum = 0.9
weight_decay = 1e-4

[loss]
loss_type = "ms"  # ce, ls, lp, ms, ols
label_smoothing = 0.1
beta = 2e-3       # For logit penalty
```

## Training
```bash
# Single GPU (adjust CUDA_VISIBLE_DEVICES for multi-GPU)
CUDA_VISIBLE_DEVICES=0 python main.py

# Multi-GPU example (4 GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3 pixi run python main.py \
    --data.train_dataset=/path/to/train.ffcv \
    --data.val_dataset=/path/to/val.ffcv \
    --loss.loss_type=ms
```

## Results Monitoring
Metrics logged:
- Train/Validation Loss
- Top-1 & Top-5 Accuracy
- Learning Rate
- GPU Utilization

View logs with:
```bash
tensorboard --logdir=logs/
```

## Customization
1. **Add New Models**:
```python
# model.py
class CustomModel(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = torchvision.models.get_model(config['model_name'])
        self._apply_blurpool(self.model)
```

2. **Implement Custom Loss**:
```python
# losses.py
class CustomLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets)
        # Add custom regularization
        return ce_loss + self.alpha * logits.pow(2).mean()
```


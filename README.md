# Maximum-Suppression-Regularization

# Train Vision Transformer with MaxSup
We adopt [Deit](https://github.com/facebookresearch/deit) as the baseline model, and MaxSup is included in the `train_one_epoch` function of `engine.py`. 
```
cd Deit
python train_with_MaxSup.sh
```
To accelerate the data loading procedure, we additionally implemented a feature which caches the compressed ImageNet dataset as Zip file in the RAM (adapted from [Swin-Transformer](https://github.com/microsoft/Swin-Transformer)). It significantly reduces the data loading time with slow I/O speed and sufficient RAM, e.g., on a cluster in our case. It is activated by additionally providing `--cache` as an argument, as shown in the bash script. 

To enable the feature, please prepare the ImageNet data as follows:

## Zip Training Prepare

### 1. ZIP Archives
```
data/ImageNet-Zip/
├── train.zip    # Contains all training images in class subfolders
└── val.zip      # Contains all validation images
```

To create this zip, just zip the imagenet datset folder.

### 2. Mapping Files
```
data/ImageNet-Zip/
├── train_map.txt    # Training image paths and labels
└── val_map.txt      # Validation image paths and labels
```

#### Training Map File (train_map.txt)
- **Format**: `<class_folder>/<image_filename>\t<class_label>`
- **Example entries**:
  ```
  n01440764/n01440764_10026.JPEG    0
  n01440764/n01440764_10027.JPEG    0
  n01440764/n01440764_10029.JPEG    0
  ```

#### Validation Map File (val_map.txt)
- **Format**: `<image_filename>\t<class_label>`
- **Example entries**:
  ```
  ILSVRC2012_val_00000001.JPEG    65
  ILSVRC2012_val_00000002.JPEG    970
  ILSVRC2012_val_00000003.JPEG    230
  ```

You should make sure 
  - Paths include class folder structure
  - Labels are zero-based integers


# Training ConvNets with MaxSup

We use Resnet-50 as baseline, and MaxSup is included in the train_one_epoch function of `main.py`. You can easily select any model for training in `main.py`.


> Please find the pretrained weights as well as the training log in the release "checkpoint_deit".






# Maximum-Suppression-Regularization

# Train Vision Transformer with MaxSup
We adopt [Deit](https://github.com/facebookresearch/deit) as the baseline model, and MaxSup is included in the `train_one_epoch` function of `engine.py`. 
```
cd Deit
python train_with_MaxSup.sh
```
To accelerate the data loading procedure, we additionally implemented a feature which caches the compressed ImageNet dataset as Zip file in the RAM (adapted from [Swin-Transformer](https://github.com/microsoft/Swin-Transformer)). It significantly reduces the data loading time with slow I/O speed and sufficient RAM, e.g., on a cluster in our case. It is activated by additionally providing `--cache` as an argument, as shown in the bash script. 

To enable the feature, please prepare the ImageNet data as follows:


Please find the pretrained weights as well as the training log in the release "checkpoint_deit".






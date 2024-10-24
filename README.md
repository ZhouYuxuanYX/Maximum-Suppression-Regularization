# Maximum-Suppression-Regularization

# Train Vision Transformer with MaxSup
We adopt [Deit](https://github.com/facebookresearch/deit) as the baseline model, and MaxSup is included in the `train_one_epoch` function of `engine.py`. 
```
cd Deit
python train_with_MaxSup.sh
```
Notably, we additionally implemented a feature for accelerating the data loading procedure via caching the compressed ImageNet dataset as Zip file in the RAM. This significantly reduces the data loading time with slow I/O speed but large RAM, e.g., on a cluster in our case. It is activated by providing `--cache` as an argument, please remove it for your own need. 

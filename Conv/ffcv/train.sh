# 4 GPU training (use only 1 for ResNet-18 training)
export CUDA_VISIBLE_DEVICES=0,1,2,3


python train_imagenet.py --config-file config.yaml \
    --data.train_dataset=/datadrive3/hengl/imagenet1k_ffcv/train_500_0.50_90.ffcv \
    --data.val_dataset=/datadrive3/hengl/imagenet1k_ffcv/val_500_0.50_90.ffcv \
    --data.in_memory=1 \
    --logging.folder=log \
    --experiment.experiment_name=ms_12345_long_18 \
    --loss.loss_type=ms

python train_imagenet.py --config-file config.yaml \
    --data.train_dataset=/datadrive3/hengl/imagenet1k_ffcv/train_500_0.50_90.ffcv \
    --data.val_dataset=/datadrive3/hengl/imagenet1k_ffcv/val_500_0.50_90.ffcv \
    --data.in_memory=1 \
    --logging.folder=log \
    --experiment.experiment_name=logit_penalty_12345_long_18 \
    --loss.loss_type=lp

python train_imagenet.py --config-file config.yaml \
    --data.train_dataset=/datadrive3/hengl/imagenet1k_ffcv/train_500_0.50_90.ffcv \
    --data.val_dataset=/datadrive3/hengl/imagenet1k_ffcv/val_500_0.50_90.ffcv \
    --data.in_memory=1 \
    --logging.folder=log \
    --experiment.experiment_name=ols_12345_long_18 \
    --loss.loss_type=ols

python train_imagenet.py --config-file config.yaml \
    --data.train_dataset=/datadrive3/hengl/imagenet1k_ffcv/train_500_0.50_90.ffcv \
    --data.val_dataset=/datadrive3/hengl/imagenet1k_ffcv/val_500_0.50_90.ffcv \
    --data.in_memory=1 \
    --logging.folder=log \
    --experiment.experiment_name=ls_12345_long_18 \
    --loss.loss_type=ls

python train_imagenet.py --config-file config.yaml \
    --data.train_dataset=/datadrive3/hengl/imagenet1k_ffcv/train_500_0.50_90.ffcv \
    --data.val_dataset=/datadrive3/hengl/imagenet1k_ffcv/val_500_0.50_90.ffcv \
    --data.in_memory=1 \
    --logging.folder=log \
    --experiment.experiment_name=ce_12345_long_18 \
    --loss.loss_type=ce
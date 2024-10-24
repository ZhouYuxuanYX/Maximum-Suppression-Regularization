#!/bin/bash
eval "$(/root/anaconda3/bin/conda shell.bash hook)"
source ~/.bashrc
conda activate vit
pip install timm==0.4.12 lmdb
# CISPA CLUSTER A100 PARTITION has 3 nodes, each node has 8 A100 GPUS
# on xe partition, only 4 gpus are available and 2 nodes can't run, so increase batch size
OMP_NUM_THREADS=4 torchrun --nnodes=1 --nproc_per_node=4 /home/c02yuzh/CISPA-projects/rectified_softmax_ml-2023/Deit3/main.py --cache --model our_patch16_224 --batch-size 256 --data-path /home/c02yuzh/CISPA-projects/rectified_softmax_ml-2023/data/ImageNet --output_dir /home/c02yuzh/CISPA-projects/rectified_softmax_ml-2023/Deit3/deit_small_max_suppression_cls_token_xe_no_ls_qkv_true_no_cache --seed 0 --num_workers 24 --pin-mem --dist-eval --smoothing 0 --resume /home/c02yuzh/CISPA-projects/rectified_softmax_ml-2023/Deit3/deit_small_max_suppression_cls_token_xe_no_ls_qkv_true


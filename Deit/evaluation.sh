#python main.py --eval --batch-size 2 --resume output11/best_checkpoint.pth --model deit_small_patch16_224 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet
#
#python main.py --eval --batch-size 64 --resume output_no_cls/best_checkpoint.pth --model deit_small_patch16_224 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet

#python main.py --eval --batch-size 2 --resume output_no_cls_2_segments_spline/best_checkpoint.pth --model deit_small_patch16_224 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet

#torchrun --nproc_per_node=8 --master_port 9000 main.py --eval --batch-size 4 --resume deit_s/checkpoint.pth --model deit_small_patch16_224 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet

#python main.py --eval --batch-size 4 --resume output_no_cls_classification_layer_only/best_checkpoint.pth --model deit_small_patch16_224 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet

#python main.py --eval --batch-size 64 --resume output_cls_token_recmax_all/best_checkpoint.pth --model deit_small_patch16_224 --data-path /disk1/zhouyuxuan/ImageNet


#torchrun --nproc_per_node=1 --master_port 9000 --master_addr 'localhost' main.py --eval --lmdb --batch-size 64 --resume output_no_cls_2_segments_head_and_attn_minmax_substraction_independent_first_second_order_ranges/best_checkpoint.pth --model deit_small_patch16_224 --data-path /home/c02yuzh/CISPA-projects/rectified_softmax_ml-2023/data/ImageNet

#for i in {1..300..50}
#for i in {1,49,99,149,199,249,299}
#do
#  torchrun --nproc_per_node=1 --master_port 9000 --master_addr 'localhost' evaluate.py --eval --lmdb --batch-size 64 --num_workers 16 --resume recmax/checkpoint_epoch_$i.pth --model deit_small_patch16_224 --data-path /home/c02yuzh/CISPA-projects/rectified_softmax_ml-2023/data/ImageNet
#done
#torchrun --nproc_per_node=1 --master_port 9000 --master_addr 'localhost' evaluate.py --eval --lmdb --batch-size 64 --num_workers 16 --resume recmax_truncated_normal_init/checkpoint_epoch_299.pth --model deit_small_patch16_224 --data-path /home/c02yuzh/CISPA-projects/rectified_softmax_ml-2023/data/ImageNet

#torchrun --nproc_per_node=1 --master_port 9000 --master_addr 'localhost' evaluate.py --eval --lmdb --batch-size 64 --num_workers 16 --resume deit-s_baseline_gap/checkpoint_epoch_299.pth --model deit_small_patch16_224 --data-path /home/c02yuzh/CISPA-projects/rectified_softmax_ml-2023/data/ImageNet


# cam visualization scopion
#torchrun --nproc_per_node=1 --master_port 9000 --master_addr 'localhost' evaluate.py --cam --lmdb --image-path /home/c02yuzh/CISPA-projects/rectified_softmax_ml-2023/data/ImageNet/val/n01770393/ILSVRC2012_val_00028434.JPEG --batch-size 64 --num_workers 16 --resume deit-s_baseline_gap/checkpoint_epoch_299.pth --model deit_small_patch16_224 --data-path /home/c02yuzh/CISPA-projects/rectified_softmax_ml-2023/data/ImageNet

## bird
torchrun --nproc_per_node=1 --master_port 9000 --master_addr 'localhost' evaluate.py --eval --lmdb --batch-size 64 --num_workers 16 --resume coded_attn_deit_small/checkpoint_epoch_299.pth --model deit_small_patch16_224 --data-path /home/c02yuzh/CISPA-projects/rectified_softmax_ml-2023/data/ImageNet


# shell
#torchrun --nproc_per_node=1 --master_port 9000 --master_addr 'localhost' evaluate.py --eval --lmdb --image-path /home/c02yuzh/CISPA-projects/rectified_softmax_ml-2023/data/ImageNet/val/n01669191/ILSVRC2012_val_00037739.JPEG --batch-size 64 --num_workers 16 --resume recmax_truncated_normal_init/checkpoint_epoch_299.pth --model deit_small_patch16_224 --data-path /home/c02yuzh/CISPA-projects/rectified_softmax_ml-2023/data/ImageNet


#torchrun --nproc_per_node=1 --master_port 9000 --master_addr 'localhost' evaluate.py --lrp --lmdb --batch-size 64 --num_workers 16 --resume recmax_truncated_normal_init/checkpoint_epoch_299.pth --model deit_small_patch16_224 --data-path /home/c02yuzh/CISPA-projects/rectified_softmax_ml-2023/data/ImageNet


# this is the first order recmax, but checkpoint overwritten by epoch 2
#torchrun --nproc_per_node=1 --master_port 9000 --master_addr 'localhost' evaluate.py --eval --lmdb --batch-size 64 --num_workers 16 --resume output_no_cls_classification_layer_too/best_checkpoint.pth --model deit_small_patch16_224 --data-path /home/c02yuzh/CISPA-projects/rectified_softmax_ml-2023/data/ImageNet

# due to the stupid automatic upload, best_checkpoint.pth and checkpoint_epoch_299.pth is damaged
# no_cls_baseline forgot to remove cls, but it uses average pooling
# this acc seems higher using engine_eval just because only the first 960 images are used
#torchrun --nproc_per_node=1 --master_port 9000 --master_addr 'localhost' evaluate.py --eval --lmdb --batch-size 64 --num_workers 16 --resume output7/best_checkpoint.pth --model deit_small_patch16_224 --data-path /home/c02yuzh/CISPA-projects/rectified_softmax_ml-2023/data/ImageNet

##### important note #####

# the train_map.txt and val_map.txt maps the folder to classes different from the default imagefolder, due to the orders of directories
# so if i train with zip, i have to evaluate using zip as well to obtain the correct test acurracy!!!!

#torchrun --nproc_per_node=1 --master_port 9000 --master_addr 'localhost' main.py --eval --cache --batch-size 64 --num_workers 16 --resume checkpoint_epoch_49.pth --model deit_small_patch16_224 --data-path /home/c02yuzh/CISPA-projects/rectified_softmax_ml-2023/data/ImageNet


#python3 evaluate.py --eval --lmdb --batch-size 64 --num_workers 16 --resume recmax/checkpoint_epoch_291.pth --model deit_small_patch16_224 --data-path /home/c02yuzh/CISPA-projects/rectified_softmax_ml-2023/data/ImageNet

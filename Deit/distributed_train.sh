#!/bin/bash



#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_small_patch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output4 \

# have defined setup_for_distributed2() in utils.py for debugging

# change to torchrun

#export NCCL_DEBUG_SUBSYS=COLL
#export NCCL_DEBUG=INFO
#torchrun --nproc_per_node=8 main.py --model deit_small_patch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output4 \
#--clip-grad

# even with clip grad, still easily run into nan error with amp, also encountered by others, see deit issues

# this was indeed original transformer 5min per train epoch
#torchrun --nproc_per_node=8 main.py --model our_patch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output7 \

# even if i remove the whole attn layer, run time just reduce to 4min30s, very small reduction, this is true for the deit orginal code too
# and remove all mlps, it's also 4min 30 s per train epoch
# remove both still 4 min 27 s
# change num workers from 12 to 24 change to 4 min 23s
# the bottleneck must be dataloading!
# set pin memory to false and num workder 24, time is reduced to 4min 16s
# adding OMP_NUM_THREADS=1 no change
# change num workers to 64 and not adding omp_num_threads, time is increased to 4min33s
# change to 48

# learnable margin w/o learnable ratio 80.1 vs 79.9 + 0.2, with learnable ratio, 80.0
#torchrun --nproc_per_node=8 main.py --model our_patch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output9 \

# output10 80.21
#torchrun --nproc_per_node=8 main.py --model our_patch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output10

# output11 89.34
#torchrun --nproc_per_node=8 main.py --model our_patch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output11


# output12 headwise parameter much lower 80.08
#torchrun --nproc_per_node=8 main.py --model our_patch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output12

## output13 much worse, stopped at 55 peochs
#torchrun --nproc_per_node=8 main.py --model our_patch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output13

## output13 much worse, stopped at 55 peochs
#torchrun --nproc_per_node=8 main.py --model our_patch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output14

# gap instead of cls, 80.53%!!! best until now
#torchrun --nproc_per_node=8 main.py --model our_patch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output_no_cls

# try to renormalize attention to [-1, 1], substract negative feature?
# how about use tanh instead of softmax for cls token

# test time suddenly becomes very slow, but i checked with no flip setup, it's slow now as well
# not sure what's the problem, solved, forgot to comment out the code for saving images at each iteration

# 80.02 can't really learn to flip, because it's a sudden change, no reliable supervision signal
# was using cls token, but still worse than 80.34!!! no need to rerun

#torchrun --nproc_per_node=8 main.py --model our_patch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output_no_cls_auto_flip # was indeed using cls token

# before it can only add, but not substract, after putting t1,t2 inside, we still need a parameter outside
# 80.25, better, but still the flip operation is non-differentiable, thus not learnable
# was using cls token, but still worse than 80.34!!! no need to rerun
#torchrun --nproc_per_node=8 main.py --model our_patch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output_no_cls_auto_flip_corrected # was indeed using cls token

# ab-bx:
# if bx<ab: if b<0, it's equivalent to x>a, else it's x<a
# the sign flipping is not differentiable and thus not learnable
# we could use soft sign funciton, i.e., tanh and multiply by it's scale

# 80.61!!! better than before!!
# because all parameters are initialized as zeros, there are products between two parameters
# the gradient is always zero, rerun!!!!
#torchrun --nproc_per_node=8 main.py --model our_patch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output_no_cls_auto_flip_add_tanh

# correct init， 80.3%
#torchrun --nproc_per_node=8 main.py --model our_patch16_224 -batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output_no_cls_auto_flip_add_tanh_correct_init


# try soft sign

#torchrun --nproc_per_node=8 main.py --model our_patch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output_no_cls_auto_flip_gate_random_init

# 12min 30s using original implementation,
# becomes much faster (9min 30s) when wrapped as a function and the parameters are combined as tensors instead of individual paramters
# 80.51 better than 2 segments, this is summed not averaged
#torchrun --nproc_per_node=8 main.py --model our_patch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output_no_cls_auto_flip_gate_random_init_3_segments

# 4 segments, 80.51, 10min30s
#torchrun --nproc_per_node=8 main.py --model our_patch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output_no_cls_auto_flip_gate_random_init_4_segments

#torchrun --nproc_per_node=8 main.py --model our_patch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output_no_cls_auto_flip_gate_random_init_4_segments



# 79.89 correct implement
#torchrun --nproc_per_node=8 main.py --model our_patch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output_no_cls_2_segments_spline

# becomes worse after normalization
#torchrun --nproc_per_node=8 main.py --model our_patch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output_no_cls_norm_to_angle

# 80.3 not good
#torchrun --nproc_per_node=8 main.py --model our_patch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output_no_cls_weighted_attn

#torchrun --nproc_per_node=8 main.py --model our_patch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output_no_cls_add_linear_transform

# multiply temperature becomes worse!!! stopped to save time
#torchrun --nproc_per_node=8 main.py --model our_patch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output_no_cls_add_linear_transform_no_bias

# adding a bias already enables full control, because the segments can be shifted anyway and the turning point can be adjusted by bias
# worse than with temperature and than without both!!!!
#torchrun --nproc_per_node=8 main.py --model our_patch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output_no_cls_add_bias_only

# not good no matter multiply w/o abs or with abs
#torchrun --nproc_per_node=8 main.py --model our_padtch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output_no_cls_add_abs_temperature_only


## 80.69!! best ever! adding to cls head alone could have more than 0.2 improvement!
#torchrun --nproc_per_node=8 main.py --model our_patch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output_no_cls_classification_layer_too

# not good
#torchrun --nproc_per_node=8 main.py --model our_patch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output_no_cls_classification_layer_too_add_learnable_base

# 80.4!!!! it works!!!!
#torchrun --nproc_per_node=8 main.py --model our_patch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output_no_cls_classification_layer_only

# 80.35
#torchrun --nproc_per_node=8 main.py --model our_patch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output_no_cls_baseline_vanilla_softmax

# 80.5 worse!
#torchrun --nproc_per_node=8 main.py --model our_patch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output_no_cls_4_segments_head_and_attn

# worse ! stopped
#torchrun --nproc_per_node=8 main.py --model our_patch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output_no_cls_3_segments_head_and_attn

# min max and bias, temperature,事实证明那个x-range的差值很关键，不要差值的话效果很不好
#torchrun --nproc_per_node=8 main.py --model our_patch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output_no_cls_2_segments_head_and_attn_minmax_simple_bias_temperature


# 没有residual的话效果很差, temperature 负作用

# 80.71!!!!! square of the substraction
#torchrun --nproc_per_node=8 main.py --model our_patch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output_no_cls_2_segments_head_and_attn_minmax_substraction_square


#torchrun --nproc_per_node=8 main.py --model our_patch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output_no_cls_2_segments_head_and_attn_minmax_substraction_learnable_power

# 80.72
#torchrun --nproc_per_node=8 main.py --model our_patch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output_no_cls_2_segments_first_and_second_order_same_ranges

# worse with bias!!!!! 80.61, even put the bias into min max

# 80.96%!!!! best when adding the second order terms with their own ranges
#torchrun --nproc_per_node=8 main.py --model our_patch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output_no_cls_2_segments_head_and_attn_minmax_substraction_independent_first_second_order_ranges


# not work, loss keeps oscillating and not reduces
#torchrun --nproc_per_node=8 main.py --model our_patch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output_no_cls_2_segments_head_and_attn_minmax_substraction_independent_first_second_third_order_ranges

# remove exp is very bad
#torchrun --nproc_per_node=8 main.py --model our_patch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output_no_cls_2_segments_head_and_attn_minmax_substraction_independent_first_second_order_ranges_remove_exp_relu

# attention only
#torchrun --nproc_per_node=8 main.py --model our_patch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output_no_cls_2_segments_only_attn_minmax_substraction_independent_first_second_order_ranges

# head only
#torchrun --nproc_per_node=8 main.py --model our_patch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output_no_cls_2_segments_only_head_minmax_substraction_independent_first_second_order_ranges

## head only first order
#torchrun --nproc_per_node=8 main.py --model our_patch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output_no_cls_2_segments_only_head_minmax_substraction_independent_first_order_ranges


# head only first order
#torchrun --nproc_per_node=8 main.py --model our_patch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output_no_cls_temperature_0.1

#torchrun --nproc_per_node=8 main.py --model our_patch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output_no_cls_temperature_1

#torchrun --nproc_per_node=8 main.py --model our_patch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output_no_cls_classification_layer_too_exp_divide_max_attn

# worse than with bias
#torchrun --nproc_per_node=8 main.py --model our_patch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output_no_cls_2_segments_spline_remove_bias


# add one more component 8min55s per epoch, time increased
# seems to be even better, but stopped first, because the zero initialization might be wrong, no effect
#torchrun --nproc_per_node=8 main.py --model our_patch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output_no_cls_auto_flip_add_tanh_3_components


# 80.56 no improvement over the best config, linear mapping
#torchrun --nproc_per_node=8 main.py --model our_patch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output_no_cls_auto_flip_add_tanh_qk_dependent_range

# only q dependent range, linear mapping, no improvement, stopped
#torchrun --nproc_per_node=8 main.py --model our_patch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output_no_cls_auto_flip_add_tanh_q_dependent_range

# non linear mapping, only q dependent range, no improvement, stopped
#torchrun --nproc_per_node=8 main.py --model our_patch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output_no_cls_auto_flip_add_tanh_q_dependent_range_non_linear


# 80.41 with sign
#torchrun --nproc_per_node=8 main.py --model our_patch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output_no_cls_auto_flip_add_sign

# tanh inside 80.41
#torchrun --nproc_per_node=8 main.py --model our_patch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output_no_cls_auto_flip_add_tanh_inside



# it becomes much slower using the gate, 7min30s to 9 min30s per epoch, 80.44
#torchrun --nproc_per_node=8 main.py --model our_patch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output_no_cls_auto_flip_gate

# very bad using the elegant solution, change to the complicated one
# still worse than the best ones, stopped
#torchrun --nproc_per_node=8 main.py --model our_patch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output_no_cls_3_segment_corrected


# next !!!!

# softsign is better than tanh

# seed 0
#torchrun --nproc_per_node=8 main.py --model our_patch16_224 --batch-size 128 --data-path /disk1/zhouyuxuan/ImageNet --output_dir output_cls_token_recmax_all

# seed 0
#torchrun --nproc_per_node=8 main.py --model our_patch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output_cls_token_softmax_all

# seed 1
#torchrun --nproc_per_node=8 main.py --model our_patch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output_cls_token_softmax_all_seed1 --seed 1


# was not using cls token, forgot to change the mean back!!!!
## seed 0
#torchrun --nproc_per_node=8 main.py --model our_patch16_224 --batch-size 128 --data-path /disk1/zhouyuxuan/ImageNet --output_dir output_cls_token_recmax_all_correct


# seed 0
#torchrun --nproc_per_node=8 main.py --model our_patch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output_cls_token_softmax_all_correct

# seed 1 zero init ts and ranges
# 80.49 when multiply scale before selu and after cosine attention
# tanh restrict ts to -1, 1 becomes slightly worse
# 80.81 tanh + temperature
#torchrun --nproc_per_node=8 main.py --model our_patch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir output_gap_segmax_all_correct_seed0_tanh_ts_temperature --seed 0
eval "$(/root/anaconda3/bin/conda shell.bash hook)"
source ~/.bashrc
conda activate vit
conda list
# CISPA CLUSTER A100 PARTITION has 3 nodes, each node has 8 A100 GPUS
OMP_NUM_THREADS=8 torchrun --nnodes=2 --rdzv_backend=c10d --rdzv_id 123 --max_restarts=100 --rdzv_endpoint $1:29600 --nproc_per_node=4 /home/c02yuzh/CISPA-projects/rectified_softmax_ml-2023/Deit/main.py --cache --model our_patch16_224 --batch-size 128 --data-path /home/c02yuzh/CISPA-projects/rectified_softmax_ml-2023/data/ImageNet --output_dir deit_small_normalization --seed 0 --num_workers 16 --pin-mem --dist-eval

#torchrun --nproc_per_node=1 /home/c02yuzh/CISPA-projects/rectified_softmax_ml-2023/Deit/main.py --model our_patch16_224 --batch-size 128 --data-path /home/c02yuzh/CISPA-projects/rectified_softmax_ml-2023/data/ImageNet --output_dir test --seed 0


# base model always throw error, try clip grad doesn't work
# i have to set amp false outside the whole function, because amp is used by default, disable autocast in engine.py
# but it needs 49 min one epoch without amp, can't afford

# grad-clip is not applied to deit-s
# but without grad-clip deit-b doesen't converge, check if grad-clip helps


#torchrun --nproc_per_node=8 main.py --model deit_base_patch16_224 --batch-size 128 --data-path /ssd2/zhouyuxuan/Dataset/ImageNet --output_dir deit-b_output_gap_segmax_all_correct_seed0 --seed 0

## chebyshev polynomials
## next step, remove softmax, error after few epochs
#torchrun --nproc_per_node=8 main.py --model our_patch16_224 --batch-size 128 --data-path /disk1/zhouyuxuan/ImageNet --output_dir deit-s_output_gap_segmax_all_correct_seed0_chebyshev --seed 0

#!/bin/bash

CIFAR='--data_path data/ --log_every 100 --dataset cifar100 --cuda --log_dir logs/'

for lr in 0.001
do
    for i in {0..9}
    do
        python3 main.py $CIFAR --model lamaml_cifar --expt_name lamaml --memories 400 --batch_size 5 --replay_batch_size 5 --n_epochs 1 \
                            --multiplier_lr 0.0005 --opt_lr 0.00002 --alpha_init $lr  --glances 1 --loader class_incremental_loader --increment 5 \
                            --arch "pc_cnn" --cifar_batches 5  --log_every 3125  --class_order random \
                            --seed ${i} --grad_clip_norm 1.0 --calc_test_accuracy --validation 0.1  --remove_bias --normalize_hidden --learn_lr --learn_layer_lr --learn_inhibition_multiplier 
    done
done


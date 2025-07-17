#!/bin/bash

ROT="--n_layers 4  --n_hiddens 100 --data_path data/ --log_every 100 --samples_per_task 200 --dataset mnist_rotations  --log_dir logs/"

for lr in 0.01
do

    for i in {0..9}
    do
        python3 main.py $ROT  --model lamaml --seed ${i} --memories 200 --batch_size 5 --replay_batch_size 5 --n_epochs 1 --glances 1 --alpha_init $lr --use_old_task_memory  \
                        --add_item_labels 0.0 --num_of_item_labels 0 --item_option per_item --multiplier_lr 0.05 --opt_lr 0.0002  --remove_bias  --calc_test_accuracy  --normalize_hidden  --small_test  --learn_lr --learn_layer_lr --learn_inhibition_multiplier     
    done
done


#!/bin/bash
FASHION="--n_layers 4 --n_hiddens 200 --data_path data/ --log_every 100 --samples_per_task 200 --dataset fashion_mnist  --log_dir logs/"

ROT="--n_layers 4 --n_hiddens 200 --data_path data/ --log_every 100 --samples_per_task 100 --dataset mnist_rotations  --log_dir logs/"

for lr in 0.15
do
    for i in {0..9}
    do
        python3 main.py $ROT --model lamaml --seed ${i} --memories 200 --batch_size 5 --replay_batch_size 5 --n_epochs 1 --glances 1 --alpha_init $lr --use_old_task_memory  \
                            --add_item_labels 0.0 --num_of_item_labels 1 --item_option per_item --multiplier_lr 0.2 --opt_lr 0.002 --opt_wt 0.05  --remove_bias  --normalize_hidden  --small_test  --learn_lr --learn_layer_lr  --learn_inhibition_multiplier  --hsplit_idx 3 
    done
done





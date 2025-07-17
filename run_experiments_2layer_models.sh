#!/bin/bash

ROT="--n_layers 2 --n_hiddens 200 --data_path data/ --log_every 100 --samples_per_task 200 --dataset mnist_rotations  --log_dir logs/"
FASHION="--n_layers 2 --n_hiddens 200  --data_path data/ --log_every 100 --samples_per_task 200 --dataset fashion_mnist  --log_dir logs/"


# FASHION MNIST runs
for lr in 0.0001 0.00025 0.0005 0.001 0.0025 0.005 0.01 0.025 0.05 0.1 0.25 0.5
do
    # baseline
    for i in {0..9}
    do
        nohup python3 main.py $FASHION  --model lamaml --seed ${i} --memories 200 --batch_size 5 --replay_batch_size 5 --n_epochs 1 --glances 1 --alpha_init $lr --use_old_task_memory  \
                        --add_item_labels 0.0 --num_of_item_labels 0 --item_option per_item --multiplier_lr 0.1 --opt_lr 0.0005  --remove_bias  --calc_test_accuracy  --normalize_hidden  --small_test
    done

    # s-only
    for i in {0..9}
    do
        nohup python3 main.py $FASHION  --model lamaml --seed ${i} --memories 200 --batch_size 5 --replay_batch_size 5 --n_epochs 1 --glances 1 --alpha_init $lr --use_old_task_memory  \
                        --add_item_labels 0.0 --num_of_item_labels 0 --item_option per_item --multiplier_lr 0.1 --opt_lr 0.0005  --remove_bias  --calc_test_accuracy  --normalize_hidden  --small_test   --learn_inhibition_multiplier
    done

    # lr-only
    for i in {0..9}
    do
        nohup python3 main.py $FASHION  --model lamaml --seed ${i} --memories 200 --batch_size 5 --replay_batch_size 5 --n_epochs 1 --glances 1 --alpha_init $lr --use_old_task_memory  \
                        --add_item_labels 0.0 --num_of_item_labels 0 --item_option per_item --multiplier_lr 0.1 --opt_lr 0.0005  --remove_bias  --calc_test_accuracy  --normalize_hidden  --small_test   --learn_lr --learn_layer_lr
    done

    # meta-s+lr
    for i in {0..9}
    do
        nohup python3 main.py $FASHION  --model lamaml --seed ${i} --memories 200 --batch_size 5 --replay_batch_size 5 --n_epochs 1 --glances 1 --alpha_init $lr --use_old_task_memory  \
                        --add_item_labels 0.0 --num_of_item_labels 0 --item_option per_item --multiplier_lr 0.1 --opt_lr 0.0005  --remove_bias  --calc_test_accuracy  --normalize_hidden  --small_test    --learn_inhibition_multiplier   --learn_lr --learn_layer_lr
    done
done


# ROTATED MNIST runs
for lr in 0.0001 0.00025 0.0005 0.001 0.0025 0.005 0.01 0.025 0.05 0.1 0.25 0.5
do
    # baseline
    for i in {0..9}
    do
        python3 main.py $ROT  --model lamaml --seed ${i} --memories 200 --batch_size 5 --replay_batch_size 5 --n_epochs 1 --glances 1 --alpha_init $lr --use_old_task_memory  \
                        --add_item_labels 0.0 --num_of_item_labels 0 --item_option per_item --multiplier_lr 0.02 --opt_lr 0.0002  --remove_bias  --calc_test_accuracy   --normalize_hidden  --small_test  
    done

    # s-only
    for i in {0..9}
    do
        nohup python3 main.py $ROT  --model lamaml --seed ${i} --memories 200 --batch_size 5 --replay_batch_size 5 --n_epochs 1 --glances 1 --alpha_init $lr --use_old_task_memory  \
                        --add_item_labels 0.0 --num_of_item_labels 0 --item_option per_item --multiplier_lr 0.02 --opt_lr 0.0002  --remove_bias  --calc_test_accuracy   --normalize_hidden  --small_test  --learn_inhibition_multiplier 
    done

    # lr-only
    for i in {0..9}
    do
        nohup python3 main.py $ROT  --model lamaml --seed ${i} --memories 200 --batch_size 5 --replay_batch_size 5 --n_epochs 1 --glances 1 --alpha_init $lr --use_old_task_memory  \
                        --add_item_labels 0.0 --num_of_item_labels 0 --item_option per_item --multiplier_lr 0.02 --opt_lr 0.0002  --remove_bias  --calc_test_accuracy  --normalize_hidden  --small_test  --learn_lr --learn_layer_lr 
    done

    # meta-s+lr
    for i in {0..9}
    do
        python3 main.py $ROT  --model lamaml --seed ${i} --memories 200 --batch_size 5 --replay_batch_size 5 --n_epochs 1 --glances 1 --alpha_init $lr --use_old_task_memory  \
                        --add_item_labels 0.0 --num_of_item_labels 0 --item_option per_item --multiplier_lr 0.02 --opt_lr 0.0002  --remove_bias  --calc_test_accuracy    --normalize_hidden  --small_test  --learn_lr --learn_layer_lr --learn_inhibition_multiplier  
    done

done


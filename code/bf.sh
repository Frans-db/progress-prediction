#!/bin/sh

# embed
python main.py --seed 42 \
    --dataset breakfast \
    --data_dir rgb-images \
    --train_split all_scrambledegg.txt \
    --test_split test_scrambledegg_s1.txt \
    --batch_size 1 \
    --network progressnet_flat \
    --backbone vgg16 \
    --load_experiment progressnet_flat_bf_scrambledegg_1 \
    --load_iteration 500 \
    --embed \
    --embed_batch_size 32 \
    --embed_dir features/progressnet_scrambledegg_1 \
    --num_workers 1

# sequence
python main.py --seed 42 \
    --experiment_name progressnet_bf_scrambledegg_1 \
    --wandb_tags fold_s1 scrambledegg \
    --dataset breakfast \
    --data_dir features/progressnet_scrambledegg_1 \
    --train_split train_scrambledegg_s1.txt \
    --test_split test_scrambledegg_s1.txt \
    --feature_dim 2048 \
    --batch_size 1 \
    --iterations 5000 \
    --network progressnet \
    --dropout_chance 0.3 \
    --optimizer sgd \
    --loss smooth_l1 \
    --momentum 0.9 \
    --lr 1e-3 \
    --weight_decay 5e-4 \
    --lr_decay 0.1 \
    --lr_decay_every 10000 \
    --log_every 50 \
    --test_every 500

# segment
python main.py --seed 42 \
    --experiment_name progressnet_bf_scrambledegg_segment_1 \
    --wandb_tags fold_s1 scrambledegg \
    --dataset breakfast \
    --data_dir features/progressnet_scrambledegg_1 \
    --train_split train_scrambledegg_s1.txt \
    --test_split test_scrambledegg_s1.txt \
    --feature_dim 2048 \
    --subsample \
    --batch_size 1 \
    --iterations 5000 \
    --network progressnet \
    --dropout_chance 0.3 \
    --optimizer sgd \
    --loss smooth_l1 \
    --momentum 0.9 \
    --lr 1e-3 \
    --weight_decay 5e-4 \
    --lr_decay 0.1 \
    --lr_decay_every 10000 \
    --log_every 50 \
    --test_every 500

# indices
python main.py     --seed 42 \
    --experiment_name progressnet_bf_scrambledegg_indices_1 \
    --wandb_tags fold_s1 scrambledegg \
    --dataset breakfast \
    --data_dir features/progressnet_scrambledegg_1 \
    --train_split train_scrambledegg_s1.txt \
    --test_split test_scrambledegg_s1.txt \
    --feature_dim 2048 \
    --indices \
    --indices_normalizer 3117 \
    --subsample \
    --batch_size 1 \
    --iterations 5000 \
    --network progressnet \
    --dropout_chance 0.3 \
    --optimizer sgd \
    --loss smooth_l1 \
    --momentum 0.9 \
    --lr 1e-3 \
    --weight_decay 5e-4 \
    --lr_decay 0.1 \
    --lr_decay_every 10000 \
    --log_every 50 \
    --test_every 500
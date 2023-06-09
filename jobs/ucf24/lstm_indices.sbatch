#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

srun python /home/nfs/fransdeboer/mscfransdeboer/code/main.py \
    --seed 42 \
    --experiment_name lstmnet_segments_ucf24 \
    --dataset ucf24 \
    --data_dir features/resnet152 \
    --feature_dim 2048 \
    --indices \
    --indices_normalizer 128 \
    --train_split train_tubes.txt \
    --test_split test_tubes.txt \
    --batch_size 1 \
    --iterations 30000 \
    --network lstmnet \
    --optimizer sgd \
    --loss smooth_l1 \
    --momentum 0.9 \
    --lr 1e-3 \
    --dropout_chance 0.3 \
    --weight_decay 1e-2 \
    --lr_decay 0.1 \
    --lr_decay_every 10000 \
    --log_every 100 \
    --test_every 5000 \
    --subsample
#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

srun python /home/nfs/fransdeboer/mscfransdeboer/code/main.py \
    --seed 42 \
    --experiment_name lstmnet_ucf24 \
    --dataset ucf24 \
    --data_dir features/resnet152 \
    --train_split train_tubes.txt \
    --test_split test_tubes.txt \
    --feature_dim 2048 \
    --batch_size 1 \
    --iterations 100000 \
    --network lstmnet \
    --dropout_chance 0.5 \
    --optimizer sgd \
    --loss smooth_l1 \
    --momentum 0.9 \
    --lr 1e-3 \
    --weight_decay 5e-4 \
    --lr_decay 0.1 \
    --lr_decay_every 50000 \
    --log_every 50 \
    --test_every 500
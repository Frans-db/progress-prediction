#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=1:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8192
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

srun python /home/nfs/fransdeboer/mscfransdeboer/code/main.py \
    --seed 42 \
    --wandb_project progressnet \
    --wandb_tags progressnet lstm ucf24 \
    --dataset ucf24 \
    --data_dir features/i3d_embeddings \
    --feature_dim 1024 \
    --indices \
    --indices_normalizer 128 \
    --train_split train_tubes.txt \
    --test_split test_tubes.txt \
    --batch_size 1 \
    --iterations 100000 \
    --network progressnet \
    --optimizer adam \
    --loss l2 \
    --lr 1e-4 \
    --dropout_chance 0.5 \
    --weight_decay 0 \
    --lr_decay 1.0 \
    --lr_decay_every 10000 \
    --log_every 250 \
    --test_every 1000
#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=0:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

srun python /home/nfs/fransdeboer/mscfransdeboer/code/main.py \
    --seed 42 \
    --experiment_name rsd_bars \
    --wandb_group normal \
    --dataset bars \
    --data_dir features/resnet18 \
    --feature_dim 512 \
    --rsd_type minutes \
    --fps 1 \
    --rsd_normalizer 5 \
    --train_split train.txt \
    --test_split test.txt \
    --batch_size 1 \
    --iterations 30000 \
    --network rsdnet \
    --dropout_chance 0.3 \
    --lr_decay 0.1 \
    --lr_decay_every 10000 \
    --log_every 100 \
    --test_every 1000
#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=32:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

srun python /home/nfs/fransdeboer/mscfransdeboer/code/main.py \
    --seed 42 \
    --experiment_name ute_ucf24 \
    --dataset ucf24 \
    --data_dir features/resnet152 \
    --flat \
    --train_split train_tubes.txt \
    --test_split test_tubes.txt \
    --batch_size 256 \
    --iterations 50000 \
    --network ute \
    --lr 1e-4 \
    --lr_decay 1.0 \
    --lr_decay_every 1000000 \
    --log_every 100 \
    --test_every 1000
#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=14:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

srun python /home/nfs/fransdeboer/mscfransdeboer/code/main.py \
    --seed 42 \
    --experiment_name resnet152_ucf_24 \
    --dataset ucf24 \
    --data_dir rgb-images \
    --flat \
    --train_split train.txt \
    --test_split test.txt \
    --batch_size 48 \
    --iterations 50000 \
    --network rsdnet_flat \
    --backbone resnet152 \
    --load_backbone resnet152.pth \
    --dropout_chance 0.0 \
    --optimizer sgd \
    --loss smooth_l1 \
    --momentum 0.9 \
    --lr 1e-3 \
    --weight_decay 5e-4 \
    --lr_decay 0.1 \
    --lr_decay_every 20000 \
    --log_every 100 \
    --test_every 2500
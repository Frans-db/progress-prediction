#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

srun python /home/nfs/fransdeboer/mscfransdeboer/code/main.py \
    --seed 42 \
    --dataset ucf24 \
    --data_dir rgb-images \
    --train_split all.txt \
    --test_split test.txt \
    --batch_size 1 \
    --network rsdnet_flat \
    --backbone resnet152 \
    --load_backbone resnet152.pth \
    --load_experiment resnet152_ucf24 \
    --load_iteration 50000 \
    --embed \
    --embed_batch_size 32 \
    --embed_dir features/resnet152 \
    --num_workers 1
    
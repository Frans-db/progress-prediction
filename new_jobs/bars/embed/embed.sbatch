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
    --dataset bars \
    --data_dir rgb-images \
    --train_split all.txt \
    --test_split test.txt \
    --batch_size 1 \
    --network rsdnet_flat \
    --backbone resnet18 \
    --load_backbone resnet18.pth \
    --load_experiment bars_resnet \
    --load_iteration 50000 \
    --embed \
    --embed_batch_size 32 \
    --embed_dir features/resnet18 \
    --num_workers 1
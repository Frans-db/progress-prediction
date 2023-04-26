#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=14:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=3072
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

srun python /home/nfs/fransdeboer/temp/code/main.py \
    --data_root /tudelft.net/staff-umbrella/StudentsCVlab/fransdeboer/ \
    --experiment_name resnet_fold3 \
    --wandb_name resnet_fold3 \
    --network rsdflat \
    --dataset cholec80 \
    --dataset_type images \
    --train_split t1_3.txt \
    --test_split v_3.txt \
    --backbone resnet152 \
    --backbone_name resnet152_v2.pth \
    --backbone_channels 2048 \
    --resize 224 \
    --iterations 50000 \
    --test_every 5000 \
    --log_every 50 \
    --batch_size 48 \
    --optimizer sgd \
    --momentum 0.9 \
    --lr 1e-3 \
    --lr_decay 0.1 \
    --lr_decay_every 20000 \
    --weight_decay 5e-4
#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8192
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:a40

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

srun python /home/nfs/fransdeboer/mscfransdeboer/code/main.py \
    --data_root /tudelft.net/staff-umbrella/StudentsCVlab/fransdeboer/ \
    --experiment_name 1024_34 \
    --wandb_name 1024_34 \
    --roi_size 2 \
    --roi_scale 0.0633 \
    --backbone vgg1024 \
    --backbone_name vgg_1024.pth \
    --backbone_depth 34 \
    --backbone_channels 1024 \
    --resize 300 \
    --iterations 1000000 \
    --test_every 2500

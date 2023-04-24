#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8192
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:a40

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

srun python /home/nfs/fransdeboer/mscfransdeboer/code/main.py \
    --data_root /tudelft.net/staff-umbrella/StudentsCVlab/fransdeboer/ \
    --experiment_name 512_30 \
    --wandb_name 512_30 \
    --roi_size 3 \
    --roi_scale 7/224 \
    --backbone vgg512 \
    --backbone_name vgg_512.pth \
    --backbone_depth 30 \
    --backbone_channels 512 \
    --resize 224 \
    --iterations 1000000 \
    --test_every 2500 \

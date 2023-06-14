#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=14:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

srun python /home/nfs/fransdeboer/mscfransdeboer/code/i3d_embeddings.py \
    --model_path /tudelft.net/staff-umbrella/StudentsCVlab/fransdeboer/models/rgb_imagenet.pth \
    --root /tudelft.net/staff-umbrella/StudentsCVlab/fransdeboer \
    --dataset breakfast \
    --splitfile all.txt \
    --save_dir features/i3d_embeddings_sampled \
    --subsample_fps 10
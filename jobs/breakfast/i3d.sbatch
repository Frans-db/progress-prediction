#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=3G
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu
#SBATCH --array=0,1,2,3

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

srun python /home/nfs/fransdeboer/mscfransdeboer/code/i3d_embedding.py \
    --model_path /tudelft.net/staff-umbrella/StudentsCVlab/fransdeboer/models/rgb_imagenet.pth \
    --root /tudelft.net/staff-umbrella/StudentsCVlab/fransdeboer \
    --dataset breakfast \
    --splitfile all.txt
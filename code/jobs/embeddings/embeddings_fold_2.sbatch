#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=3072
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

srun python /home/nfs/fransdeboer/thesis/code/rsd_embed.py \
    --data_root /tudelft.net/staff-umbrella/StudentsCVlab/fransdeboer/ \
    --batch_size 32 \
    --num_workers 4 \
    --load_experiment resnet_fold2 \
    --load_iteration 50000 \
    --backbone resnet152 \
    --backbone_channels 2048 \
    --target_directory fold2_embeddings
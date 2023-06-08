#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu
#SBATCH --array=0,1,2,3

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

srun python /home/nfs/fransdeboer/mscfransdeboer/code/main.py \
    --seed 42 \
    --dataset cholec80 \
    --data_dir rgb-images \
    --train_split all_${SLURM_ARRAY_TASK_ID}.txt \
    --test_split v_${SLURM_ARRAY_TASK_ID}.txt \
    --batch_size 1 \
    --network rsdnet_flat \
    --backbone resnet152 \
    --load_backbone resnet152.pth \
    --load_experiment resnet152_cholec80_sampled_${SLURM_ARRAY_TASK_ID} \
    --load_iteration 50000 \
    --embed \
    --embed_batch_size 32 \
    --embed_dir features/resnet152_sampled_${SLURM_ARRAY_TASK_ID} \
    --num_workers 1 \
    --subsample_fps 10
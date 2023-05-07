#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8192
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu
#SBATCH --array=1,2,3,4

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

srun python /home/nfs/fransdeboer/mscfransdeboer/code/main.py \
    --seed 42 \
    --wandb_project rsdnet \
    --wandb_tags rsdnet lstm breakfast tea \
    --dataset breakfast \
    --data_dir features/dense_trajectories \
    --feature_dim 64 \
    --rsd_type minutes \
    --fps 1 \
    --rsd_normalizer 1 \
    --indices \
    --indices_normalizer 716 \
    --train_split train_tea_s${SLURM_ARRAY_TASK_ID}.txt \
    --test_split test_tea_s${SLURM_ARRAY_TASK_ID}.txt \
    --batch_size 1 \
    --iterations 30000 \
    --network rsdnet \
    --optimizer sgd \
    --loss smooth_l1 \
    --momentum 0.9 \
    --lr 1e-3 \
    --dropout_chance 0.3 \
    --weight_decay 1e-2 \
    --lr_decay 0.1 \
    --lr_decay_every 10000 \
    --log_every 250 \
    --test_every 1000
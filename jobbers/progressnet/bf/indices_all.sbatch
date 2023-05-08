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
    --wandb_tags rsdnet lstm breakfast cereals \
    --dataset breakfast \
    --data_dir features/dense_trajectories \
    --feature_dim 64 \
    --train_split train_s${SLURM_ARRAY_TASK_ID}.txt \
    --test_split test_s${SLURM_ARRAY_TASK_ID}.txt \
    --indices \
    --indices_normalizer 2096 \
    --batch_size 1 \
    --iterations 30000 \
    --network progressnet \
    --optimizer adam \
    --loss l2 \
    --lr 1e-4 \
    --dropout_chance 0.5 \
    --weight_decay 0 \
    --lr_decay 1.0 \
    --lr_decay_every 10000 \
    --log_every 250 \
    --test_every 1000
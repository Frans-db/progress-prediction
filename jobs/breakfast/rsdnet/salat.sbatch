#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu
#SBATCH --array=1,2,3,4

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

srun python /home/nfs/fransdeboer/mscfransdeboer/code/main.py \
    --seed 42 \
    --experiment_name rsdnet_bf_salat_${SLURM_ARRAY_TASK_ID} \
    --wandb_tags fold_s${SLURM_ARRAY_TASK_ID} salat \
    --dataset breakfast \
    --data_dir features/resnet152_salat_${SLURM_ARRAY_TASK_ID} \
    --train_split train_salat_s${SLURM_ARRAY_TASK_ID}.txt \
    --test_split test_salat_s${SLURM_ARRAY_TASK_ID}.txt \
    --feature_dim 2048 \
    --rsd_type minutes \
    --fps 1 \
    --rsd_normalizer 5 \
    --batch_size 1 \
    --iterations 30000 \
    --network rsdnet \
    --dropout_chance 0.3 \
    --optimizer sgd \
    --loss smooth_l1 \
    --momentum 0.9 \
    --lr 1e-3 \
    --weight_decay 5e-4 \
    --lr_decay 0.1 \
    --lr_decay_every 10000 \
    --log_every 50 \
    --test_every 500
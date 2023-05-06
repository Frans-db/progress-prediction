#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8192
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu
#SBATCH --array=0,1,2,3

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

srun python /home/nfs/fransdeboer/mscfransdeboer/code/main.py \
    --seed 42 \
    --experiment_name i3d_lstm_fold_${SLURM_ARRAY_TASK_ID} \
    --wandb_project rsdnet \
    --wandb_name i3d_lstm_fold_${SLURM_ARRAY_TASK_ID} \
    --wandb_tags rsdnet lstm cholec80 \
    --dataset cholec80 \
    --data_dir features/i3d_embeddings \
    --feature_dim 2048 \
    --rsd_type minutes \
    --fps 1 \
    --rsd_normalizer 5 \
    --train_split t12_p${SLURM_ARRAY_TASK_ID}.txt \
    --test_split v_p${SLURM_ARRAY_TASK_ID}.txt \
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
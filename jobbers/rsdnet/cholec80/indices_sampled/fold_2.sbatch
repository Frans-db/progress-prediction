#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8192
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

srun python /home/nfs/fransdeboer/mscfransdeboer/code/main.py \
    --seed 42 \
    --experiment_name rsd_lstm_fold_2_indices_sampled \
    --wandb_project rsdnet \
    --wandb_name rsd_lstm_fold_2_indices_sampled \
    --wandb_tags rsdnet lstm cholec80 indices \
    --dataset cholec80 \
    --data_dir features/i3d_embeddings \
    --feature_dim 1024 \
    --rsd_type minutes \
    --fps 1 \
    --rsd_normalizer 5 \
    --subsample \
    --train_split t12_p2.txt \
    --test_split v_p2.txt \
    --batch_size 1 \
    --indices \
    --indices_normalizer 2332 \
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
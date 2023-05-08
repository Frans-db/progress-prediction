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
    --experiment_name rsd_lstm_fold_${SLURM_ARRAY_TASK_ID} \
    --wandb_project rsdnet \
    --wandb_name rsd_lstm_fold_${SLURM_ARRAY_TASK_ID} \
    --wandb_tags rsdnet lstm cholec80 \
    --dataset cholec80 \
    --data_dir features/rsd_embeds_${SLURM_ARRAY_TASK_ID} \
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

python /home/nfs/fransdeboer/mscfransdeboer/code/main.py \
    --seed 42 \
    --wandb_project rsdnet \
    --dataset cholec80 \
    --feature_dim 2048 \
    --rsd_type minutes \
    --fps 1 \
    --rsd_normalizer 5 \
    --batch_size 1 \
    --network rsdnet \
    --eval \
    --load_iteration 30000 \
    --load_experiment rsd_lstm_fold_2 \
    --data_dir features/rsd_embeds_2 \
    --train_split v_p2.txt \
    --test_split e_p2.txt
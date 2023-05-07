#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=15:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16384
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu
#SBATCH --array=0,1,2,3

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

srun python /home/nfs/fransdeboer/mscfransdeboer/code/main.py \
    --seed 42 \
    --experiment_name rsd_cnn_fold_${SLURM_ARRAY_TASK_ID} \
    --wandb_project rsdnet \
    --wandb_name rsd_cnn_fold_${SLURM_ARRAY_TASK_ID} \
    --wandb_tags rsdnet cnn cholec80 \
    --dataset cholec80 \
    --data_dir rgb-images \
    --flat \
    --train_split t1_p${SLURM_ARRAY_TASK_ID}.txt \
    --test_split v_p${SLURM_ARRAY_TASK_ID}.txt \
    --batch_size 48 \
    --iterations 50000 \
    --network rsdnet \
    --backbone resnet152 \
    --load_backbone resnet152.pth \
    --dropout_chance 0.0 \
    --optimizer sgd \
    --loss smooth_l1 \
    --momentum 0.9 \
    --lr 1e-3 \
    --weight_decay 5e-4 \
    --lr_decay 0.1 \
    --lr_decay_every 20000 \
    --log_every 250 \
    --test_every 1000
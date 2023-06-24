#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:a40
#SBATCH --array=1,2,3,4

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

srun python /home/nfs/fransdeboer/mscfransdeboer/code/main.py \
    --seed 42 \
    --experiment_name pn_bf_indices_${SLURM_ARRAY_TASK_ID} \
    --wandb_group indices \
    --wandb_tags fold_${SLURM_ARRAY_TASK_ID} \
    --dataset breakfast \
    --data_dir rgb-images \
    --train_split train_s${SLURM_ARRAY_TASK_ID}.txt \
    --test_split test_s${SLURM_ARRAY_TASK_ID}.txt \
    --batch_size 1 \
    --iterations 50000 \
    --network progressnet \
    --backbone vgg16 \
    --load_backbone vgg16.pth \
    --dropout_chance 0.5 \
    --lr_decay 1.0 \
    --lr_decay_every 1000000 \
    --log_every 100 \
    --test_every 5000 \
    --num_workers 2 \
    --subsample \
    --max_length 550 \
    --indices \
    --indices_normalizer 2162 \
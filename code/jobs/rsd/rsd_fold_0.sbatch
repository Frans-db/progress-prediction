#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8192
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

srun python /home/nfs/fransdeboer/thesis/code/rsd.py \
    --data_root /tudelft.net/staff-umbrella/StudentsCVlab/fransdeboer/ \
    --wandb_name lstm_fold0 \
    --experiment_name lstm_fold0 \
    --network rsdnet \
    --dataset cholec80 \
    --dataset_type features \
    --data_type fold0_embeddings \
    --train_split t1_t2_0.txt \
    --test_split e_0.txt \
    --backbone_channels 2048 \
    --embedding_size 512 \
    --iterations 30000 \
    --test_every 500 \
    --log_every 25 \
    --optimizer sgd \
    --momentum 0.9 \
    --lr 1e-3 \
    --lr_decay 0.1 \
    --lr_decay_every 10000 \
    --weight_decay 1e-2 \
    --dropout_chance 0.3
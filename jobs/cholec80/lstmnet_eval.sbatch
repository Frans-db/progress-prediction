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
    --dataset cholec80 \
    --data_dir features/resnet152_${SLURM_ARRAY_TASK_ID} \
    --feature_dim 2048 \
    --train_split t12_p${SLURM_ARRAY_TASK_ID}.txt \
    --test_split e_p${SLURM_ARRAY_TASK_ID}.txt \
    --batch_size 1 \
    --network lstmnet \
    --load_experiment lstmnet_cholec80_${SLURM_ARRAY_TASK_ID} \
    --load_iteration 30000 \
    --eval
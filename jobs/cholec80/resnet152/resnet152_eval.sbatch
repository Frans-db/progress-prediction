#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu
#SBATCH --array=0,1,2,3

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

srun python /home/nfs/fransdeboer/mscfransdeboer/code/main.py \
    --seed 42 \
    --dataset cholec80 \
    --data_dir rgb-images \
    --flat \
    --train_split t1_p${SLURM_ARRAY_TASK_ID}.txt \
    --test_split e_p${SLURM_ARRAY_TASK_ID}.txt \
    --batch_size 32 \
    --network rsdnet_flat \
    --backbone resnet152 \
    --load_experiment resnet152_cholec80_${SLURM_ARRAY_TASK_ID} \
    --load_iteration 10000 \
    --eval
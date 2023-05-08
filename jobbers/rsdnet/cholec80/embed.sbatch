#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=4:00:00
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
    --dataset cholec80 \
    --data_dir rgb-images \
    --train_split all_p${SLURM_ARRAY_TASK_ID}.txt \
    --test_split v_p${SLURM_ARRAY_TASK_ID}.txt \
    --batch_size 1 \
    --network rsdnet \
    --backbone resnet152 \
    --load_experiment rsd_cnn_fold_${SLURM_ARRAY_TASK_ID} \
    --load_iteration 50000 \
    --num_workers 1 \
    --embed \
    --embed_batch_size 25 \
    --embed_dir features/rsd_embeds_${SLURM_ARRAY_TASK_ID}
#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=5120
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
    --train_split v_p0.txt \
    --test_split e_p0.txt \
    --batch_size 48 \
    --network rsdnet \
    --backbone resnet152 \
    --load_experiment rsd_cnn_fold_0 \
    --load_iteration 40000 \
    --eval 

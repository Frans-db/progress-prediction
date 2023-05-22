#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:a40

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

srun python /home/nfs/fransdeboer/mscfransdeboer/code/main.py --wandb_project cholec80_tests --dataset cholec80 --data_dir rgb-images --train_split t12_p0.txt --test_split v_p0.txt --batch_size 1 --iterations 5000 --network resnet --backbone resnet18 --dropout_chance 0.3 --log_every 25 --test_every 100 --subsample --num_workers 1 --experiment_name cholec80_resnet18
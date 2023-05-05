#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=5120
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu
#SBATCH --array=0-5

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

srun python -m wandb agent mscfransdeboer/progressnet/q3qxrxz2
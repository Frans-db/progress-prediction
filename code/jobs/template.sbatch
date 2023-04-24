#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8192
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:a40

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

srun python /home/nfs/fransdeboer/mscfransdeboer/code/main.py --data_root /tudelft.net/staff-umbrella/StudentsCVlab/fransdeboer/ --experiment_name batch_4 --wandb_name batch_4 --train_split train_telic.txt --test_split test_telic.txt --bounding_boxes --iterations 1000000 --lr 1e-4 --test_every 2500 --pooling_layers 3 6 9 --roi_size 3 --basemodel vgg512 --basemodel_name vgg_512.pth --batch_size 4
srun python main.py \
    --experiment_name batch_4 \
    --wandb_name batch_4 \
    --train_split train_telic.txt \
    --test_split test_telic.txt \
    --iterations 1000000 \
    --lr 1e-4 \
    --test_every 2500 \
    --pooling_layers 1 2 3 \
    --roi_size 3 \
    --backbone vgg512 \
    --backbone_name vgg_512.pth \
    --batch_size 4
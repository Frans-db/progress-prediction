# show embedding acts random or collapses
python main.py --seed 42 --wandb_group default
python main.py --seed 43 --wandb_group default
python main.py --seed 44 --wandb_group default
python main.py --seed 45 --wandb_group default
python main.py --seed 46 --wandb_group default

python main.py --seed 42 --wandb_group augmented --augmentations subsample subsection removal
python main.py --seed 43 --wandb_group augmented --augmentations subsample subsection removal
python main.py --seed 44 --wandb_group augmented --augmentations subsample subsection removal
python main.py --seed 45 --wandb_group augmented --augmentations subsample subsection removal
python main.py --seed 46 --wandb_group augmented --augmentations subsample subsection removal

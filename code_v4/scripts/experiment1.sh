python main.py --seed 42 --wandb_group default --wandb_tags experiment1
python main.py --seed 43 --wandb_group default --wandb_tags experiment1
python main.py --seed 44 --wandb_group default --wandb_tags experiment1

python main.py --seed 42 --wandb_group forecast --losses progress forecast embedding --wandb_tags experiment1
python main.py --seed 43 --wandb_group forecast --losses progress forecast embedding --wandb_tags experiment1
python main.py --seed 44 --wandb_group forecast --losses progress forecast embedding --wandb_tags experiment1

python main.py --seed 42 --wandb_group augmented --augmentations subsample subsection removal --delta_t 1 --wandb_tags experiment1
python main.py --seed 43 --wandb_group augmented --augmentations subsample subsection removal --delta_t 1 --wandb_tags experiment1
python main.py --seed 44 --wandb_group augmented --augmentations subsample subsection removal --delta_t 1 --wandb_tags experiment1
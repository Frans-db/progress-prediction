# show embedding acts random or collapses
python main.py --seed 42 --group default
python main.py --seed 43 --group default
python main.py --seed 44 --group default
python main.py --seed 45 --group default
python main.py --seed 46 --group default

python main.py --seed 42 --group augmented --augmentations subsample subsection removal
python main.py --seed 43 --group augmented --augmentations subsample subsection removal
python main.py --seed 44 --group augmented --augmentations subsample subsection removal
python main.py --seed 45 --group augmented --augmentations subsample subsection removal
python main.py --seed 46 --group augmented --augmentations subsample subsection removal

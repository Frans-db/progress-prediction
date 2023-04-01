python main.py --dataset toy_difficult --delta_t 14 --seed 42 --group default
python main.py --dataset toy_difficult --delta_t 14 --seed 43 --group default
python main.py --dataset toy_difficult --delta_t 14 --seed 44 --group default
python main.py --dataset toy_difficult --delta_t 14 --seed 45 --group default
python main.py --dataset toy_difficult --delta_t 14 --seed 46 --group default

python main.py --dataset toy_difficult --forecast --delta_t 14 --seed 42 --group forecast
python main.py --dataset toy_difficult --forecast --delta_t 14 --seed 43 --group forecast
python main.py --dataset toy_difficult --forecast --delta_t 14 --seed 44 --group forecast
python main.py --dataset toy_difficult --forecast --delta_t 14 --seed 45 --group forecast
python main.py --dataset toy_difficult --forecast --delta_t 14 --seed 46 --group forecast

python main.py --dataset toy_difficult --delta_t 14 --seed 42 --augmentations subsection subsample removal --group augmentations
python main.py --dataset toy_difficult --delta_t 14 --seed 43 --augmentations subsection subsample removal --group augmentations
python main.py --dataset toy_difficult --delta_t 14 --seed 44 --augmentations subsection subsample removal --group augmentations
python main.py --dataset toy_difficult --delta_t 14 --seed 45 --augmentations subsection subsample removal --group augmentations
python main.py --dataset toy_difficult --delta_t 14 --seed 46 --augmentations subsection subsample removal --group augmentations
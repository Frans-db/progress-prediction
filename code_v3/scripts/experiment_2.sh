# show effect of subsection augmentation
python main.py --dataset toy --seed 42 --group default --network pooled_progressnet --augmentations subsection
python main.py --dataset toy --seed 43 --group default --network pooled_progressnet --augmentations subsection
python main.py --dataset toy --seed 44 --group default --network pooled_progressnet --augmentations subsection

python main.py --dataset toy --seed 42 --group forecast --losses forecast --network pooled_progressnet --augmentations subsection
python main.py --dataset toy --seed 43 --group forecast --losses forecast --network pooled_progressnet --augmentations subsection
python main.py --dataset toy --seed 44 --group forecast --losses forecast --network pooled_progressnet --augmentations subsection

python main.py --dataset toy --seed 42 --group forecast_embedding --losses forecast embedding --network pooled_progressnet --augmentations subsection
python main.py --dataset toy --seed 43 --group forecast_embedding --losses forecast embedding --network pooled_progressnet --augmentations subsection
python main.py --dataset toy --seed 44 --group forecast_embedding --losses forecast embedding --network pooled_progressnet --augmentations subsection

python main.py --dataset toy --seed 42 --group embedding --losses embedding --network pooled_progressnet --augmentations subsection
python main.py --dataset toy --seed 43 --group embedding --losses embedding --network pooled_progressnet --augmentations subsection
python main.py --dataset toy --seed 44 --group embedding --losses embedding --network pooled_progressnet --augmentations subsection

python main.py --dataset toy --seed 42 --no_wandb --losses embedding --network pooled_progressnet --augmentations subsection --plots --plot_directory experiment_2
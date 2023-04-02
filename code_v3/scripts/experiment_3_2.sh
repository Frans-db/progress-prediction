# toy 2, pooled progressnet
python main.py --dataset toy_2 --seed 42 --group default --network pooled_progressnet
python main.py --dataset toy_2 --seed 43 --group default --network pooled_progressnet
python main.py --dataset toy_2 --seed 44 --group default --network pooled_progressnet

python main.py --dataset toy_2 --seed 42 --group forecast --losses forecast --network pooled_progressnet
python main.py --dataset toy_2 --seed 43 --group forecast --losses forecast --network pooled_progressnet
python main.py --dataset toy_2 --seed 44 --group forecast --losses forecast --network pooled_progressnet

python main.py --dataset toy_2 --seed 42 --group forecast_embedding --losses forecast embedding --network pooled_progressnet
python main.py --dataset toy_2 --seed 43 --group forecast_embedding --losses forecast embedding --network pooled_progressnet
python main.py --dataset toy_2 --seed 44 --group forecast_embedding --losses forecast embedding --network pooled_progressnet

python main.py --dataset toy_2 --seed 42 --no_wandb --losses forecast embedding --network pooled_progressnet --test_every 250 --plots --plots_dir experiment_3_2
# toy 2, pooled progressnet
python main.py --dataset toy_start_info --seed 42 --group default --network pooled_progressnet --test_every 25
python main.py --dataset toy_start_info --seed 43 --group default --network pooled_progressnet --test_every 25
python main.py --dataset toy_start_info --seed 44 --group default --network pooled_progressnet --test_every 25
python main.py --dataset toy_start_info --seed 45 --group default --network pooled_progressnet --test_every 25
python main.py --dataset toy_start_info --seed 46 --group default --network pooled_progressnet --test_every 25

python main.py --dataset toy_start_info --seed 42 --group forecast --losses forecast --network pooled_progressnet --test_every 25
python main.py --dataset toy_start_info --seed 43 --group forecast --losses forecast --network pooled_progressnet --test_every 25
python main.py --dataset toy_start_info --seed 44 --group forecast --losses forecast --network pooled_progressnet --test_every 25
python main.py --dataset toy_start_info --seed 45 --group forecast --losses forecast --network pooled_progressnet --test_every 25
python main.py --dataset toy_start_info --seed 46 --group forecast --losses forecast --network pooled_progressnet --test_every 25

python main.py --dataset toy_start_info --seed 42 --group forecast_embedding --losses forecast embedding --network pooled_progressnet --test_every 25
python main.py --dataset toy_start_info --seed 43 --group forecast_embedding --losses forecast embedding --network pooled_progressnet --test_every 25
python main.py --dataset toy_start_info --seed 44 --group forecast_embedding --losses forecast embedding --network pooled_progressnet --test_every 25
python main.py --dataset toy_start_info --seed 45 --group forecast_embedding --losses forecast embedding --network pooled_progressnet --test_every 25
python main.py --dataset toy_start_info --seed 46 --group forecast_embedding --losses forecast embedding --network pooled_progressnet --test_every 25

python main.py --dataset toy_start_info --seed 42 --no_wandb --losses forecast embedding --network pooled_progressnet --test_every 250 --plots --plot_directory experiment_3_3
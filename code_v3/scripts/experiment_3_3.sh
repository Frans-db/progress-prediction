# toy 2, pooled progressnet
python main.py --dataset toy_start_info --seed 45 --group default --network pooled_progressnet --test_every 10
python main.py --dataset toy_start_info --seed 46 --group default --network pooled_progressnet --test_every 10
python main.py --dataset toy_start_info --seed 47 --group default --network pooled_progressnet --test_every 10
python main.py --dataset toy_start_info --seed 48 --group default --network pooled_progressnet --test_every 10
python main.py --dataset toy_start_info --seed 49 --group default --network pooled_progressnet --test_every 10

python main.py --dataset toy_start_info --seed 45 --group forecast --losses forecast --network pooled_progressnet --test_every 10
python main.py --dataset toy_start_info --seed 46 --group forecast --losses forecast --network pooled_progressnet --test_every 10
python main.py --dataset toy_start_info --seed 47 --group forecast --losses forecast --network pooled_progressnet --test_every 10
python main.py --dataset toy_start_info --seed 48 --group forecast --losses forecast --network pooled_progressnet --test_every 10
python main.py --dataset toy_start_info --seed 49 --group forecast --losses forecast --network pooled_progressnet --test_every 10

python main.py --dataset toy_start_info --seed 45 --group forecast_embedding --losses forecast embedding --network pooled_progressnet --test_every 10
python main.py --dataset toy_start_info --seed 46 --group forecast_embedding --losses forecast embedding --network pooled_progressnet --test_every 10
python main.py --dataset toy_start_info --seed 47 --group forecast_embedding --losses forecast embedding --network pooled_progressnet --test_every 10
python main.py --dataset toy_start_info --seed 48 --group forecast_embedding --losses forecast embedding --network pooled_progressnet --test_every 10
python main.py --dataset toy_start_info --seed 49 --group forecast_embedding --losses forecast embedding --network pooled_progressnet --test_every 10

python main.py --dataset toy_start_info --seed 45 --no_wandb --losses forecast embedding --network pooled_progressnet --test_every 250 --plots --plot_directory experiment_3_2
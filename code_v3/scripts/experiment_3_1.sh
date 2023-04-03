# toy shuffle, pooled progressnet
python main.py --dataset toy_shuffle --seed 42 --group default --network pooled_progressnet --test_every 25
python main.py --dataset toy_shuffle --seed 43 --group default --network pooled_progressnet --test_every 25
python main.py --dataset toy_shuffle --seed 44 --group default --network pooled_progressnet --test_every 25

python main.py --dataset toy_shuffle --seed 42 --group forecast --losses forecast --network pooled_progressnet --test_every 25
python main.py --dataset toy_shuffle --seed 43 --group forecast --losses forecast --network pooled_progressnet --test_every 25
python main.py --dataset toy_shuffle --seed 44 --group forecast --losses forecast --network pooled_progressnet --test_every 25

python main.py --dataset toy_shuffle --seed 42 --group forecast_embedding --losses forecast embedding --network pooled_progressnet --test_every 25
python main.py --dataset toy_shuffle --seed 43 --group forecast_embedding --losses forecast embedding --network pooled_progressnet --test_every 25
python main.py --dataset toy_shuffle --seed 44 --group forecast_embedding --losses forecast embedding --network pooled_progressnet --test_every 25

python main.py --dataset toy_shuffle --seed 42 --no_wandb --losses forecast embedding --network pooled_progressnet --test_every 250 --plots --plot_directory experiment_3_1
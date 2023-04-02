# toy shuffle, pooled progressnet
python main.py --dataset toy_shuffle --seed 42 --group default --network pooled_progressnet
python main.py --dataset toy_shuffle --seed 43 --group default --network pooled_progressnet
python main.py --dataset toy_shuffle --seed 44 --group default --network pooled_progressnet

python main.py --dataset toy_shuffle --seed 42 --group forecast --losses forecast --network pooled_progressnet
python main.py --dataset toy_shuffle --seed 43 --group forecast --losses forecast --network pooled_progressnet
python main.py --dataset toy_shuffle --seed 44 --group forecast --losses forecast --network pooled_progressnet

python main.py --dataset toy_shuffle --seed 42 --group forecast_embedding --losses forecast embedding --network pooled_progressnet
python main.py --dataset toy_shuffle --seed 43 --group forecast_embedding --losses forecast embedding --network pooled_progressnet
python main.py --dataset toy_shuffle --seed 44 --group forecast_embedding --losses forecast embedding --network pooled_progressnet

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
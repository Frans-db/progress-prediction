# show embedding acts random or collapses
# filter: progressnet
python main.py --dataset toy --seed 42 --group default
python main.py --dataset toy --seed 43 --group default
python main.py --dataset toy --seed 44 --group default

python main.py --dataset toy --seed 42 --group forecast --losses forecast
python main.py --dataset toy --seed 43 --group forecast --losses forecast
python main.py --dataset toy --seed 44 --group forecast --losses forecast

python main.py --dataset toy --seed 42 --group forecast_embedding --losses forecast embedding
python main.py --dataset toy --seed 43 --group forecast_embedding --losses forecast embedding
python main.py --dataset toy --seed 44 --group forecast_embedding --losses forecast embedding

python main.py --dataset toy --seed 42 --group embedding --losses embedding
python main.py --dataset toy --seed 43 --group embedding --losses embedding
python main.py --dataset toy --seed 44 --group embedding --losses embedding

# show static embedding can be learnt
# filter pooled_progressnet, no augmentations
python main.py --dataset toy --seed 42 --group default --network pooled_progressnet
python main.py --dataset toy --seed 43 --group default --network pooled_progressnet
python main.py --dataset toy --seed 44 --group default --network pooled_progressnet

python main.py --dataset toy --seed 42 --group forecast --losses forecast --network pooled_progressnet
python main.py --dataset toy --seed 43 --group forecast --losses forecast --network pooled_progressnet
python main.py --dataset toy --seed 44 --group forecast --losses forecast --network pooled_progressnet

python main.py --dataset toy --seed 42 --group forecast_embedding --losses forecast embedding --network pooled_progressnet
python main.py --dataset toy --seed 43 --group forecast_embedding --losses forecast embedding --network pooled_progressnet
python main.py --dataset toy --seed 44 --group forecast_embedding --losses forecast embedding --network pooled_progressnet

# plots on the first one
python main.py --dataset toy --seed 42 --group embedding --losses embedding --network pooled_progressnet 
python main.py --dataset toy --seed 43 --group embedding --losses embedding --network pooled_progressnet
python main.py --dataset toy --seed 44 --group embedding --losses embedding --network pooled_progressnet

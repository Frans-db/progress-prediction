# toy 2, pooled progressnet
python main.py --dataset ucf24 --seed 42 --group default --network pooled_progressnet --test_every 150 --delta_t 15 --iterations 15000
python main.py --dataset ucf24 --seed 42 --group forecast --losses forecast --network pooled_progressnet --test_every 150 --delta_t 15 --iterations 15000
python main.py --dataset ucf24 --seed 42 --group forecast_embedding --losses forecast embedding --network pooled_progressnet --test_every 150 --delta_t 15 --iterations 15000

python main.py --dataset ucf24 --seed 43 --group default --network pooled_progressnet --test_every 150 --delta_t 15 --iterations 15000
python main.py --dataset ucf24 --seed 43 --group forecast --losses forecast --network pooled_progressnet --test_every 150 --delta_t 15 --iterations 15000
python main.py --dataset ucf24 --seed 43 --group forecast_embedding --losses forecast embedding --network pooled_progressnet --test_every 150 --delta_t 15 --iterations 15000

python main.py --dataset ucf24 --seed 44 --group default --network pooled_progressnet --test_every 150 --delta_t 15 --iterations 15000
python main.py --dataset ucf24 --seed 44 --group forecast --losses forecast --network pooled_progressnet --test_every 150 --delta_t 15 --iterations 15000
python main.py --dataset ucf24 --seed 44 --group forecast_embedding --losses forecast embedding --network pooled_progressnet --test_every 150 --delta_t 15 --iterations 15000
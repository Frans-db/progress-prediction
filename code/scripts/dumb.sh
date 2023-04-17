for seed in "$@"
do
python main.py --network dumb_static --seed $seed --wandb_group "static 0.5"
python main.py --network dumb_random --seed $seed --wandb_group "random"

python main.py --seed $seed --network dumb_static --embedding_size 2048  --train_set ucf24 --test_set ucf24 --data_type rgb-images --bounding_boxes --iterations 10000 --test_every 1000 --wandb_group "static 0.5"
python main.py --seed $seed --network dumb_random --embedding_size 2048  --train_set ucf24 --test_set ucf24 --data_type rgb-images --bounding_boxes --iterations 10000 --test_every 1000 --wandb_group "random"
done
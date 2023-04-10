for seed in "$@"
do
    python main.py --train_set toy_shuffle_speediest --test_set toy_shuffle_speedier --seed $seed --wandb_group forecast_embedding_1 --losses progress forecast embedding --wandb_tags experiment4 --delta_t 1
    python main.py --train_set toy_shuffle_speediest --test_set toy_shuffle_speedier --seed $seed --wandb_group forecast_1 --losses progress forecast --wandb_tags experiment4 --delta_t 1

    python main.py --train_set toy_shuffle_speediest --test_set toy_shuffle_speedier --seed $seed --wandb_group forecast_embedding_5 --losses progress forecast embedding --wandb_tags experiment4 --delta_t 5
    python main.py --train_set toy_shuffle_speediest --test_set toy_shuffle_speedier --seed $seed --wandb_group forecast_5 --losses progress forecast --wandb_tags experiment4 --delta_t 5

    python main.py --train_set toy_shuffle_speediest --test_set toy_shuffle_speedier --seed $seed --wandb_group forecast_embedding_20 --losses progress forecast embedding --wandb_tags experiment4 --delta_t 20
    python main.py --train_set toy_shuffle_speediest --test_set toy_shuffle_speedier --seed $seed --wandb_group forecast_20 --losses progress forecast --wandb_tags experiment4 --delta_t 20

    python main.py --train_set toy_shuffle_speediest --test_set toy_shuffle_speedier --seed $seed --wandb_group forecast_embedding_30 --losses progress forecast embedding --wandb_tags experiment4 --delta_t 30
    python main.py --train_set toy_shuffle_speediest --test_set toy_shuffle_speedier --seed $seed --wandb_group forecast_30 --losses progress forecast --wandb_tags experiment4 --delta_t 30

    python main.py --train_set toy_shuffle_speediest --test_set toy_shuffle_speedier --seed $seed --wandb_group forecast_embedding_40 --losses progress forecast embedding --wandb_tags experiment4 --delta_t 40
    python main.py --train_set toy_shuffle_speediest --test_set toy_shuffle_speedier --seed $seed --wandb_group forecast_40 --losses progress forecast --wandb_tags experiment4 --delta_t 40

    python main.py --train_set toy_shuffle_speediest --test_set toy_shuffle_speedier --seed $seed --wandb_group forecast_embedding_50 --losses progress forecast embedding --wandb_tags experiment4 --delta_t 50
    python main.py --train_set toy_shuffle_speediest --test_set toy_shuffle_speedier --seed $seed --wandb_group forecast_50 --losses progress forecast --wandb_tags experiment4 --delta_t 50

    python main.py --train_set toy_shuffle_speediest --test_set toy_shuffle_speedier --seed $seed --wandb_group forecast_embedding_60 --losses progress forecast embedding --wandb_tags experiment4 --delta_t 60
    python main.py --train_set toy_shuffle_speediest --test_set toy_shuffle_speedier --seed $seed --wandb_group forecast_60 --losses progress forecast --wandb_tags experiment4 --delta_t 60

    python main.py --train_set toy_shuffle_speediest --test_set toy_shuffle_speedier --seed $seed --wandb_group forecast_embedding_70 --losses progress forecast embedding --wandb_tags experiment4 --delta_t 70
    python main.py --train_set toy_shuffle_speediest --test_set toy_shuffle_speedier --seed $seed --wandb_group forecast_70 --losses progress forecast --wandb_tags experiment4 --delta_t 70

    python main.py --train_set toy_shuffle_speediest --test_set toy_shuffle_speedier --seed $seed --wandb_group forecast_embedding_80 --losses progress forecast embedding --wandb_tags experiment4 --delta_t 80
    python main.py --train_set toy_shuffle_speediest --test_set toy_shuffle_speedier --seed $seed --wandb_group forecast_80 --losses progress forecast --wandb_tags experiment4 --delta_t 80

    python main.py --train_set toy_shuffle_speediest --test_set toy_shuffle_speedier --seed $seed --wandb_group forecast_embedding_90 --losses progress forecast embedding --wandb_tags experiment4 --delta_t 90
    python main.py --train_set toy_shuffle_speediest --test_set toy_shuffle_speedier --seed $seed --wandb_group forecast_90 --losses progress forecast --wandb_tags experiment4 --delta_t 90

    python main.py --train_set toy_shuffle_speediest --test_set toy_shuffle_speedier --seed $seed --wandb_group forecast_embedding_100 --losses progress forecast embedding --wandb_tags experiment4 --delta_t 100
    python main.py --train_set toy_shuffle_speediest --test_set toy_shuffle_speedier --seed $seed --wandb_group forecast_100 --losses progress forecast --wandb_tags experiment4 --delta_t 100
done

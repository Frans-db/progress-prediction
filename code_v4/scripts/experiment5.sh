for seed in "$@"
do
    python main.py --train_set toy_long --test_set toy_long --seed $seed --wandb_group default --wandb_tags experiment5

    python main.py --train_set toy_long --test_set toy_long --seed $seed --wandb_group forecast_embedding_10 --losses progress forecast embedding --wandb_tags experiment5 --delta_t 10
    python main.py --train_set toy_long --test_set toy_long --seed $seed --wandb_group forecast_10 --losses progress forecast --wandb_tags experiment5 --delta_t 10

    python main.py --train_set toy_long --test_set toy_long --seed $seed --wandb_group forecast_embedding_100 --losses progress forecast embedding --wandb_tags experiment5 --delta_t 100
    python main.py --train_set toy_long --test_set toy_long --seed $seed --wandb_group forecast_100 --losses progress forecast --wandb_tags experiment5 --delta_t 100

    python main.py --train_set toy_long --test_set toy_long --seed $seed --wandb_group forecast_embedding_200 --losses progress forecast embedding --wandb_tags experiment5 --delta_t 200
    python main.py --train_set toy_long --test_set toy_long --seed $seed --wandb_group forecast_200 --losses progress forecast --wandb_tags experiment5 --delta_t 200

    python main.py --train_set toy_long --test_set toy_long --seed $seed --wandb_group forecast_embedding_300 --losses progress forecast embedding --wandb_tags experiment5 --delta_t 300
    python main.py --train_set toy_long --test_set toy_long --seed $seed --wandb_group forecast_300 --losses progress forecast --wandb_tags experiment5 --delta_t 300
done
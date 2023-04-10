for seed in "$@"
do
    python main.py --train_set toy_speedier --test_set toy_speed --seed $seed --wandb_group default --wandb_tags experiment2
    python main.py --train_set toy_speedier --test_set toy_speed --seed $seed --wandb_group forecast_embedding --losses progress forecast embedding --wandb_tags experiment2
    python main.py --train_set toy_speedier --test_set toy_speed --seed $seed --wandb_group forecast --losses progress forecast --wandb_tags experiment2
    python main.py --train_set toy_speedier --test_set toy_speed --seed $seed --wandb_group augmented --augmentations subsample subsection removal --delta_t 1 --wandb_tags experiment2
done
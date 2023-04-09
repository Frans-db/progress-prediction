for seed in "$@"
do
    python main.py --train_set toy_shuffle_speediest --test_set toy_shuffle_speedier --seed $seed --wandb_group default --wandb_tags experiment4
    python main.py --train_set toy_shuffle_speediest --test_set toy_shuffle_speedier --seed $seed --wandb_group forecast_embedding --losses progress forecast embedding --wandb_tags experiment4
    python main.py --train_set toy_shuffle_speediest --test_set toy_shuffle_speedier --seed $seed --wandb_group forecast --losses progress forecast --wandb_tags experiment4
    python main.py --train_set toy_shuffle_speediest --test_set toy_shuffle_speedier --seed $seed --wandb_group augmented --augmentations subsample subsection removal --delta_t 1 --wandb_tags experiment4
done

# python main.py --train_set toy_shuffle_speediest --test_set toy_shuffle_speedier --seed 42 --wandb_group default --wandb_tags experiment4
# python main.py --train_set toy_shuffle_speediest --test_set toy_shuffle_speedier --seed 43 --wandb_group default --wandb_tags experiment4
# python main.py --train_set toy_shuffle_speediest --test_set toy_shuffle_speedier --seed 44 --wandb_group default --wandb_tags experiment4

# python main.py --train_set toy_shuffle_speediest --test_set toy_shuffle_speedier --seed 42 --wandb_group forecast_embedding --losses progress forecast embedding --wandb_tags experiment4
# python main.py --train_set toy_shuffle_speediest --test_set toy_shuffle_speedier --seed 43 --wandb_group forecast_embedding --losses progress forecast embedding --wandb_tags experiment4
# python main.py --train_set toy_shuffle_speediest --test_set toy_shuffle_speedier --seed 44 --wandb_group forecast_embedding --losses progress forecast embedding --wandb_tags experiment4

# python main.py --train_set toy_shuffle_speediest --test_set toy_shuffle_speedier --seed 42 --wandb_group forecast --losses progress forecast --wandb_tags experiment4
# python main.py --train_set toy_shuffle_speediest --test_set toy_shuffle_speedier --seed 43 --wandb_group forecast --losses progress forecast --wandb_tags experiment4
# python main.py --train_set toy_shuffle_speediest --test_set toy_shuffle_speedier --seed 44 --wandb_group forecast --losses progress forecast --wandb_tags experiment4

# python main.py --train_set toy_shuffle_speediest --test_set toy_shuffle_speedier --seed 42 --wandb_group augmented --augmentations subsample subsection removal --delta_t 1 --wandb_tags experiment4
# python main.py --train_set toy_shuffle_speediest --test_set toy_shuffle_speedier --seed 43 --wandb_group augmented --augmentations subsample subsection removal --delta_t 1 --wandb_tags experiment4
# python main.py --train_set toy_shuffle_speediest --test_set toy_shuffle_speedier --seed 44 --wandb_group augmented --augmentations subsample subsection removal --delta_t 1 --wandb_tags experiment4

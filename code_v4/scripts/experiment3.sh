python main.py --train_set toy_shuffle --test_set toy_shuffle_speed --seed 42 --wandb_group default --wandb_tags experiment3
python main.py --train_set toy_shuffle --test_set toy_shuffle_speed --seed 43 --wandb_group default --wandb_tags experiment3
python main.py --train_set toy_shuffle --test_set toy_shuffle_speed --seed 44 --wandb_group default --wandb_tags experiment3

python main.py --train_set toy_shuffle --test_set toy_shuffle_speed --seed 42 --wandb_group forecast_embedding --losses progress forecast embedding --wandb_tags experiment3
python main.py --train_set toy_shuffle --test_set toy_shuffle_speed --seed 43 --wandb_group forecast_embedding --losses progress forecast embedding --wandb_tags experiment3
python main.py --train_set toy_shuffle --test_set toy_shuffle_speed --seed 44 --wandb_group forecast_embedding --losses progress forecast embedding --wandb_tags experiment3

python main.py --train_set toy_shuffle --test_set toy_shuffle_speed --seed 42 --wandb_group forecast --losses progress forecast --wandb_tags experiment3
python main.py --train_set toy_shuffle --test_set toy_shuffle_speed --seed 43 --wandb_group forecast --losses progress forecast --wandb_tags experiment3
python main.py --train_set toy_shuffle --test_set toy_shuffle_speed --seed 44 --wandb_group forecast --losses progress forecast --wandb_tags experiment3

python main.py --train_set toy_shuffle --test_set toy_shuffle_speed --seed 42 --wandb_group forecast_embedding_t --losses progress forecast embedding --wandb_tags experiment3 --delta_t 23
python main.py --train_set toy_shuffle --test_set toy_shuffle_speed --seed 43 --wandb_group forecast_embedding_t --losses progress forecast embedding --wandb_tags experiment3 --delta_t 23
python main.py --train_set toy_shuffle --test_set toy_shuffle_speed --seed 44 --wandb_group forecast_embedding_t --losses progress forecast embedding --wandb_tags experiment3 --delta_t 23

python main.py --train_set toy_shuffle --test_set toy_shuffle_speed --seed 42 --wandb_group forecast_t --losses progress forecast --wandb_tags experiment3 --delta_t 23
python main.py --train_set toy_shuffle --test_set toy_shuffle_speed --seed 43 --wandb_group forecast_t --losses progress forecast --wandb_tags experiment3 --delta_t 23
python main.py --train_set toy_shuffle --test_set toy_shuffle_speed --seed 44 --wandb_group forecast_t --losses progress forecast --wandb_tags experiment3 --delta_t 23

python main.py --train_set toy_shuffle --test_set toy_shuffle_speed --seed 42 --wandb_group augmented --augmentations subsample subsection removal --delta_t 1 --wandb_tags experiment3
python main.py --train_set toy_shuffle --test_set toy_shuffle_speed --seed 43 --wandb_group augmented --augmentations subsample subsection removal --delta_t 1 --wandb_tags experiment3
python main.py --train_set toy_shuffle --test_set toy_shuffle_speed --seed 44 --wandb_group augmented --augmentations subsample subsection removal --delta_t 1 --wandb_tags experiment3

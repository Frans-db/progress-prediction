python main.py --train_set toy --test_set toy --seed 42 --plots --plot_directory toy --wandb_group toy --wandb_tags experiment_temp
python main.py --train_set toy --test_set toy --seed 42 --plots --plot_directory toy_forecast --losses progress forecast --wandb_group toy_forecast --wandb_tags experiment_temp
python main.py --train_set toy --test_set toy_repeated --seed 42 --plots --plot_directory toy_repeated --wandb_group toy_repeated --wandb_tags experiment_temp
python main.py --train_set toy --test_set toy_repeated --seed 42 --plots --plot_directory toy_repeated_forecast --losses progress forecast --wandb_group toy_repeated_forecast --wandb_tags experiment_temp

python main.py --train_set toy --test_set toy --seed 42 --plots --plot_directory toy --wandb_group toy
python main.py --train_set toy --test_set toy --seed 42 --plots --plot_directory toy_forecast --losses progress forecast --wandb_group toy_forecast
python main.py --train_set toy --test_set toy_repeated --seed 42 --plots --plot_directory toy_repeated --wandb_group toy_repeated
python main.py --train_set toy --test_set toy_repeated --seed 42 --plots --plot_directory toy_repeated_forecast --losses progress forecast --wandb_group toy_repeated_forecast

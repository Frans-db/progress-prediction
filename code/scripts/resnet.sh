for seed in "$@"
do
python main.py --seed $seed --network progressnet_resnet --train_set ucf24 --test_set ucf24 --data_type rgb-images --iterations 10000 --loss l1 --bo --lr 1e-4 --lr_decay_every 100000 --lr_decay 1 --subsection_chance 0.0 --subsample_chance 0.0 --test_every 1000 --wandb_group "progressnet resnet"
python main.py --seed $seed --data_type rgb-images --network progressnet_resnet --wandb_group "progressnet resnet
done
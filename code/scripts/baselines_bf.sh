for seed in "$@"
do
python main.py --seed $seed --wandb_group "progressnet"
python main.py --seed $seed --network progressnet_features_2d --wandb_group "progressnet 2d"
python main.py --seed $seed --data_modifier indices --data_modifier_value 2113.340410958904 --network progressnet_features_2d --wandb_group "progressnet 2d indices"
python main.py --seed $seed --subsection_chance 0.5 --subsample_chance 0.5 --wandb_group "progressnet augmented"
python main.py --seed $seed --data_modifier indices --data_modifier_value 2113.340410958904 --wandb_group "progressnet indices"
python main.py --seed $seed --data_modifier ones --wandb_group "progressnet ones"
python main.py --seed $seed --data_modifier randoms  --wandb_group "progressnet randoms"
done
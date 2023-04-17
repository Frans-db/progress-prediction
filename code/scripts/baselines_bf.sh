for seed in "$@"
do
python main.py --seed $seed --subsection_chance 1.0 --subsample_chance 1.0 --wandb_group "progressnet"
python main.py --seed $seed --subsection_chance 1.0 --subsample_chance 1.0 --data_modifier indices --data_modifier_value 2113.340410958904 --wandb_group "progressnet indices"
python main.py --seed $seed --subsection_chance 1.0 --subsample_chance 1.0 --data_modifier ones --wandb_group "progressnet ones"
python main.py --seed $seed --subsection_chance 1.0 --subsample_chance 1.0 --data_modifier randoms  --wandb_group "progressnet randoms"
done
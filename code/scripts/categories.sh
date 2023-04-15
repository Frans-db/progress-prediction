for seed in "$@"
do
python main.py --seed $seed --network progressnet_categories --category_directory groundtruth --wandb_name "progressnet categories $seed" --wandb_group "progressnet categories"
done
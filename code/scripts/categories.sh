for seed in "$@"
do
python main.py --train_split train_s1.txt --test_split test_s1.txt --seed $seed --network progressnet_categories --category_directory groundtruth --wandb_group "progressnet categories"
python main.py --train_split train_s2.txt --test_split test_s2.txt --seed $seed --network progressnet_categories --category_directory groundtruth --wandb_group "progressnet categories"
python main.py --train_split train_s3.txt --test_split test_s3.txt --seed $seed --network progressnet_categories --category_directory groundtruth --wandb_group "progressnet categories"
python main.py --train_split train_s3.txt --test_split test_s4.txt --seed $seed --network progressnet_categories --category_directory groundtruth --wandb_group "progressnet categories"

python main.py --train_split train_s1.txt --test_split test_s1.txt --seed $seed --network progressnet_categories --category_directory groundtruth --data_modifier indices --data_modifier_value 2113.340410958904 --wandb_group "progressnet categories indices"
python main.py --train_split train_s2.txt --test_split test_s2.txt --seed $seed --network progressnet_categories --category_directory groundtruth --data_modifier indices --data_modifier_value 2113.340410958904 --wandb_group "progressnet categories indices"
python main.py --train_split train_s3.txt --test_split test_s3.txt --seed $seed --network progressnet_categories --category_directory groundtruth --data_modifier indices --data_modifier_value 2113.340410958904 --wandb_group "progressnet categories indices"
python main.py --train_split train_s3.txt --test_split test_s4.txt --seed $seed --network progressnet_categories --category_directory groundtruth --data_modifier indices --data_modifier_value 2113.340410958904 --wandb_group "progressnet categories indices"
done
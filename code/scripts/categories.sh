for seed in "$@"
do
python main.py --train_split train_s1.txt --test_split test_s1.txt --seed $seed --network progressnet_categories --category_directory groundtruth --wandb_group "progressnet categories all" --wandb_tags all

python main.py --train_split train_cereals_s1.txt --test_split test_cereals_s1.txt --seed $seed --network progressnet_categories --category_directory groundtruth --wandb_group "progressnet categories" --num_categories 5 --wandb_tags cereals
python main.py --train_split train_coffee_s1.txt --test_split test_coffee_s1.txt --seed $seed --network progressnet_categories --category_directory groundtruth --wandb_group "progressnet categories" --num_categories 7 --wandb_tags coffee
python main.py --train_split train_friedegg_s1.txt --test_split test_friedegg_s1.txt --seed $seed --network progressnet_categories --category_directory groundtruth --wandb_group "progressnet categories" --num_categories 9 --wandb_tags friedegg
python main.py --train_split train_juice_s1.txt --test_split test_juice_s1.txt --seed $seed --network progressnet_categories --category_directory groundtruth --wandb_group "progressnet categories" --num_categories 8 --wandb_tags juice
python main.py --train_split train_milk_s1.txt --test_split test_milk_s1.txt --seed $seed --network progressnet_categories --category_directory groundtruth --wandb_group "progressnet categories" --num_categories 5 --wandb_tags milk
python main.py --train_split train_pancake_s1.txt --test_split test_pancake_s1.txt --seed $seed --network progressnet_categories --category_directory groundtruth --wandb_group "progressnet categories" --num_categories 14 --wandb_tags pancake
python main.py --train_split train_salat_s1.txt --test_split test_salat_s1.txt --seed $seed --network progressnet_categories --category_directory groundtruth --wandb_group "progressnet categories" --num_categories 8 --wandb_tags salat
python main.py --train_split train_sandwich_s1.txt --test_split test_sandwich_s1.txt --seed $seed --network progressnet_categories --category_directory groundtruth --wandb_group "progressnet categories" --num_categories 9 --wandb_tags sandwich
python main.py --train_split train_scrambledegg_s1.txt --test_split test_scrambledegg_s1.txt --seed $seed --network progressnet_categories --category_directory groundtruth --wandb_group "progressnet categories" --num_categories 12 --wandb_tags scrambledegg
python main.py --train_split train_tea_s1.txt --test_split test_tea_s1.txt --seed $seed --network progressnet_categories --category_directory groundtruth --wandb_group "progressnet categories" --num_categories 7 --wandb_tags tea

python main.py --train_split train_cereals_s1.txt --test_split test_cereals_s1.txt --seed $seed --network progressnet_features --wandb_group "progressnet per action" --wandb_tags cereals
python main.py --train_split train_coffee_s1.txt --test_split test_coffee_s1.txt --seed $seed --network progressnet_features --wandb_group "progressnet per action" --wandb_tags coffee
python main.py --train_split train_friedegg_s1.txt --test_split test_friedegg_s1.txt --seed $seed --network progressnet_features --wandb_group "progressnet per action" --wandb_tags friedegg
python main.py --train_split train_juice_s1.txt --test_split test_juice_s1.txt --seed $seed --network progressnet_features --wandb_group "progressnet per action" --wandb_tags juice
python main.py --train_split train_milk_s1.txt --test_split test_milk_s1.txt --seed $seed --network progressnet_features --wandb_group "progressnet per action" --wandb_tags milk
python main.py --train_split train_pancake_s1.txt --test_split test_pancake_s1.txt --seed $seed --network progressnet_features --wandb_group "progressnet per action" --wandb_tags pancake
python main.py --train_split train_salat_s1.txt --test_split test_salat_s1.txt --seed $seed --network progressnet_features --wandb_group "progressnet per action" --wandb_tags salat
python main.py --train_split train_sandwich_s1.txt --test_split test_sandwich_s1.txt --seed $seed --network progressnet_features --wandb_group "progressnet per action" --wandb_tags sandwich
python main.py --train_split train_scrambledegg_s1.txt --test_split test_scrambledegg_s1.txt --seed $seed --network progressnet_features --wandb_group "progressnet per action" --wandb_tags scrambledegg
python main.py --train_split train_tea_s1.txt --test_split test_tea_s1.txt --seed $seed --network progressnet_features --wandb_group "progressnet per action" --wandb_tags tea
done


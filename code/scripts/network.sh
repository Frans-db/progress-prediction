for seed in "$@"
do
python main.py --seed $seed --train_split test_1.txt --test_split test_1.txt --bounding_boxes --iterations 2500 --loss l1 --lr 1e-4 --test_every 500 --basemodel vgg512 --basemodel_name vgg_512.pth --wandb_group vgg_512 --wandb_tags network_choice --wandb_name vgg_512
python main.py --seed $seed --train_split test_1.txt --test_split test_1.txt --bounding_boxes --iterations 2500 --loss l1 --lr 1e-4 --test_every 500 --basemodel vgg512 --basemodel_name vgg_512_features.pth --wandb_group vgg_512_features --wandb_tags network_choice --wandb_name vgg_512_features
python main.py --seed $seed --train_split test_1.txt --test_split test_1.txt --bounding_boxes --iterations 2500 --loss l1 --lr 1e-4 --test_every 500 --basemodel vgg1024 --basemodel_name vgg_1024.pth --wandb_group vgg_1024 --wandb_tags network_choice --wandb_name vgg_1024
python main.py --seed $seed --train_split test_1.txt --test_split test_1.txt --bounding_boxes --iterations 2500 --loss l1 --lr 1e-4 --test_every 500 --basemodel vgg1024 --basemodel_name vgg_1024_reduced.pth --wandb_group vgg_1024_reduced --wandb_tags network_choice --wandb_name vgg_1024_reduced
done
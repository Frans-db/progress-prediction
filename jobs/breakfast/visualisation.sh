python /home/nfs/fransdeboer/mscfransdeboer/code/main.py \
    --seed 42 \
    --dataset breakfast \
    --data_dir features/resnet152_sampled_1 \
    --feature_dim 2048 \
    --train_split test_small.txt \
    --test_split test_small.txt \
    --batch_size 1 \
    --network lstmnet \
    --load_experiment lstm_bf_random_1 \
    --load_iteration 30000 \
    --random \
    --save_dir lstm_bf_random

python /home/nfs/fransdeboer/mscfransdeboer/code/main.py \
    --seed 42 \
    --dataset breakfast \
    --data_dir features/resnet152_sampled_1 \
    --feature_dim 2048 \
    --rsd_type minutes \
    --fps 1 \
    --rsd_normalizer 5 \
    --train_split test_small.txt \
    --test_split test_small.txt \
    --batch_size 1 \
    --network rsdnet  \
    --load_experiment rsd_bf_random_1 \
    --load_iteration 30000 \
    --random \
    --save_dir rsd_bf_random


python /home/nfs/fransdeboer/mscfransdeboer/code/main.py \
    --seed 42 \
    --dataset breakfast \
    --data_dir rgb-images \
    --train_split test_small.txt \
    --test_split test_small.txt \
    --batch_size 1 \
    --network progressnet \
    --backbone vgg16 \
    --load_backbone vgg16.pth \
    --num_workers 2 \
    --max_length 550 \
    --load_experiment pn_bf_random_1 \
    --load_iteration 50000 \
    --subsample_fps 15 \
    --random \
    --save_dir pn_bf_random
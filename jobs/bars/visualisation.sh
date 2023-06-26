python main.py \
    --seed 42 \
    --dataset bars \
    --data_dir features/resnet18 \
    --feature_dim 512 \
    --train_split 1.txt \
    --test_split 1.txt \
    --batch_size 1 \
    --network lstmnet \
    --subsample \
    --load_experiment lstm_bars \
    --load_iteration 30000 \
    --save_dir lstm_bars

python main.py \
    --seed 42 \
    --dataset bars \
    --data_dir features/resnet18 \
    --feature_dim 512 \
    --rsd_type minutes \
    --fps 1 \
    --rsd_normalizer 5 \
    --train_split 1.txt \
    --test_split 1.txt \
    --batch_size 1 \
    --network rsdnet \
    --subsample \
    --load_experiment rsd_bars \
    --load_iteration 30000 \
    --save_dir rsd_bars

python main.py \
    --seed 42 \
    --dataset bars \
    --data_dir features/i3d_embeddings \
    --flat \
    --train_split 1.txt \
    --test_split 1.txt \
    --batch_size 10000 \
    --network ute \
    --feature_dim 1024 \
    --embed_dim 20 \
    --num_workers 2 \
    --load_experiment ute_bars \
    --load_iteration 50000 \
    --save_dir ute_bars

python main.py \
    --seed 42 \
    --dataset bars \
    --data_dir rgb-images \
    --flat \
    --train_split 1.txt \
    --test_split 1.txt \
    --batch_size 500 \
    --network rsdnet_flat \
    --backbone resnet18 \
    --load_backbone resnet18.pth \
    --load_experiment resnet_bars \
    --load_iteration 50000 \
    --save_dir resnet_bars \
    --no_resize

python main.py \
    --seed 42 \
    --dataset bars \
    --data_dir rgb-images \
    --train_split 1.txt \
    --test_split 1.txt \
    --batch_size 1 \
    --network progressnet \
    --backbone vgg16 \
    --load_backbone vgg16.pth \
    --num_workers 2 \
    --subsample \
    --max_length 100 \
    --load_experiment pn_bars_segments \
    --load_iteration 40000 \
    --save_dir pn_bars \
    --no_resize
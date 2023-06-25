python /home/nfs/fransdeboer/mscfransdeboer/code/main.py \
    --seed 42 \
    --dataset cholec80 \
    --data_dir features/resnet152_0 \
    --feature_dim 2048 \
    --train_split 1.txt \
    --test_split 1.txt \
    --batch_size 1 \
    --network lstmnet \
    --subsample \
    --load_experiment lstm_cholec_segments_0 \
    --load_iteration 30000 \
    --save_dir lstm_cholec

python /home/nfs/fransdeboer/mscfransdeboer/code/main.py \
    --seed 42 \
    --dataset cholec80 \
    --data_dir features/resnet152_0 \
    --feature_dim 2048 \
    --rsd_type minutes \
    --fps 1 \
    --rsd_normalizer 5 \
    --train_split 1.txt \
    --test_split 1.txt \
    --batch_size 1 \
    --network rsdnet \
    --subsample \
    --load_experiment rsd_cholec_segments_0 \
    --load_iteration 30000 \
    --save_dir rsd_cholec

python /home/nfs/fransdeboer/mscfransdeboer/code/main.py \
    --seed 42 \
    --dataset cholec80 \
    --data_dir features/i3d_embeddings \
    --flat \
    --train_split 1.txt \
    --test_split 1.txt \
    --batch_size 10000 \
    --network ute \
    --feature_dim 1024 \
    --embed_dim 20 \
    --num_workers 2 \
    --load_experiment ute_cholec_segments_0 \
    --load_iteration 50000 \
    --save_dir ute_cholec

python /home/nfs/fransdeboer/mscfransdeboer/code/main.py \
    --seed 42 \
    --dataset cholec80 \
    --data_dir rgb-images \
    --flat \
    --train_split 1.txt \
    --test_split 1.txt \
    --batch_size 500 \
    --network rsdnet_flat \
    --backbone resnet152 \
    --load_backbone resnet152.pth \
    --load_experiment resnet152_cholec80_0 \
    --load_iteration 50000 \
    --save_dir resnet_cholec

python /home/nfs/fransdeboer/mscfransdeboer/code/main.py \
    --seed 42 \
    --dataset cholec80 \
    --data_dir rgb-images \
    --train_split 1.txt \
    --test_split 1.txt \
    --batch_size 1 \
    --network progressnet \
    --backbone vgg16 \
    --load_backbone vgg16.pth \
    --num_workers 2 \
    --subsample \
    --max_length 550 \
    --load_experiment pn_cholec_segments_0 \
    --load_iteration 50000 \
    --save_dir pn_cholec
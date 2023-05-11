python /home/frans/Projects/mscfransdeboer/code/main.py \
            --seed 42 \
            --wandb_group ucf24 \
            --wandb_project ute \
            --dataset ucf24 \
            --data_dir features/i3d_embeddings \
            --flat \
            --train_split train_tubes.txt \
            --test_split test_tubes.txt \
            --batch_size 256 \
            --iterations 100000 \
            --network ute \
            --feature_dim 1024 \
            --embed_dim 20 \
            --optimizer adam \
            --lr 1e-3 \
            --dropout_chance 0.5 \
            --weight_decay 5e-6 \
            --lr_decay 1 \
            --lr_decay_every 10000 \
            --log_every 500 \
            --test_every 1000

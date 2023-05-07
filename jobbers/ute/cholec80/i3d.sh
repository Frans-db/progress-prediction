# cholec normal
for i in 0 1 2 3
do
    python /home/frans/Projects/mscfransdeboer/code/main.py \
        --seed 42 \
        --wandb_group i3d \
        --wandb_tags cholec80 i3d \
        --wandb_project ute \
        --dataset cholec80 \
        --data_dir features/i3d_embeddings \
        --flat \
        --train_split t12_p${i}.txt \
        --test_split e_p${i}.txt \
        --batch_size 256 \
        --iterations 30000 \
        --network ute \
        --feature_dim 1024 \
        --embed_dim 20 \
        --optimizer adam \
        --lr 1e-3 \
        --weight_decay 1e-4 \
        --lr_decay 0.5 \
        --lr_decay_every 2000 \
        --log_every 500 \
        --test_every 500
done
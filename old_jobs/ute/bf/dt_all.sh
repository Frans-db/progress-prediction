for activity in coffee cereals tea milk juice sandwich scrambledegg friedegg salat pancake
do
        python /home/frans/Projects/mscfransdeboer/code/main.py \
            --seed 42 \
            --wandb_group all_${activity} \
            --wandb_tags all \
            --wandb_project ute \
            --dataset breakfast \
            --data_dir features/dense_trajectories \
            --flat \
            --train_split all_${activity}.txt \
            --test_split all_${activity}.txt \
            --batch_size 256 \
            --iterations 30000 \
            --network ute \
            --feature_dim 64 \
            --embed_dim 20 \
            --optimizer adam \
            --lr 1e-3 \
            --weight_decay 0 \
            --lr_decay 0.5 \
            --lr_decay_every 10000 \
            --log_every 500 \
            --test_every 1000

done
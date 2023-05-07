for i in 1 2 3 4
do
    for activity in coffee cereals tea milk juice sandwich scrambledegg friedegg salat pancake
    do
        python /home/frans/Projects/mscfransdeboer/code/main.py \
            --seed 42 \
            --wandb_group ${activity} \
            --wandb_tags individual \
            --wandb_project ute \
            --dataset breakfast \
            --data_dir features/dense_trajectories \
            --flat \
            --train_split train_${activity}_s${i}.txt \
            --test_split test_${activity}_s${i}.txt \
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
done
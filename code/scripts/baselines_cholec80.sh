for seed in "$@"
do
python main.py --seed $seed --subsection_chance 1.0 --subsample_chance 1.0 --train_set cholec80 --test_set cholec80 --train_split t1_t2_0.txt --test_split e_0.txt --data_type fold_0_embeddings --embedding_size 2048 --wandb_group "progressnet"
python main.py --seed $seed --subsection_chance 1.0 --subsample_chance 1.0 --train_set cholec80 --test_set cholec80 --train_split t1_t2_0.txt --test_split e_0.txt --data_type fold_0_embeddings --embedding_size 2048 --data_modifier indices --data_modifier_value 2382.9074074074074 --wandb_group "progressnet indices"
done
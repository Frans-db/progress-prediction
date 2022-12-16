#!/bin/sh
python progress/experiment_2d_sampling.py --epochs 100 --dataset toy_static        --num_segments 1 --frames_per_segment 25 --name toy_static_1_25
python progress/experiment_2d_sampling.py --epochs 100 --dataset toy_speed         --num_segments 1 --frames_per_segment 25 --name toy_speed_1_25
python progress/experiment_2d_sampling.py --epochs 100 --dataset toy_speed_segment --num_segments 1 --frames_per_segment 25 --name toy_speed_segment_1_25
python progress/experiment_2d_sampling.py --epochs 100 --dataset toy_segment       --num_segments 1 --frames_per_segment 25 --name toy_segment_1_25

python progress/experiment_2d_sampling.py --epochs 100 --dataset toy_static        --num_segments 25 --frames_per_segment 1 --name toy_static_25_1
python progress/experiment_2d_sampling.py --epochs 100 --dataset toy_speed         --num_segments 25 --frames_per_segment 1 --name toy_speed_25_1
python progress/experiment_2d_sampling.py --epochs 100 --dataset toy_speed_segment --num_segments 25 --frames_per_segment 1 --name toy_speed_segment_25_1
python progress/experiment_2d_sampling.py --epochs 100 --dataset toy_segment       --num_segments 25 --frames_per_segment 1 --name toy_segment_25_1
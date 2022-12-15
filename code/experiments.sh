#!/bin/sh
source ./venv/bin/activate

python progress/experiment_2d.py --epochs 15 --dataset toy_static --name toy_static
python progress/experiment_2d.py --epochs 15 --dataset toy_speed --name toy_static
python progress/experiment_2d.py --epochs 30 --dataset toy_speed_segment --name toy_static
python progress/experiment_2d.py --epochs 30 --dataset toy_segment --name toy_static

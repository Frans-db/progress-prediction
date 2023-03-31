#!/bin/sh
python data/make_toy.py --new_path /home/frans/Datasets/toy_static
python data/make_toy.py --new_path /home/frans/Datasets/toy_static_shuffle --shuffle 100
python data/make_toy.py --new_path /home/frans/Datasets/toy_static_drop --drop 50

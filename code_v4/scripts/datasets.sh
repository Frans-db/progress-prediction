rm -r /home/frans/Datasets/toy*
python make_toy.py --dataset toy --add_bg
python make_toy.py --dataset toy_speed --min-speed 1 --max-speed 3 --add_bg
python make_toy.py --dataset toy_speedier --min-speed 3 --max-speed 6 --add_bg

python make_toy.py --dataset toy_shuffle --shuffle 100 --add_bg
python make_toy.py --dataset toy_shuffle_speed --shuffle 100 --min-speed 3 --max-speed 6 --add_bg

python make_toy.py --dataset toy_shuffle_speedier --shuffle 100 --min-speed 0.9 --max-speed 1.1 --add_bg
python make_toy.py --dataset toy_shuffle_speediest --shuffle 100 --min-speed 9 --max-speed 11 --add_bg

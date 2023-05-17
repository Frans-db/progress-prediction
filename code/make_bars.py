import argparse
from typing import List
import random

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--num_videos', type=int, default=1000)

    return parser.parse_args()

COLOURS = ['r', 'g', 'b', 'c', 'y', 'm']


def main():
    args = parse_args()
    random.seed(args.seed)



if __name__ == '__main__':
    main()
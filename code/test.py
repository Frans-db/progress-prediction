import os
from typing import List, Dict
from collections import defaultdict
import math

DATA_ROOT = "/home/frans/Datasets"

def group_names(root: str, read: str, save: str) -> None:
    with open(os.path.join(root, read)) as f:
        names = [name.strip() for name in f.readlines()]

    grouped = defaultdict(list)
    for name in names:
        activity = name.split('/')[0]
        grouped[activity].append(name)
    new_names = []
    for activity in grouped:
        names = sorted(grouped[activity])
        num_names = len(names)
        cutoff = num_names - math.floor(num_names / 10) * 10
        if cutoff != 0:
            names = names[:-cutoff]
        sample_ratio = len(names) // 10

        names = names[::sample_ratio]
        new_names.extend(names)

    with open(os.path.join(root, save), 'w+') as f:
        f.write('\n'.join(new_names))

     


def main():
    with open('./data/bf_baseline.txt') as f:
        data = [float(line.strip()) for line in f.readlines()]
    data = data[::15]
    with open('./data/bf_baseline_sampled.txt', 'w+') as f:
        f.write('\n'.join([str(line) for line in data]))
    # group_names(os.path.join(DATA_ROOT, 'breakfast/splitfiles/'), 'all.txt', 'small.txt')
    # group_names(os.path.join(DATA_ROOT, 'ucf24/splitfiles/'), 'all.txt', 'small.txt')

if __name__ == '__main__':
    main()
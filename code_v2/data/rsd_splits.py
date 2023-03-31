import os
import argparse
import random
from typing import List

def save(target: str, name: str, index: int, items: List[int]):
    with open(os.path.join(target, f'{name}_{index}.txt'), 'w+') as f:
        f.writelines('\n'.join([f'video{(item+1):02d}' for item in items]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--t1', type=int, default=27)
    parser.add_argument('--t2', type=int, default=27)
    parser.add_argument('--v', type=int, default=7)
    parser.add_argument('--e', type=int, default=19)
    parser.add_argument('--num_splits', type=int, default=4)
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()

    random.seed(args.seed)

    target = os.path.join(args.dataset, 'splitfiles')
    total = args.t1 + args.t2 + args.v + args.e

    for split_index in range(args.num_splits):
        indices = list(range(total))
        random.shuffle(indices)

        t1 = indices[:args.t1]
        t2 = indices[args.t1:args.t1+args.t2]
        v = indices[args.t1 + args.t2 : args.t1 + args.t2 + args.v]
        e = indices[args.t1 + args.t2 + args.v : args.t1 + args.t2 + args.v + args.e]

        save(target, 't1', split_index, t1)
        save(target, 't2', split_index, t2)
        save(target, 't1_t2', split_index, t1+t2)
        save(target, 'v', split_index, v)
        save(target, 'e', split_index, e)


if __name__ == '__main__':
    main()
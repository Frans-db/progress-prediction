import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm
import torch
import torch.nn as nn
import statistics
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/home/frans/Datasets/experiments')
    parser.add_argument('--experiment', type=str, default='indices')
    parser.add_argument('--iteration', type=int, default=10000)
    return parser.parse_args()

def main():
    args = parse_args()

    path = os.path.join(args.data_root, args.experiment, 'results', f'{args.iteration}.json')
    with open(path) as f:
        data = json.load(f)

    lengths = []
    progress = []
    for result in tqdm(data):
        length = len(result['progress'])
        for i, p in enumerate(result['progress']):
            if len(progress) <= i:
                progress.append([])
            progress[i].append(p)
        lengths.append(length)
    average_progress = []
    for i, values in enumerate(progress):
        average_progress.append(statistics.mean(values))
    plt.plot(average_progress)
    plt.savefig('./avg.png')
    mean, median, mode = statistics.mean(lengths), statistics.median(lengths), statistics.mode(lengths)
    print('mean', mean)
    print('median', median)
    print('mode', mode)
    count = 0
    criterion = nn.MSELoss(reduction='sum')
    losses = {
        'predicted': 0.0,
        'dumb': 0.0,
        'best': 0.0,
    }
    for result in tqdm(data):
        name = result['video_name'].replace('/', '_')
        progress = torch.FloatTensor(result['progress'])
        predictions = torch.FloatTensor(result['predictions'])
        S = progress.shape[0]
        count += S
        best = torch.FloatTensor(average_progress[:S])

        losses['predicted'] += criterion(predictions, progress).item()
        losses['best'] += criterion(best, progress).item()

        plt.plot(progress, label='progress')
        plt.plot(predictions, label='predictions')
        plt.plot(best, label='best')
        if S > mean:
            plt.vlines(mean, 0, 1, colors='g')
        if S > 2 * mean:
            plt.vlines(mean*2, 0, 1, colors='g')
        plt.legend(loc='best')
        plt.savefig(os.path.join(args.data_root, args.experiment, 'plots', f'{S}_{name}.png'))
        plt.clf()

    for key in losses:
        print(key, losses[key] / count)

if __name__ == '__main__':
    main()
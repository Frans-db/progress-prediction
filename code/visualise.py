import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm
import torch
import torch.nn as nn
import statistics
import argparse
from typing import List
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

def load_splitfile(path: str) -> List[str]:
    with open(path) as f:
        names = f.readlines()
        names = [name.strip() for name in names]
    return names

def main():
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:lime'] 
    data_root = '/mnt/hdd/vgg512'
    split_root = '/home/frans/Datasets/ucf24/splitfiles'
    splitfiles = ['train_telic.txt', 'test_telic.txt']
    names = []
    for splitfile in splitfiles:
        path = os.path.join(split_root, splitfile)
        names += load_splitfile(path)

    data, activities, progress = [], [], []
    count, total = 0, len(names)
    print('--- Loading Data ---')
    for root, dirs, files in os.walk(data_root, topdown=False):
        for name in files[:20]:
            activity = root.split('/')[-1]
            cleaned_name = activity + '/' + '_'.join(name.split('_')[:-2])
            if cleaned_name not in names:
                continue
            count += 1
            print(f'{count}/{total}, {name}')
            path = os.path.join(root, name)
            with open(path) as f:
                file_data = f.readlines()
                length = len(file_data)
                for i,row in enumerate(file_data):
                    values = list(map(float, row.split(' ')))
                    data.append(values)
                    activities.append(activity)
                    progress.append((i+1) / length)

    print('--- Fitting PCA ---')
    pca = PCA(n_components=2)
    pca.fit(data)
    print('--- Transforming Data ---')
    transformed = pca.transform(data)

    num_samples = len(activities)
    unique_activities = set(activities)
    new_data = np.concatenate((transformed, np.array(activities).reshape(num_samples, 1), np.array(progress).reshape(num_samples, 1)), axis=1)

    print('--- Plotting ---')
    for activity in tqdm(unique_activities):
        indices = new_data[:, 2] == activity
        subset = new_data[indices]
        print(f'--- Plotting {activity} ({len(subset)}) ---')
        plt.scatter(subset[:, 0].astype(float), subset[:, 1].astype(float), label=activity)
    
    print('--- Saving ---')
    plt.title('PCA (n=2) grouped by activity (telic activities)')
    plt.legend(loc='best')
    plt.savefig('./scatter_activities.png')
    plt.clf()

    print('--- Plotting ---')
    for activity in tqdm(unique_activities):
        indices = new_data[:, 2] == activity
        subset = new_data[indices]
        print(f'--- Plotting {activity} ({len(subset)}) ---')
        plt.scatter(subset[:, 0].astype(float), subset[:, 1].astype(float), c=subset[:, 3].astype(float) * 100, cmap='viridis')
    
    print('--- Saving ---')
    plt.colorbar()
    plt.title('PCA (n=2) grouped by progress (telic activities)')
    # plt.legend(loc='best')
    plt.savefig('./scatter_progress.png')
    plt.clf()

if __name__ == '__main__':
    main()

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_root', type=str, default='/home/frans/Datasets/experiments')
#     parser.add_argument('--experiment', type=str, default='indices')
#     parser.add_argument('--iteration', type=int, default=10000)
#     return parser.parse_args()

# def main():
#     args = parse_args()

#     path = os.path.join(args.data_root, args.experiment, 'results', f'{args.iteration}.json')
#     with open(path) as f:
#         data = json.load(f)

#     lengths = []
#     progress = []
#     for result in tqdm(data):
#         length = len(result['progress'])
#         for i, p in enumerate(result['progress']):
#             if len(progress) <= i:
#                 progress.append([])
#             progress[i].append(p)
        
#         lengths.append((length, result['video_name']))
#     lengths.sort(key=lambda x:x[0])
#     for (length, name) in lengths:
#         print(length, name)

#     return




#     average_progress = []
#     for i, values in enumerate(progress):
#         average_progress.append(statistics.mean(values))
#     plt.plot(average_progress)
#     plt.savefig('./avg.png')
#     mean, median, mode = statistics.mean(lengths), statistics.median(lengths), statistics.mode(lengths)
#     print('mean', mean)
#     print('median', median)
#     print('mode', mode)
#     count = 0
#     criterion = nn.MSELoss(reduction='sum')
#     losses = {
#         'predicted': 0.0,
#         'dumb': 0.0,
#         'best': 0.0,
#     }
#     for result in tqdm(data):
#         name = result['video_name'].replace('/', '_')
#         progress = torch.FloatTensor(result['progress'])
#         predictions = torch.FloatTensor(result['predictions'])
#         S = progress.shape[0]
#         count += S
#         best = torch.FloatTensor(average_progress[:S])

#         losses['predicted'] += criterion(predictions, progress).item()
#         losses['best'] += criterion(best, progress).item()

#         plt.plot(progress, label='progress')
#         plt.plot(predictions, label='predictions')
#         plt.plot(best, label='best')
#         if S > mean:
#             plt.vlines(mean, 0, 1, colors='g')
#         if S > 2 * mean:
#             plt.vlines(mean*2, 0, 1, colors='g')
#         plt.legend(loc='best')
#         plt.savefig(os.path.join(args.data_root, args.experiment, 'plots', f'{S}_{name}.png'))
#         plt.clf()

#     for key in losses:
#         print(key, losses[key] / count)

# if __name__ == '__main__':
#     main()
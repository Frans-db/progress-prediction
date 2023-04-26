from typing import List
import os
import matplotlib.pyplot as plt
import math
import statistics
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def read_splitfile(path: str) -> List[str]:
    with open(path) as f:
        names = f.readlines()
        names = [name.strip() for name in names]
    return names

def get_dataset_lengths(root: str, splits: List[str]) -> List[float]:
    lengths = []
    for video_name in splits:
        video_path = os.path.join(root, video_name)
        frames = sorted(os.listdir(video_path))
        lengths.append(len(frames))

    return lengths

def get_best_loss(lengths: List[int], plot_name='all') -> float:
    average_progress = torch.zeros(max(lengths))
    counts = torch.zeros(max(lengths))
    
    for length in lengths:
        progress = torch.arange(1, length + 1) / length
        average_progress[:length] += progress
        counts[:length] = counts[:length] + 1
    average_progress = average_progress / counts
    loss = nn.MSELoss(reduction='sum')
    total_loss, total_count = 0, 0
    for length in lengths:
        progress = torch.arange(1, length + 1) / length
        predictions = average_progress[:length]
        total_loss += loss(predictions, progress)
        total_count += length
    average_loss = total_loss / total_count
    print(f'average loss for {plot_name}', average_loss)
    plt.plot(average_progress)
    plt.title(f'Loss {plot_name}: {average_loss:.3f}')
    plt.savefig(f'./plots/{plot_name}.png')
    plt.clf()

def main():
    data_root = '/home/frans/Datasets/breakfast_reshaped'
    train_path = os.path.join(data_root, 'splitfiles', 'train_s1.txt')
    test_path = os.path.join(data_root, 'splitfiles', 'test_s1.txt')
    split_names = read_splitfile(train_path) + read_splitfile(test_path)
    lengths = sorted(get_dataset_lengths(os.path.join(data_root, 'rgb-images'), split_names))
    get_best_loss(lengths)

    activites = ['coffee', 'cereals', 'tea', 'milk', 'juice', 'sandwich', 'scrambledegg', 'friedegg', 'salat', 'pancake']
    for activity in activites:
        lengths = sorted(get_dataset_lengths(os.path.join(data_root, 'rgb-images'), filter(lambda x: activity in x, split_names)))
        get_best_loss(lengths, plot_name=activity)


    # cholec80_split = [f'video{i:02d}' for i in range(1, 81)]
    # visualise_dataset_lengths(os.path.join(data_root, 'cholec80/rgb-images'), cholec80_split)

if __name__ == '__main__':
    main()


# def visualise_dataset_lengths(root: str, splits: List[str]) -> None:
#     buckets = {i*10 + 5: 0 for i in range(11)}
#     lengths = []
#     for video_name in splits:
#         video_path = os.path.join(root, video_name)
#         frames = sorted(os.listdir(video_path))
#         minutes = len(frames) / 60 # videos are subsampled at 1fps
#         rounded_minutes = math.floor(minutes / 10) * 10
#         lengths.append(minutes)
#         buckets[rounded_minutes + 5] += 1

#     q1, mean, q3 = statistics.quantiles(lengths)

#     plt.suptitle('Length Distribution')
#     plt.title(f'q1: {q1:.1f}, mean: {mean:.1f}, q3: {q3:.1f}')
#     plt.vlines([q1, mean, q3], 0, 25, linestyles='dashed')
#     plt.bar(list(buckets.keys()), list(buckets.values()), width=10, edgecolor = 'black')
#     plt.xticks(list(map(lambda x: x * 10, range(11))))
#     plt.savefig('./lengths.png')
import os
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
import random
import scipy.io

from datasets import FeatureDataset, ImageDataset

DATA_ROOT = "/home/frans/Datasets"


def plot_lengths(dataset, ax, title):
    lengths = dataset.lengths
    buckets = {}
    for length in lengths:
        length = round(length / 10) * 10
        if length not in buckets:
            buckets[length] = 0
        buckets[length] += 1

    print(title)
    print(buckets)
    keys = sorted(list(buckets.keys()))
    ax.set_title(title)
    ax.bar(keys, [buckets[key] for key in keys])


def visualise_lengths():
    breakfast = FeatureDataset(
        os.path.join(DATA_ROOT, "breakfast"),
        "features/dense_trajectories",
        f"all.txt",
        False,
        1,
        False,
        False,
        1,
        "none",
        1,
    )
    breakfast_sampled = FeatureDataset(
        os.path.join(DATA_ROOT, "breakfast"),
        "features/dense_trajectories",
        f"all.txt",
        False,
        1,
        False,
        False,
        1,
        "none",
        1,
    )
    cholec80 = FeatureDataset(
            os.path.join(DATA_ROOT, "cholec80"),
            "features/resnet152_0",
            f"all_0.txt",
            False,
            1,
            False,
            False,
            1,
            "none",
            1,
    )
    cholec80_sampled = FeatureDataset(
            os.path.join(DATA_ROOT, "cholec80"),
            "features/resnet152_0",
            f"all_0.txt",
            False,
            10,
            False,
            False,
            1,
            "none",
            1,
    )
    ucf24 = FeatureDataset(
        os.path.join(DATA_ROOT, "ucf24"),
        "features/resnet152",
        "all_tubes.txt",
        False,
        1,
        False,
        False,
        1,
        "none",
        1,
    )

    figure, (axs) = plt.subplots(2, 3)
    plot_lengths(breakfast, axs[0, 0], 'breakfast')
    plot_lengths(breakfast_sampled, axs[1, 0], 'breakfast (sampled)')

    plot_lengths(cholec80, axs[0, 1], 'cholec80')
    plot_lengths(cholec80_sampled, axs[1, 1], 'cholec80 (sampled)')

    plot_lengths(ucf24, axs[0, 2], 'ucf24')

    plt.savefig('./aaaa.png')

def main():
    visualise_lengths()

if __name__ == "__main__":
    main()

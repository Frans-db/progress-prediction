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


def get_datasetset_lengths(dataset):
    lengths = []
    for item in dataset:
        lengths.append(item[1].shape[0])
    return lengths


def calc_baseline(trainset, testset):
    train_lengths = get_datasetset_lengths(trainset)
    test_lengths = get_datasetset_lengths(testset)

    max_length = max(train_lengths)
    loss = nn.L1Loss(reduction="sum")
    averages = torch.zeros(max(train_lengths))
    counts = torch.zeros(max(train_lengths))
    for length in train_lengths:
        progress = torch.arange(1, length + 1) / length
        averages[:length] += progress
        counts[:length] += 1
    averages = averages / counts

    count = 0
    average_loss = 0.0
    mid_loss = 0.5
    random_loss = 0.5
    for length in test_lengths:
        l = min(length, max_length)
        progress = torch.arange(1, length + 1) / length

        average_predictions = torch.ones(length)
        average_predictions[:l] = averages[:l]
        average_loss += loss(average_predictions * 100, progress * 100).item()
        mid_loss += loss(torch.full_like(progress, 0.5) * 100, progress * 100).item()
        random_loss += loss(torch.rand_like(progress) * 100, progress * 100).item()

        count += length

    length = max(max(test_lengths), max_length)
    predictions = torch.ones(length)
    predictions[:max_length] = averages[:max_length]

    return average_loss / count, mid_loss / count, random_loss / count


def ucf_baseline():
    trainset = FeatureDataset(
        os.path.join(DATA_ROOT, "ucf24"),
        "features/resnet152",
        "train_tubes.txt",
        False,
        1, False,
        False,
        1,
        "none",
        1,
    )
    testset = FeatureDataset(
        os.path.join(DATA_ROOT, "ucf24"),
        "features/resnet152",
        "test_tubes.txt",
        False,
        1, False,
        False,
        1,
        "none",
        1,
    )
    losses = calc_baseline(trainset, testset)
    print(f"--- ucf ---")
    print("average", losses[0])
    print("0.5", losses[1])
    print("random", losses[2])



def cholec_baseline():
    losses = [0, 0, 0]
    for i in range(4):
        trainset = FeatureDataset(
            os.path.join(DATA_ROOT, "cholec80"),
            "features/resnet152_0",
            f"t12_{i}.txt",
            False,
            1, False,
            False,
            1,
            "none",
            1,
        )
        testset = FeatureDataset(
            os.path.join(DATA_ROOT, "cholec80"),
            "features/resnet152_0",
            f"t12_{i}.txt",
            False,
            1, False,
            False,
            1,
            "none",
            1,
        )

        for i, loss in enumerate(
            calc_baseline(trainset, testset)
        ):
            print(loss)
            losses[i] += loss / 4
    print(f"--- cholec ---")
    print("average", losses[0])
    print("0.5", losses[1])
    print("random", losses[2])

    losses = [0, 0, 0]
    for i in range(4):
        trainset = FeatureDataset(
            os.path.join(DATA_ROOT, "cholec80"),
            "features/resnet152_0",
            f"t12_{i}.txt",
            False,
            10, False,
            False,
            1,
            "none",
            1,
        )
        testset = FeatureDataset(
            os.path.join(DATA_ROOT, "cholec80"),
            "features/resnet152_0",
            f"e_{i}.txt",
            False,
            10, False,
            False,
            1,
            "none",
            1,
        )

        for i, loss in enumerate(
            calc_baseline(trainset, testset)
        ):
            print(loss)
            losses[i] += loss / 4
    print(f"--- cholec (sampled) ---")
    print("average", losses[0])
    print("0.5", losses[1])
    print("random", losses[2])




def bf_baseline():
    losses = [0, 0, 0]
    for i in range(1, 5):
        trainset = FeatureDataset(
            os.path.join(DATA_ROOT, "breakfast"),
            "features/dense_trajectories",
            f"train_s{i}.txt",
            False,
            1, False,
            False,
            1,
            "none",
            1,
        )
        testset = FeatureDataset(
            os.path.join(DATA_ROOT, "breakfast"),
            "features/dense_trajectories",
            f"test_s{i}.txt",
            False,
            1, False,
            False,
            1,
            "none",
            1,
        )
        for i, loss in enumerate(calc_baseline(trainset, testset)):
            losses[i] += loss / 4

    print(f"--- bf all ---")
    print("average", losses[0])
    print("0.5", losses[1])
    print("random", losses[2])

    losses = [0, 0, 0]
    for i in range(1, 5):
        trainset = FeatureDataset(
            os.path.join(DATA_ROOT, "breakfast"),
            "features/dense_trajectories",
            f"train_s{i}.txt",
            False,
            15, False,
            False,
            1,
            "none",
            1,
        )
        testset = FeatureDataset(
            os.path.join(DATA_ROOT, "breakfast"),
            "features/dense_trajectories",
            f"test_s{i}.txt",
            False,
            15, False,
            False,
            1,
            "none",
            1,
        )
        for i, loss in enumerate(calc_baseline(trainset, testset)):
            losses[i] += loss / 4

    print(f"--- bf all (sampled) ---")
    print("average", losses[0])
    print("0.5", losses[1])
    print("random", losses[2])


def main():
    ucf_baseline()
    cholec_baseline()
    bf_baseline()


if __name__ == "__main__":
    main()

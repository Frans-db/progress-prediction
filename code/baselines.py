import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms

from datasets import FeatureDataset, ImageDataset

DATA_ROOT = "/home/frans/Datasets"

def get_datasetset_lengths(dataset):
    lengths = []
    for item in dataset:
        lengths.append(item[1].shape[0])
    return lengths

def calc_baseline(trainset, testset, plot_name = None):
    train_lengths = get_datasetset_lengths(trainset)
    test_lengths = get_datasetset_lengths(testset)
    
    max_length = max(train_lengths)
    loss = nn.L1Loss(reduction='sum')
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

    # if plot_name:
    #     predictions = torch.ones(max(max(test_lengths), max(train_lengths)))
    #     predictions[:max_length] = averages[:max_length]
    #     plt.plot(predictions, label='progress')
    #     plt.legend(loc='best')
    #     plt.title(f'Average Progress {plot_name}')
    #     plt.xlabel('Frame')
    #     plt.ylabel('Progress (%)')
    #     plt.savefig(f'./plots/{plot_name}.png')
    #     plt.clf()

    return average_loss / count, mid_loss / count, random_loss / count


def ucf_baseline():
    trainset = FeatureDataset(
        os.path.join(DATA_ROOT, "ucf24"),
        "features/i3d_embeddings",
        "train_tubes.txt",
        False,
        False,
        1,
        "none",
        1,
    )
    testset = FeatureDataset(
        os.path.join(DATA_ROOT, "ucf24"),
        "features/i3d_embeddings",
        "test_tubes.txt",
        False,
        False,
        1,
        "none",
        1,
    )
    losses = calc_baseline(trainset, testset, plot_name='ucf101-24')
    print(f'--- ucf ---')
    print('average', losses[0])
    print('0.5', losses[1])
    print('random', losses[2])

def cholec_baseline():
    losses = [0, 0, 0]
    for i in range(4):
        trainset = FeatureDataset(
            os.path.join(DATA_ROOT, "cholec80"),
            "features/i3d_embeddings",
            f"t1_p{i}.txt",
            False,
            False,
            1,
            "none",
            1,
        )
        testset = FeatureDataset(
            os.path.join(DATA_ROOT, "cholec80"),
            "features/i3d_embeddings",
            f"v_p{i}.txt",
            False,
            False,
            1,
            "none",
            1,
        )
        for i, loss in enumerate(calc_baseline(trainset, testset, plot_name=f'cholec_{i}')):
            losses[i] += loss / 4
    print(f'--- cholec ---')
    print('average', losses[0])
    print('0.5', losses[1])
    print('random', losses[2])


def toy_baseline(dataset: str):

    transform = [transforms.ToTensor()]
    transform = transforms.Compose(transform)

    trainset = ImageDataset(
        os.path.join(DATA_ROOT, dataset),
        "rgb-images",
        "train.txt",
        False,
        False,
        1,
        False,
        transform=transform
    )
    testset = ImageDataset(
        os.path.join(DATA_ROOT, dataset),
        "rgb-images",
        "test.txt",
        False,
        False,
        1,
        False,
        transform=transform
    )
    losses = calc_baseline(trainset, testset, plot_name=dataset)
    print(f'--- {dataset} ---')
    print('average', losses[0])
    print('0.5', losses[1])
    print('random', losses[2])


def bf_baseline():
    losses = [0, 0, 0]
    for activity in ['coffee', 'cereals', 'tea', 'milk', 'juice', 'sandwich', 'scrambledegg', 'friedegg', 'salat', 'pancake']:
        for i in range(1, 5):
            trainset = FeatureDataset(
                os.path.join(DATA_ROOT, "breakfast"),
                "features/dense_trajectories",
                f"train_{activity}_s{i}.txt",
                False,
                False,
                1,
                "none",
                1,
            )
            testset = FeatureDataset(
                os.path.join(DATA_ROOT, "breakfast"),
                "features/dense_trajectories",
                f"test_{activity}_s{i}.txt",
                False,
                False,
                1,
                "none",
                1,
            )
            for i, loss in enumerate(calc_baseline(trainset, testset, plot_name=f'bf_{activity}_{i}')):
                losses[i] += loss / 40
        
    print(f'--- bf ---')
    print('average', losses[0])
    print('0.5', losses[1])
    print('random', losses[2])

def bf_baseline_all():
    losses = [0, 0, 0]
    for i in range(1, 5):
        trainset = FeatureDataset(
            os.path.join(DATA_ROOT, "breakfast"),
            "features/dense_trajectories",
            f"train_s{i}.txt",
            False,
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
            False,
            1,
            "none",
            1,
        )
        for i, loss in enumerate(calc_baseline(trainset, testset, plot_name=f'bf_{i}')):
            losses[i] += loss / 4
        
    print(f'--- bf all ---')
    print('average', losses[0])
    print('0.5', losses[1])
    print('random', losses[2])

def main():
    # toy_baseline('bars')
    # toy_baseline('bars_speed')
    # ucf_baseline()
    # cholec_baseline()
    # bf_baseline()
    bf_baseline_all()


if __name__ == "__main__":
    main()

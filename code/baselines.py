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


def calc_baseline(trainset, testset, plot_name=None):
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
    with open('./data/baseline.txt', 'w+') as f:
        f.write('\n'.join(map(str, averages.tolist())))

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
    losses = calc_baseline(trainset, testset, plot_name="ucf101-24")
    print(f"--- ucf ---")
    print("average", losses[0])
    print("0.5", losses[1])
    print("random", losses[2])


def ucf_telic_baseline():
    trainset = FeatureDataset(
        os.path.join(DATA_ROOT, "ucf24"),
        "features/i3d_embeddings",
        "train_telic_tubes.txt",
        False,
        False,
        1,
        "none",
        1,
    )
    testset = FeatureDataset(
        os.path.join(DATA_ROOT, "ucf24"),
        "features/i3d_embeddings",
        "test_telic_tubes.txt",
        False,
        False,
        1,
        "none",
        1,
    )
    losses = calc_baseline(trainset, testset, plot_name="ucf101-24_telic")
    print(f"--- ucf ---")
    print("average", losses[0])
    print("0.5", losses[1])
    print("random", losses[2])


def cholec_baseline():
    losses = [0, 0, 0]
    for i in range(1):
        trainset = FeatureDataset(
            os.path.join(DATA_ROOT, "cholec80"),
            "features/resnet152_0",
            f"t12_p{i}.txt",
            False,
            False,
            1,
            "none",
            1,
        )
        testset = FeatureDataset(
            os.path.join(DATA_ROOT, "cholec80"),
            "features/resnet152_0",
            f"v_p{i}.txt",
            False,
            False,
            1,
            "none",
            1,
        )

        for i, loss in enumerate(
            calc_baseline(trainset, testset, plot_name=f"cholec_{i}")
        ):
            losses[i] += loss
        print(losses)
    print(f"--- cholec ---")
    print("average", losses[0])
    print("0.5", losses[1])
    print("random", losses[2])


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
        transform=transform,
    )
    testset = ImageDataset(
        os.path.join(DATA_ROOT, dataset),
        "rgb-images",
        "test.txt",
        False,
        False,
        1,
        False,
        transform=transform,
    )
    losses = calc_baseline(trainset, testset, plot_name=dataset)
    print(f"--- {dataset} ---")
    print("average", losses[0])
    print("0.5", losses[1])
    print("random", losses[2])


def bf_baseline():
    losses = [0, 0, 0]
    for activity in [
        "coffee",
        "cereals",
        "tea",
        "milk",
        "juice",
        "sandwich",
        "scrambledegg",
        "friedegg",
        "salat",
        "pancake",
    ]:
        for i in range(1, 2):
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
            for i, loss in enumerate(
                calc_baseline(trainset, testset, plot_name=f"bf_{activity}_{i}")
            ):
                losses[i] += loss / 10
            print(losses)

    print(f"--- bf ---")
    print("average", losses[0])
    print("0.5", losses[1])
    print("random", losses[2])


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
        for i, loss in enumerate(calc_baseline(trainset, testset, plot_name=f"bf_{i}")):
            losses[i] += loss / 4

    print(f"--- bf all ---")
    print("average", losses[0])
    print("0.5", losses[1])
    print("random", losses[2])


def plots():
    bf = [
        "bf_1.txt",
        "bf_2.txt",
        "bf_3.txt",
        "bf_4.txt",
    ]
    ucf = ["ucf101-24.txt"]
    cholec80 = [
        "cholec_0.txt",
        "cholec_1.txt",
        "cholec_2.txt",
        "cholec_3.txt",
    ]

    averages = []
    for dataset in [bf, ucf, cholec80]:
        datas = []
        counts = []
        lengths = []
        for predictions in dataset:
            with open(f'data/{predictions}') as f:
                data = torch.FloatTensor([float(row.strip()) for row in f.readlines()])
                count = torch.ones(data.shape[0])

                lengths.append(data.shape[0])
                datas.append(data)
                counts.append(count)
        count = torch.zeros(max(lengths))
        data = torch.zeros(max(lengths))
        for d,c,l in zip(datas,counts, lengths):
            count[:l] += c[:l]
            data[:l] += d[:l]
        
        average = data / count
        averages.append(average)
    
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(6.4*3.5, 4.8*1.5))
    axs = [ax0, ax1, ax2]

    ax0.plot(averages[0])
    ax0.set_title('Breakfast')

    ax1.plot(averages[1])
    ax1.set_title('UCF101-24')

    ax2.plot(averages[2])
    ax2.set_title('cholec80')

    for ax in axs:
        ax.set_xlabel('Frame')
        ax.set_ylabel('Progress')

    plt.savefig('./plots/all.png')
            
            

    # sns.set_theme()
    # print(sns.load_dataset("dots"))
    # plot = sns.relplot(
    #     data=sns.load_dataset("dots"),
    #     kind="line",
    #     x="time",
    #     y="firing_rate",
    #     col="align",
    #     hue='choice',
    # )
    # plot.fig.savefig("./out.png")
    # fig.savefig('./out.png')

def visualisation():
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(6.4*4.5, 4.8*1.5))
    axs = [ax0, ax1, ax2, ax3]

    ax0.plot(list(range(10)), list(map(lambda x: (x+1) / 10, range(10))))
    ax0.set_title('Video 1')

    ax1.plot(list(range(20)), list(map(lambda x: (x+1) / 20, range(20))))
    ax1.set_title('Video 2')

    ax2.plot(list(range(30)), list(map(lambda x: (x+1) / 20, range(30))))
    ax2.set_title('Video 3')

    averages = [0 for i in range(50)]
    for i in range(10):
        averages[i] += ((i + 1) / 10) / 3
        averages[i] += ((i + 1) / 20) / 3
        averages[i] += ((i + 1) / 30) / 3
    for i in range(10):
        averages[i+10] += ((i + 11) / 20) / 2
        averages[i+10] += ((i + 11) / 30) / 2
    for i in range(10):
        averages[i+20] += ((i + 21) / 30) / 1
    for i in range(20):
        averages[i+30] = 1
    ax3.plot(list(range(50)), averages)
    ax3.set_title('Averages')

    for ax in axs:
        ax.set_xlabel('Frame')
        ax.set_ylabel('Progress')
        ax.set_xlim([0, 50])

    plt.savefig('./plots/visualisation.png')

def main():
    for i in range(20):
        with open(f'./data/{i}.txt') as f:
            predictions = list(map(lambda x: float(x.strip()), f.readlines()))
        with open(f'./data/baseline.txt') as f:
            baseline = list(map(lambda x: float(x.strip()), f.readlines()))
        
        plt.plot(predictions, label='predictions')
        plt.plot(baseline, label='baseline')
        plt.legend(loc='best')
        plt.xlabel('Frame')
        plt.ylabel('Progress')
        plt.title(f'Video {i}')
        plt.savefig(f'./plots/{i}.png')
        plt.clf()
    return
    # vids = map(lambda x: f'{x:04d}', range(1, 2326 + 1))
    # train, test = [], []
    # for video_name in vids:
    #     path = f'/home/frans/Datasets/Penn_Action/labels/{video_name}.mat'
    #     mat = scipy.io.loadmat(path)
    #     if mat['action'][0] in ['jump_rope', 'strum_guitar']:
    #         continue

    #     if mat['train'][0][0] == 1:
    #         train.append(video_name)
    #     else:
    #         test.append(video_name)
    # with open('./train.txt', 'w+') as f:
    #     f.write('\n'.join(train))
    # with open('./test.txt', 'w+') as f:
    #     f.write('\n'.join(test))
    
    # random.shuffle(vids)
    # with open('./train.txt', 'w+') as f:
    #     f.write('\n'.join(map(lambda x: f'{x:04d}', sorted(vids[2326 // 2:]))))
    # with open('./test.txt', 'w+') as f:
        # f.write('\n'.join(map(lambda x: f'{x:04d}', sorted(vids[:2326 // 2]))))
    # visualisation()
    # toy_baseline('Penn_Action')
    # toy_baseline('bars_speed')
    # ucf_baseline()
    # cholec_baseline()
    # bf_baseline()
    # bf_baseline_all()


if __name__ == "__main__":
    main()

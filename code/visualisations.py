import os
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List
from PIL import Image

from datasets import FeatureDataset, ImageDataset, UCFDataset

plt.style.use("seaborn-v0_8-paper")
# plt.rc("axes", titlesize=15)  # Controls Axes Title
# plt.rc("axes", labelsize=15)  # Controls Axes Labels
# plt.rc("xtick", labelsize=12)  # Controls x Tick Labels
# plt.rc("ytick", labelsize=12)  # Controls y Tick Labels
# plt.rc("legend", fontsize=13)  # Controls Legend Font
# plt.rc("figure", titlesize=15)  # Controls Figure Title

TITLE_X_OFFSET = 0.5
TITLE_Y_OFFSET = -0.3
DATA_ROOT = "/home/frans/Datasets"


def calc_baseline(train_lengths, test_lengths):
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

    return predictions, average_loss / count


def calculate_average_baseline(train_lengths, test_lengths):
    avg_loss = 0
    max_length = 0
    for train in train_lengths:
        max_length = max(max_length, max(train))
    for test in test_lengths:
        max_length = max(max_length, max(test))

    average_predictions = torch.zeros(max_length)
    for train, test in zip(train_lengths, test_lengths):
        predictions, loss = calc_baseline(train, test)
        average_predictions += predictions / len(train_lengths)
        avg_loss += loss / len(train_lengths)
        print(loss)
    return average_predictions, avg_loss


# def plot_lengths(dataset, ax, title):
#     lengths = dataset.lengths
#     buckets = {}
#     for length in lengths:
#         length = round(length / 10) * 10
#         if length not in buckets:
#             buckets[length] = 0
#         buckets[length] += 1

#     print(title)
#     print(buckets)
#     ax.set_title(title)
#     ax.bar(buckets.keys(), buckets.values(), 5)


# def visualise_lengths():
#     bucket_size = 10
#     buckets = {}
#     lengths = [round(l / bucket_size) * bucket_size for l in dataset.lengths]
#     for length in lengths:
#         if length not in buckets:
#             buckets[length] = 0
#         buckets[length] += 1
#     plt.bar(buckets.keys(), buckets.values(), width=bucket_size * 0.9)
#     plt.savefig('./plots/test.png')

# figure, axs = plt.subplots(2, 3)
# plot_lengths(breakfast, axs[0,0], 'breakfast')
# plot_lengths(breakfast_sampled, axs[1,0], 'breakfast (sampled)')

# plot_lengths(cholec80, axs[0,1], 'cholec80')
# plot_lengths(cholec80_sampled, axs[1,1], 'cholec80 (sampled)')

# plot_lengths(ucf24, axs[0,2], 'ucf24')
# for ax in axs.flat:
#     ax.set(xlabel='length', ylabel='amount')

# axs[1,2].set_visible(False)()
# plt.savefig('./plots/lengths.pdf')
# plt.savefig('./plots/lengths.png')


def baseline_example():
    predictions, average_loss = calc_baseline([10, 20, 30], [50])

    figure, axs = plt.subplots(1, 2, figsize=(12.8, 4.8))
    axs[0].plot([0] + [(i + 1) / 10 for i in range(10)], label="Video 1")
    axs[0].plot([0] + [(i + 1) / 20 for i in range(20)], label="Video 2")
    axs[0].plot([0] + [(i + 1) / 30 for i in range(30)], label="Video 3")
    axs[1].plot(predictions, label="predictions")
    for ax in axs.flat:
        ax.set_xlabel("Frame")
        ax.set_ylabel("Progress")
        ax.set_xlim(0, 50)
        # ax.set(xlabel='Frame', ylabel='Progress')
        ax.legend()
    axs[0].set_title(
        "(a) 3 Videos of length 10, 20, and 30", y=TITLE_Y_OFFSET, x=TITLE_X_OFFSET
    )
    axs[1].set_title("(b) Average Index baseline", y=TITLE_Y_OFFSET, x=TITLE_X_OFFSET)
    # for ax in axs.flat:
    # ax.label_outer()
    plt.tight_layout()
    plt.savefig("./plots/avg_index_example.pdf")
    plt.savefig("./plots/avg_index_example.png")
    plt.clf()


def baselines():
    train_lengths = []
    test_lengths = []
    for i in range(1, 5):
        train = ImageDataset(os.path.join(DATA_ROOT, "breakfast"), "rgb-images", f"train_s{i}.txt")
        test = ImageDataset(os.path.join(DATA_ROOT, "breakfast"), "rgb-images", f"test_s{i}.txt")
        train_lengths.append(train.lengths)
        test_lengths.append(test.lengths)
    breakfast_predictions, bf_loss = calculate_average_baseline(
        train_lengths, test_lengths
    )

    train_lengths = []
    test_lengths = []
    for i in range(0, 4):
        train = ImageDataset(os.path.join(DATA_ROOT, "cholec80"), "rgb-images", f"t12_{i}.txt", subsample_fps=10)
        test = ImageDataset(os.path.join(DATA_ROOT, "cholec80"), "rgb-images", f"e_{i}.txt", subsample_fps=10)
        train_lengths.append(train.lengths)
        test_lengths.append(test.lengths)
    cholec_predictions, cholec_loss = calculate_average_baseline(
        train_lengths, test_lengths
    )

    train_lengths = UCFDataset(os.path.join(DATA_ROOT, "ucf24"), "rgb-images", f"train.txt").lengths
    test_lengths = UCFDataset(os.path.join(DATA_ROOT, "ucf24"), "rgb-images", f"test.txt").lengths
    ucf_predictions, ucf_loss = calc_baseline(train_lengths, test_lengths)
    print(f'BF: {bf_loss:.2f}')
    print(f'C80: {cholec_loss:.2f}')
    print(f'UCF: {ucf_loss:.2f}')

    figure, axs = plt.subplots(1, 3, figsize=(19.2, 4.8 * 1.3))
    axs[0].plot(breakfast_predictions, label="Average Index")
    axs[1].plot(cholec_predictions, label="Average Index")
    axs[2].plot(ucf_predictions, label="Average Index")

    with open('./data/bf_baseline.txt', 'w+') as f:
        f.write('\n'.join([str(val) for val in breakfast_predictions.tolist()]))
    with open('./data/cholec_baseline.txt', 'w+') as f:
        f.write('\n'.join([str(val) for val in cholec_predictions.tolist()]))
    with open('./data/ucf_baseline.txt', 'w+') as f:
        f.write('\n'.join([str(val) for val in ucf_predictions.tolist()]))
    # for ax in axs.flat:
    #     ax.set_xlabel("Frame")
    #     ax.set_ylabel("Progress")
    #     ax.legend()
    # axs[0].set_title(
    #     "(a) Average index baseline on Breakfast",
    #     y=TITLE_Y_OFFSET / 1.3,
    #     x=TITLE_X_OFFSET,
    # )
    # axs[1].set_title(
    #     "(b) Average index baseline on Cholec80",
    #     y=TITLE_Y_OFFSET / 1.3,
    #     x=TITLE_X_OFFSET,
    # )
    # axs[2].set_title(
    #     "(c) Average index baseline on UCF101-24",
    #     y=TITLE_Y_OFFSET / 1.3,
    #     x=TITLE_X_OFFSET,
    # )

    # print(bf_loss, cholec_loss, ucf_loss)
    # plt.tight_layout()
    # plt.savefig("./plots/avg_index_baseline.pdf")
    # plt.savefig("./plots/avg_index_baseline.png")
    # plt.clf()
# BASELINE_COLOUR = 'blue'
# LEARNER_COLOUR = 'red'
# METHOD_COLOUR = 'green'
theme = {
    'Average Index': ('blue', ':'),
    'Static 0.5': ('red', ':'),
    'ResNet-2D': ('blue', '-.'),
    'ResNet-LSTM': ('red', '-.'),
    'RSDNet': ('blue', '-'),
    'UTE': ('red', '-') 
}

def visualise_video(frame_path: str, result_paths: List[str], length: int):
    frame = Image.open(frame_path)
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 4.2))

    truth = [(i + 1) / length for i in range(length)]
    axs[1].plot(truth, label='Ground Truth', color='purple')

    static = [0.5 for _ in range(length)]
    axs[1].plot(static, label='Static 0.5', color=theme['Static 0.5'][0], linestyle=theme['Static 0.5'][1])

    for name, path in result_paths:
        with open(path) as f:
            data = [float(row.strip()) for row in f.readlines()][:length]
        axs[1].plot(data, label=name, color=theme[name][0], linestyle=theme[name][1])


    axs[0].imshow(frame)
    axs[0].axis('off')
    axs[1].set_xlabel('Frame')
    axs[1].set_ylabel('Progress')

    plt.grid(axis='y')
    yticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ytick_labels = ['0%', '20%', '40%', '60%', '80%', '100%']
    axs[1].tick_params(axis='y', length = 0)
    axs[1].set_yticks(yticks, ytick_labels)
    axs[1].set_xlim(0, length)


    # axs[1].legend()
    plt.legend()
    plt.tight_layout()
    plt.savefig('./plots/test.png')
    plt.clf()

        # plt.axis('off')
        # plt.tight_layout()
        # plt.savefig(f'./plots/bars/{i:03d}.jpg')
        # plt.savefig(f'./plots/bars/{i:03d}.pdf')
        # plt.clf()

def main():
    baselines()
    visualise_video(
        "/home/frans/Datasets/cholec80/rgb-images/video04/frame_017026.jpg",
        [('Average Index', './data/cholec_baseline.txt'), 
         ('ResNet-LSTM', './data/lstm_cholec/video04.txt'), 
         ('RSDNet', './data/rsd_cholec/video04.txt'),
         ('UTE', './data/ute_cholec/video04_0.txt')],
        153
    )
    # baseline_example()
    # visualise_lengths()


if __name__ == "__main__":
    main()

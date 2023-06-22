import os
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List
from PIL import Image

from datasets import FeatureDataset, ImageDataset, UCFDataset



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

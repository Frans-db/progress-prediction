import json
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn as nn
import torch
import statistics
from PIL import Image
import string

from datasets import FeatureDataset, ImageDataset, UCFDataset

# Constants
DATA_ROOT = "/home/frans/Datasets"
BAR_WIDTH = 0.5
SPACING = 1.5
MODE_COLOURS = {
    "full video": (0.2980392156862745, 0.4470588235294118, 0.6901960784313725),
    "random": (0.8666666666666667, 0.5176470588235295, 0.3215686274509804),
    "video segments": (0.3333333333333333, 0.6588235294117647, 0.40784313725490196),
    "indices": (0.7686274509803922, 0.3058823529411765, 0.3215686274509804),
}
BASELINES = ["Average Index", "Static 0.5", "Random"]
LINEWIDTH = 2
TITLE_X_OFFSET = 0.5
TITLE_Y_OFFSET = -0.25

# Matplotlib parameters
plt.style.use("seaborn-v0_8-paper")
plt.rcParams['axes.axisbelow'] = True

def set_spines(enable: bool):
    plt.rcParams['axes.spines.left'] = enable
    plt.rcParams['axes.spines.right'] = enable
    plt.rcParams['axes.spines.top'] = enable
    plt.rcParams['axes.spines.bottom'] = enable


# Datasets
if not os.path.isfile('./data/lengths.json'):
    print('loading ucf101')
    ucf101 = {
        'all': sorted(UCFDataset(os.path.join(DATA_ROOT, "ucf24"), "rgb-images", f"all.txt").lengths),
        'train': [sorted(UCFDataset(os.path.join(DATA_ROOT, "ucf24"), "rgb-images", f"train.txt").lengths)],
        'test': [sorted(UCFDataset(os.path.join(DATA_ROOT, "ucf24"), "rgb-images", f"test.txt").lengths)]
    }
    print('loading cholec80')
    cholec80 = {
        'all': sorted(ImageDataset(os.path.join(DATA_ROOT, "cholec80"), "rgb-images", f"all_0.txt").lengths),
        'train': [sorted(ImageDataset(os.path.join(DATA_ROOT, "cholec80"), "rgb-images", f"t12_{i}.txt").lengths) for i in range(4)],
        'test': [sorted(ImageDataset(os.path.join(DATA_ROOT, "cholec80"), "rgb-images", f"e_{i}.txt").lengths) for i in range(4)],
    }
    print('loading breakfast')
    breakfast = {
        'all': sorted(ImageDataset(os.path.join(DATA_ROOT, "breakfast"), "rgb-images", f"all.txt").lengths),
        'train': [sorted(ImageDataset(os.path.join(DATA_ROOT, "breakfast"), "rgb-images", f"train_s{i}.txt").lengths) for i in range(1, 5)],
        'test': [sorted(ImageDataset(os.path.join(DATA_ROOT, "breakfast"), "rgb-images", f"test_s{i}.txt").lengths) for i in range(1, 5)],
    }
    with open('./data/lengths.json', 'w+') as f:
        json.dump({
            'ucf101': ucf101,
            'cholec80': cholec80,
            'breakfast': breakfast
        }, f)
else:
    with open('./data/lengths.json') as f:
        data = json.load(f)
    ucf101 = data['ucf101']
    cholec80 = data['cholec80']
    breakfast = data['breakfast']

# Helper functions
def load_results(path: str):
    with open(path) as f:
        results = json.load(f)
    return results

def calc_baselines(train_lengths, test_lengths):
    max_length = max(train_lengths)
    loss = nn.L1Loss(reduction="sum")
    averages = torch.zeros(max(train_lengths))
    counts = torch.zeros(max(train_lengths))
    for length in train_lengths:
        progress = torch.arange(1, length + 1) / length
        averages[:length] += progress
        counts[:length] += 1
    averages = averages / counts

    index_loss, static_loss, random_loss, count = 0, 0, 0, 0
    for length in test_lengths:
        l = min(length, max_length)
        progress = torch.arange(1, length + 1) / length

        average_predictions = torch.ones(length)
        average_predictions[:l] = averages[:l]
        index_loss += loss(average_predictions * 100, progress * 100).item()
        static_loss += loss(torch.full_like(progress, 0.5) * 100, progress * 100).item()
        random_loss += loss(torch.rand_like(progress) * 100, progress * 100).item()

        count += length

    length = max(max(test_lengths), max_length)
    predictions = torch.ones(length)
    predictions[:max_length] = averages[:max_length]

    return predictions, index_loss / count, static_loss / count, random_loss / count

def calculate_average_baseline(trains, tests):
    num_sets = len(trains)
    max_length = 0
    for (train, test) in zip(trains, tests):
        max_length = max(max_length, max(train), max(test))

    average_predictions = torch.zeros(max_length)
    avg_index_loss, avg_static_loss, avg_random_loss = 0, 0, 0
    for (train, test) in zip(trains, tests):
        predictions, index_loss, static_loss, random_loss = calc_baselines(train, test)
        avg_index_loss += index_loss / num_sets
        avg_static_loss += static_loss / num_sets
        avg_random_loss += random_loss / num_sets
        average_predictions += predictions / num_sets
    
    return average_predictions, avg_index_loss, avg_static_loss, avg_random_loss

# Plots
def plot_result_bar(results: Dict, dataset: str, modes: List[str]):
    set_spines(False)
    plt.figure(figsize=(7.2, 4.2))

    data = [[] for _ in modes]
    networks = [key for key in results[dataset] if key not in BASELINES]
    xs_indices = np.array([0, 1, 3, 4, 5])
    for network in networks:
        if network in BASELINES:
            continue
        for i, mode in enumerate(modes):
            if mode in results[dataset][network]:
                data[i].append(results[dataset][network][mode])
            else:
                data[i].append(0)

    for i, (values, mode) in enumerate(zip(data, modes)):
        bar_xs = xs_indices * SPACING + i * BAR_WIDTH
        plt.bar(bar_xs, values, width=BAR_WIDTH, label=mode, color=MODE_COLOURS[mode])
    xticks = xs_indices * SPACING + BAR_WIDTH * 0.5
    if dataset != 'Bars':
        plt.axhline(
            y=results[dataset]["Average Index"]["full video"],
            linestyle="-",
            label="Average Index",
            color=(0.8, 0.7254901960784313, 0.4549019607843137),
            linewidth=LINEWIDTH,
        )
        plt.axhline(
            y=results[dataset]["Static 0.5"]["full video"],
            linestyle="-",
            label="Static 0.5",
            color=(0.8549019607843137, 0.5450980392156862, 0.7647058823529411),
            linewidth=LINEWIDTH,
        )
        plt.axhline(
            y=results[dataset]["Random"]["full video"],
            linestyle="-",
            label="Random",
            color=(0.5058823529411764, 0.4470588235294118, 0.7019607843137254),
            linewidth=LINEWIDTH,
        )
        plt.text(
            3,
            results[dataset]["Average Index"]["full video"] - 0.5,
            str(results[dataset]["Average Index"]["full video"]),
        ).set_bbox({"facecolor": "white", "edgecolor": "white"})
        plt.text(
            3,
            results[dataset]["Static 0.5"]["full video"] - 0.5,
            str(results[dataset]["Static 0.5"]["full video"]),
        ).set_bbox({"facecolor": "white", "edgecolor": "white"})
        plt.text(
            3,
            results[dataset]["Random"]["full video"] - 0.5,
            str(results[dataset]["Random"]["full video"]),
        ).set_bbox({"facecolor": "white", "edgecolor": "white"})

    plt.grid(axis='y')
    plt.axhline(y=0,linestyle='-', color='grey', zorder=-1)

    plt.xticks(xticks, networks)
    if dataset == 'Bars':
        yticks = [0, 5, 10]
    else:
        yticks = [0, 5, 10, 15, 20, 25, 30, 35]
    plt.tick_params(axis='y', length = 0)
    plt.yticks(yticks, [f'{tick}%' for tick in yticks])
    plt.ylabel("MAE")
    plt.legend(loc='upper right')
    plt.tight_layout()
    filename = f"./plots/results/{dataset}_{'_'.join(modes).replace(' ', '_')}"
    plt.savefig(f"{filename}.pdf")
    plt.savefig(f"{filename}.png")
    plt.clf()
    set_spines(True)

def plot_baselines():
    c80_predictions, c80_index_loss, c80_static_loss, c80_random_loss = calculate_average_baseline(cholec80['train'], cholec80['test'])
    bf_predictions, bf_index_loss, bf_static_loss, bf_random_loss = calculate_average_baseline(breakfast['train'], breakfast['test'])
    ucf_predictions, ucf_index_loss, ucf_static_loss, ucf_random_loss = calculate_average_baseline(ucf101['train'], ucf101['test'])
    print('c80', c80_index_loss, c80_static_loss, c80_random_loss)
    print('bf', bf_index_loss, bf_static_loss, bf_random_loss)
    print('ucf', ucf_index_loss, ucf_static_loss, ucf_random_loss)

    figure, axs = plt.subplots(1, 3, figsize=(19.2, 4.8 * 1.3))
    axs[0].plot(ucf_predictions, label="Average Index")
    axs[1].plot(c80_predictions, label="Average Index")
    axs[2].plot(bf_predictions, label="Average Index")

    with open('./data/ucf_baseline.txt', 'w+') as f:
        f.write('\n'.join([str(val) for val in ucf_predictions.tolist()]))
    with open('./data/cholec_baseline.txt', 'w+') as f:
        f.write('\n'.join([str(val) for val in c80_predictions.tolist()]))
    with open('./data/bf_baseline.txt', 'w+') as f:
        f.write('\n'.join([str(val) for val in bf_predictions.tolist()]))


    for ax in axs.flat:
        ax.set_xlabel("Frame")
        ax.set_ylabel("Progress")
        ax.legend()
    axs[0].set_title(
        "(a) Average index baseline on UCF101-24",
        y=TITLE_Y_OFFSET / 1.3,
        x=TITLE_X_OFFSET,
    )
    axs[1].set_title(
        "(b) Average index baseline on Cholec80",
        y=TITLE_Y_OFFSET / 1.3,
        x=TITLE_X_OFFSET,
    )
    axs[2].set_title(
        "(c) Average index baseline on Breakfast",
        y=TITLE_Y_OFFSET / 1.3,
        x=TITLE_X_OFFSET,
    )

    plt.tight_layout()
    plt.savefig("./plots/baselines/avg_index_baseline.pdf")
    plt.savefig("./plots/baselines/avg_index_baseline.png")
    plt.clf()

def plot_baseline_example():
    predictions, _, _, _ = calc_baselines([10, 20, 30], [50])

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
    axs[0].set_title("(a) 3 Videos of length 10, 20, and 30", y=TITLE_Y_OFFSET, x=TITLE_X_OFFSET)
    axs[1].set_title("(b) Average Index baseline", y=TITLE_Y_OFFSET, x=TITLE_X_OFFSET)
    plt.tight_layout()
    plt.savefig("./plots/baselines/avg_index_example.pdf")
    plt.savefig("./plots/baselines/avg_index_example.png")
    plt.clf()

def make_length_plot(lengths, ax: plt.Axes, title: str, bucket_size: int = 10):
    buckets = {}
    q1 = np.percentile(lengths, 25)
    mean = np.percentile(lengths, 50)
    q3 = np.percentile(lengths, 65)
    for length in lengths:
        length = round(length / bucket_size) * bucket_size
        if length not in buckets:
            buckets[length] = 0
        buckets[length] += 1
    ax.bar(buckets.keys(), buckets.values(), width=bucket_size)
    # ax.axvline(q1, color='red', linestyle=':')
    ax.axvline(mean, color='red')
    # ax.axvline(q3, color='red', linestyle=':')
    # title = title + f'\nQ1={round(q1)}, mean={round(mean)}, Q3={round(q3)}'
    ax.set_title(title, y=TITLE_Y_OFFSET * 1.5, x=TITLE_X_OFFSET)


def plot_dataset_lengths():
    figure, axs = plt.subplots(1, 3, figsize=(6.4 * 1.5, 4.4 / 1.5))
    make_length_plot(ucf101['all'], axs[0], '(a) video length distribution for ucf101-24')
    make_length_plot(cholec80['all'], axs[1], '(b) video length distribution for cholec80', bucket_size=100)
    make_length_plot(breakfast['all'], axs[2], '(c) video length distribution for breakfast', bucket_size=100)
    for ax in axs.flat:
        ax.set_xlabel("Video Length")
        ax.set_ylabel("Number of Videos")
    plt.tight_layout()
    plt.savefig('./plots/dataset_lengths.pdf')
    plt.savefig('./plots/dataset_lengths.png')
    plt.clf()

def plot_synthetic(video_index: int, frame_indices: List[int]):
    data_dir = os.path.join(DATA_ROOT, 'bars/rgb-images')
    video_name = sorted(os.listdir(data_dir))[video_index]
    video_path = os.path.join(data_dir, video_name)
    frame_names = sorted(os.listdir(video_path))
    num_frames = len(frame_names)
    
    figure, axs = plt.subplots(1, len(frame_indices), figsize=(6.4, 2.4))
    for letter, ax, frame_index in zip(string.ascii_lowercase, axs, frame_indices):
        frame_name = frame_names[frame_index]
        frame_path = os.path.join(video_path, frame_name)
        frame = Image.open(frame_path)
        ax.imshow(frame)
        ax.axis('off')
        progress = (frame_index + 1) / num_frames
        ax.set_title(f'({letter}) \nt={frame_index}\np={round(progress * 100, 1)}%', y=TITLE_Y_OFFSET * 2.2, x=TITLE_X_OFFSET)

    plt.tight_layout()
    plt.savefig('./plots/bars.png')
    plt.savefig('./plots/bars.pdf')
    plt.clf()



def main():
    # result plots
    results = load_results('./results.json')
    for dataset in ["UCF101-24", "Cholec80", "Breakfast"]:
        plot_result_bar(results, dataset, ["full video", "random"])
        plot_result_bar(results, dataset, ["video segments", "indices"])
    plot_result_bar(results, "Bars", ["full video", "video segments"])
    # average index baseline
    plot_baselines()
    plot_baseline_example()
    # dataset statistics
    plot_dataset_lengths()
    # syntethic dataset example
    plot_synthetic(4, [0, 15, 35, 58, 71])
    # example progress predictions

if __name__ == '__main__':
    main()
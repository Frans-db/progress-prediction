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
import pandas as pd
import os
import argparse
import math
import torchvision.transforms as transforms
import random

from datasets import FeatureDataset, ImageDataset, UCFDataset
from datasets import Middle

# Constants
DATA_ROOT = "/home/frans/Datasets"
BAR_WIDTH = 0.5
SPACING = 1.5
MODE_COLOURS = {
    "'full-video' inputs": (0.2980392156862745, 0.4470588235294118, 0.6901960784313725),
    "'random-noise' inputs": (0.8666666666666667, 0.5176470588235295, 0.3215686274509804),
    "'video-segments' inputs": (0.3333333333333333, 0.6588235294117647, 0.40784313725490196),
    "'frame-indices' inputs": (0.7686274509803922, 0.3058823529411765, 0.3215686274509804),
}
BASELINES = ["average-index", "static-0.5", "random"]
LINEWIDTH = 3
TITLE_X_OFFSET = 0.5
TITLE_Y_OFFSET = -0.25

# Matplotlib parameters
plt.style.use("seaborn-v0_8-paper")
plt.rcParams['axes.axisbelow'] = True


def set_font_sizes(small=12, medium=14, big=16):
    plt.rc('font', size=small)          # controls default text sizes
    plt.rc('axes', titlesize=small)     # fontsize of the axes title
    plt.rc('axes', labelsize=medium)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=small)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=small)    # fontsize of the tick labels
    plt.rc('legend', fontsize=small)    # legend fontsize
    plt.rc('figure', titlesize=big)  # fontsize of the figure title

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
    print('loading bars')
    bars = {
        'all': sorted(ImageDataset(os.path.join(DATA_ROOT, "bars"), "rgb-images", f"all.txt").lengths),
        'train': [sorted(ImageDataset(os.path.join(DATA_ROOT, "bars"), "rgb-images", f"train.txt").lengths)],
        'test': [sorted(ImageDataset(os.path.join(DATA_ROOT, "bars"), "rgb-images", f"test.txt").lengths)]
    }
    with open('./data/lengths.json', 'w+') as f:
        json.dump({
            'ucf101': ucf101,
            'cholec80': cholec80,
            'breakfast': breakfast,
            'bars': bars
        }, f)
else:
    with open('./data/lengths.json') as f:
        data = json.load(f)
    ucf101 = data['ucf101']
    cholec80 = data['cholec80']
    breakfast = data['breakfast']
    bars = data['bars']

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
def plot_result_bar(results: Dict, dataset: str, modes: List[str], names: List[str]):
    set_spines(False)
    plt.figure(figsize=(7.2, 5.2))

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
        print(bar_xs, values, mode, dataset)
        plt.bar(bar_xs, values, width=BAR_WIDTH, label=mode, color=MODE_COLOURS[mode])
    xticks = xs_indices * SPACING + BAR_WIDTH * 0.5
    if dataset != 'Bars':
        plt.axhline(
            y=results[dataset]["average-index"]["'full-video' inputs"],
            linestyle="-",
            label="average-index",
            color=(0.8, 0.7254901960784313, 0.4549019607843137),
            linewidth=LINEWIDTH,
        )
        plt.axhline(
            y=results[dataset]["static-0.5"]["'full-video' inputs"],
            linestyle="-",
            label="static-0.5",
            color=(0.8549019607843137, 0.5450980392156862, 0.7647058823529411),
            linewidth=LINEWIDTH,
        )
        plt.axhline(
            y=results[dataset]["random"]["'full-video' inputs"],
            linestyle="-",
            label="random",
            color=(0.5058823529411764, 0.4470588235294118, 0.7019607843137254),
            linewidth=LINEWIDTH,
        )
        plt.text(
            3,
            results[dataset]["average-index"]["'full-video' inputs"] - 0.5,
            str(results[dataset]["average-index"]["'full-video' inputs"]) + '%',
        ).set_bbox({"facecolor": "white", "edgecolor": "white"})
        plt.text(
            3,
            results[dataset]["static-0.5"]["'full-video' inputs"] - 0.5,
            str(results[dataset]["static-0.5"]["'full-video' inputs"]) + '%',
        ).set_bbox({"facecolor": "white", "edgecolor": "white"})
        plt.text(
            3,
            results[dataset]["random"]["'full-video' inputs"] - 0.5,
            str(results[dataset]["random"]["'full-video' inputs"]) + '%',
        ).set_bbox({"facecolor": "white", "edgecolor": "white"})

    plt.grid(axis='y')
    plt.axhline(y=0,linestyle='-', color='grey', zorder=-1)

    plt.xticks(xticks, networks)
    if dataset == 'Bars':
        yticks = [0, 1, 2, 3, 4]
    else:
        yticks = [0, 5, 10, 15, 20, 25, 30, 35]
    plt.tick_params(axis='y', length = 0)
    plt.yticks(yticks, [f'{tick}%' for tick in yticks])
    plt.ylabel("MAE")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=False, shadow=False, ncol=3)
    plt.tight_layout()
    filename = f"./plots/results/{dataset}_{'_'.join(names).replace(' ', '_')}"
    plt.savefig(f"{filename}.{FILE}")
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
    axs[0].plot(ucf_predictions, label="average-index")
    axs[1].plot(c80_predictions, label="average-index")
    axs[2].plot(bf_predictions, label="average-index")

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
        "(a) Average-index baseline on UCF101-24",
        y=TITLE_Y_OFFSET / 1.2,
        x=TITLE_X_OFFSET,
    )
    axs[1].set_title(
        "(b) Average-index baseline on Cholec80",
        y=TITLE_Y_OFFSET / 1.2,
        x=TITLE_X_OFFSET,
    )
    axs[2].set_title(
        "(c) Average-index baseline on Breakfast",
        y=TITLE_Y_OFFSET / 1.2,
        x=TITLE_X_OFFSET,
    )

    plt.tight_layout()
    plt.savefig(f"./plots/avg_index_baseline.{FILE}")
    plt.clf()

def plot_baseline_example():
    predictions, _, _, _ = calc_baselines([10, 20, 30], [50])

    plt.figure(figsize=(6.4*1.5, 4.8*1.5))
    # figure, axs = plt.subplots(1, 1, figsize=(12.8, 4.8))
    # axs[0].plot([(i + 1) / 10 for i in range(10)], label="Video 1")
    # axs[0].plot([(i + 1) / 20 for i in range(20)], label="Video 2")
    # axs[0].plot([(i + 1) / 30 for i in range(30)], label="Video 3")
    # axs[0].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0%', '20%', '40%', '60%', '80%', '100%'])
    # axs[1].plot([(i + 1) / 10 for i in range(10)], label="Video 1", linestyle=':')
    # axs[1].plot([(i + 1) / 20 for i in range(20)], label="Video 2", linestyle=':')
    # axs[1].plot([(i + 1) / 30 for i in range(30)], label="Video 3", linestyle=':')
    plt.plot([random.random() for _ in predictions], label='random', linestyle=':', linewidth=3)
    plt.plot([0.5 for _ in predictions], label='static-0.5', linewidth=3)
    plt.plot(predictions, label="average-index", linewidth=3)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0%', '20%', '40%', '60%', '80%', '100%'])
    plt.xlabel('Frame')
    plt.ylabel('Progress')
    plt.xlim(0, 50)
    plt.legend()
    # for ax in axs.flat:
    #     ax.set_xlabel("Frame")
    #     ax.set_ylabel("Progress")
    #     ax.set_xlim(0, 50)
    #     # ax.set(xlabel='Frame', ylabel='Progress')
    #     ax.legend()
    # axs[0].set_title("(a) 3 Videos of length 10, 20, and 30", y=TITLE_Y_OFFSET, x=TITLE_X_OFFSET)
    # axs[1].set_title("(b) Average-index baseline", y=TITLE_Y_OFFSET, x=TITLE_X_OFFSET)
    plt.tight_layout()
    plt.savefig(f"./plots/avg_index_example.{FILE}")
    plt.clf()

def make_length_plot(lengths, ax: plt.Axes, title: str, bucket_size: int = 10):
    buckets = {}
    q1 = np.percentile(lengths, 25)
    mean = np.percentile(lengths, 50)
    q3 = np.percentile(lengths, 65)
    for length in lengths:
        length = math.floor(length / bucket_size) * bucket_size
        if length not in buckets:
            buckets[length] = 0
        buckets[length] += 1
    ax.bar(buckets.keys(), buckets.values(), width=bucket_size)
    # ax.axvline(q1, color='red', linestyle=':')
    ax.axvline(mean, color='red')
    # ax.axvline(q3, color='red', linestyle=':')
    # title = title + f'\nQ1={round(q1)}, mean={round(mean)}, Q3={round(q3)}'
    ax.set_title(title, y=TITLE_Y_OFFSET*1.1, x=TITLE_X_OFFSET)


def plot_dataset_lengths():
    figure, axs = plt.subplots(1, 3, figsize=(6.4 * 2, 4.4))
    make_length_plot(ucf101['all'], axs[0], '(a) video length distribution for UCF101-24', bucket_size=10)
    make_length_plot(cholec80['all'], axs[1], '(b) video length distribution for Cholec80', bucket_size=100)
    make_length_plot(breakfast['all'], axs[2], '(c) video length distribution for Breakfast', bucket_size=100)
    # make_length_plot(bars['all'], axs[3], '(d) video length distribution for synthetic dataset', bucket_size=10)
    for ax in axs.flat:
        ax.set_xlabel("Video Length")
        ax.set_ylabel("Number of Videos")
    plt.tight_layout()
    plt.savefig(f'./plots/dataset_lengths.{FILE}')
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
    plt.savefig(f'./plots/bars.{FILE}')
    plt.clf()


def visualise_video(video_dir: str, timestamps: List[int], result_paths: List[str], video_name: str, N: int, offset: int = 0, subsample: int = 1):
    gs = plt.GridSpec(len(timestamps), 4)
    fig = plt.figure(figsize=(6.4*2, 4.8*1.5))
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[1, 0])
    ax3 = plt.subplot(gs[:, 1:])
    axs = [ax1, ax2, ax3]

    # fig, axs = plt.subplots(1, 2, figsize=(6.4*2, 4.8*1.5), gridspec_kw={'width_ratios': [1, 3]})
    frames = sorted(os.listdir(video_dir))[::subsample]
    num_frames = len(frames)
    index = timestamps[0]
    for i,index in enumerate(timestamps):
        frame_path = os.path.join(video_dir, frames[index+offset])
        frame = Image.open(frame_path)
        axs[i].imshow(frame)
        axs[i].axis('off')
        axs[i].set_title(f't={index}', fontsize=20)

    ground_truth = [(i+1) / num_frames for i in range(num_frames)]
    static = [0.5 for _ in range(num_frames)]
    for index in timestamps:
        axs[-1].axvline(index, color='red', linestyle=':')
    for name, path, linestyle in result_paths:
        with open(path) as f:
            data = [float(row.strip()) for row in f.readlines()][:num_frames]
        if name != 'average-index':
            axs[-1].plot(np.convolve(data, np.ones(N)/N, mode='full'), label=name, linestyle=linestyle, linewidth=LINEWIDTH)
        else:
            axs[-1].plot(data, label=name, linewidth=LINEWIDTH)

    axs[-1].plot(ground_truth, label='Ground Truth', linewidth=LINEWIDTH)
    axs[-1].set_xlabel('Frame')
    axs[-1].set_ylabel('Progress')
    axs[-1].tick_params(axis='both', which='major')
    axs[-1].tick_params(axis='both', which='minor')

    plt.grid(axis='y')
    yticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ytick_labels = ['0%', '20%', '40%', '60%', '80%', '100%']
    axs[-1].tick_params(axis='y', length = 0)
    axs[-1].set_yticks(yticks, ytick_labels)
    axs[-1].set_xlim(0, num_frames)

    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(f'./plots/examples/{video_name}.{FILE}')
    plt.clf()

def stats(dataset: str, splitfiles: List[str], length=False):
    root = os.path.join(DATA_ROOT, dataset)
    for splitfile in splitfiles:
        with open(f'{os.path.join(root, f"splitfiles/{splitfile}.txt")}') as f:
            lines = f.readlines()
        counts_per_class = {}
        num_frames_per_class = {}
        total = 0
        for line in lines:
            line = line.strip()
            if length:
                frames_path = os.path.join(root, 'rgb-images', line)
                num_frames = len(os.listdir(frames_path))
            activity_class = line.split('/')[0]
            if activity_class not in counts_per_class:
                counts_per_class[activity_class] = 0
            if activity_class not in num_frames_per_class:
                num_frames_per_class[activity_class] = 0
            if length:
                num_frames_per_class[activity_class] += num_frames
            counts_per_class[activity_class] += 1
            total += 1
        
        plt.bar(counts_per_class.keys(), [num_frames_per_class[key] / counts_per_class[key] for key in counts_per_class])
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f'./plots/{dataset}_{splitfile}.png')
        plt.clf()
        print(f'--- {total} videos in {dataset}/{splitfile} ({len(counts_per_class)} classes) ---')
        for activity_class in counts_per_class:
            print(f'{activity_class}: {counts_per_class[activity_class]} ({num_frames_per_class[activity_class] / counts_per_class[activity_class]})')

def tube_stats(splitfile: str):
    dataset = FeatureDataset(os.path.join(DATA_ROOT, 'ucf24'), 'features/resnet152', splitfile, False, 1, False, False, 1, 'none', 1)
    counts_per_class = {}
    num_frames_per_class = {}
    total = 0
    for (name, data, _) in dataset:
        length = data.shape[0]
        activity_class = name.split('/')[0]
        if activity_class not in counts_per_class:
            counts_per_class[activity_class] = 0
        if activity_class not in num_frames_per_class:
            num_frames_per_class[activity_class] = 0
        if length:
            num_frames_per_class[activity_class] += length
        counts_per_class[activity_class] += 1
        total += 1
        plt.bar(counts_per_class.keys(), [num_frames_per_class[key] / counts_per_class[key] for key in counts_per_class])
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f'./plots/ucf24__{splitfile}.png')
        plt.clf()
    print(f'--- {total} videos in ucf24/{splitfile} ({len(counts_per_class)} classes) ---')
    for activity_class in counts_per_class:
        print(f'{activity_class}: {counts_per_class[activity_class]} ({num_frames_per_class[activity_class] / counts_per_class[activity_class]})')

def dataset_statistics():
    stats('ucf24', ['all', 'train', 'test'], length=True)
    tube_stats('all_tubes.txt')
    stats('breakfast', ['all'], length=True)

def surgery_duration():
    durations = [num_frames / 60 for num_frames in cholec80['all']]

    buckets = {}
    q1 = np.percentile(durations, 25)
    mean = np.percentile(durations, 50)
    q3 = np.percentile(durations, 65)
    for length in durations:
        length = math.floor(length / 10) * 10
        if length not in buckets:
            buckets[length] = 0
        buckets[length] += 1

    plt.rcParams['axes.linewidth'] = 0.4
    plt.figure(figsize=(6.8, 4.1))
    plt.bar([5+key for key in buckets.keys()], buckets.values(), width=10, edgecolor='black', linewidth=0.75, color='#75a9d3')
    plt.xlabel('Surgery duration (minute)')
    plt.ylabel('# of surgeries')
    plt.axvline(q1, color='blue', linestyle='-.', linewidth=1)
    plt.axvline(q3, color='blue', linestyle='-.', linewidth=1)
    plt.xticks([i*10 for i in range(12)])
    plt.yticks([i*5 for i in range(9)])
    plt.tick_params(axis='y', direction='in', length=4, left=True, right=True, width=0.4)
    plt.tick_params(axis='x', direction='in', length=4, bottom=True, top=True, width=0.4)
    # title = title + f'\nQ1={round(q1)}, mean={round(mean)}, Q3={round(q3)}'
    plt.tight_layout()
    plt.savefig('./plots/cholec80_durations.pdf')
    plt.rcParams['axes.linewidth'] = 0.8

def dataset_visualisations():
    transform = transforms.Resize((240, 320))
    # dataset = ImageDataset(os.path.join(DATA_ROOT, "breakfast"), "rgb-images", f"small.txt", sample_transform=Middle())
    dataset = UCFDataset(os.path.join(DATA_ROOT, "ucf24"), "rgb-images", f"small.txt", sample_transform=Middle())
    num_activities = len(dataset) // 10

    frames = []
    fig, axs = plt.subplots(num_activities, 10, figsize=(6.4, 4.8*(24/10)))
    unique_names = []
    for name, frame, _ in dataset:
        frames.append(transform(frame[0]))
        activity_name = name.split('/')[0]
        if activity_name not in unique_names:
            unique_names.append(activity_name)
    for (frame, ax) in zip(frames, axs.flat):
        ax.imshow(frame)
        # ax.axis('off')
        ax.set_xticks([])
        ax.set_xticks([], minor=True)
        ax.set_yticks([])
        ax.set_yticks([], minor=True)
    # for (unique_name, ax) in zip(unique_names, axs[:, 0]):
    #     ax.axis('off')
    #     ax.text(0, 0, unique_name)
        # ax.set_aspect('equal')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig('./plots/ucf_visualisations.png')
    # ucf101_small = UCFDataset(os.path.join(DATA_ROOT, "ucf24"), "rgb-images", f"small.txt")

def cholec_visualisations():
    # transform = transforms.Resize((240, 320))
    transform = transforms.Compose([])
    dataset = ImageDataset(os.path.join(DATA_ROOT, "cholec80"), "rgb-images", f"v_0.txt", sample_transform=Middle())

    frames = []
    fig, axs = plt.subplots(1, 6, figsize=(6.4, 4.8 / 5))
    for name, frame, _ in dataset:
        frames.append(transform(frame[0]))
    for (frame, ax) in zip(frames, axs.flat):
        ax.imshow(frame)
        # ax.axis('off')
        ax.set_xticks([])
        ax.set_xticks([], minor=True)
        ax.set_yticks([])
        ax.set_yticks([], minor=True)
    # for (unique_name, ax) in zip(unique_names, axs[:, 0]):
    #     ax.axis('off')
    #     ax.text(0, 0, unique_name)
        # ax.set_aspect('equal')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig('./plots/c80_visualisations.pdf')

def visualise_results():
    for index, timestamps in zip(['04', '05'], [(210, 1250), (200, 1650)]):
        visualise_video(
            os.path.join(DATA_ROOT, f'cholec80/rgb-images/video{index}'), timestamps,
            [('ResNet-2D', f'./data/cholec/resnet_cholec/video{index}.txt', ':'),
            ('ResNet-LSTM', f'./data/cholec/lstm_cholec/video{index}.txt', ':'),

            ('UTE', f'./data/cholec/ute_cholec/video{index}.txt', '-.'),
            ('RSDNet', f'./data/cholec/rsd_cholec/video{index}.txt', '-.'),
            ('ProgressNet', f'./data/cholec/pn_cholec/video{index}.txt', '-.'),
            ('average-index', f'./data/cholec_baseline.txt', '-')],
            f'cholec80_video{index}', 200
    )
    for index, timestamp in zip(['00015'], [(10, 55)]):
        visualise_video(
            os.path.join(DATA_ROOT, f'bars/rgb-images/{index}'), timestamp,
            [('ResNet', f'./data/bars/resnet_bars/{index}.txt', ':'),
            ('ResNet-LSTM', f'./data/bars/lstm_bars/{index}.txt', ':'),

            ('UTE', f'./data/bars/ute_bars/{index}.txt', '-.'),
            ('RSDNet', f'./data/bars/rsd_bars/{index}.txt', '-.'),
            ('ProgressNet', f'./data/bars/pn_bars/{index}.txt', '-.')],
            f'bars_video{index}', 1
    )
    # # 0, 122, 0, 0, 0
    # for index, timestamp, offset in zip(['Biking/v_Biking_g01_c02', 'Fencing/v_Fencing_g01_c01', 'FloorGymnastics/v_FloorGymnastics_g01_c03', 'GolfSwing/v_GolfSwing_g01_c03', 'GolfSwing/v_GolfSwing_g01_c02', 'HorseRiding/v_HorseRiding_g01_c01'], [80, 45, 60, 125, 45, 125], [0, 0, 0, 0, 0, 0]):
    #     visualise_video(
    #         os.path.join(DATA_ROOT, f'ucf24/rgb-images/{index}'), timestamp,
    #         [('ProgressNet (full-video)', f'./data/ucf/pn_ucf/{index.replace("/", "_")}_0.txt', '-.'),
    #          ('ProgressNet (full-video)', f'./data/ucf/pn_ucf_segments/{index.replace("/", "_")}_0.txt', '-.'),
    #          ('average-index', f'./data/ucf_baseline.txt', '-')],
    #          f'ucf_video_{index.replace("/", "_")}', 1, offset=offset
    #     )
    # for index, timestamp in zip(['pancake/P12/cam01', 'sandwich/P07/webcam01'], [150, 65]):
    #     visualise_video(
    #         os.path.join(DATA_ROOT, f'breakfast/rgb-images/{index}'), timestamp,
    #         [('ResNet-LSTM', f'./data/breakfast/lstm_bf_random/{index.replace("/", "_")}.txt', '-.'),
    #          ('RSDNet', f'./data/breakfast/rsd_bf_random/{index.replace("/", "_")}.txt', '-.'),
    #          ('ProgressNet', f'./data/breakfast/pn_bf_random/{index.replace("/", "_")}.txt', '-.'),
    #          ('average-index', f'./data/bf_baseline_sampled.txt', '-')],
    #          f'bf_video_{index.replace("/", "_")}', 1, subsample=15,
    #     )

parser = argparse.ArgumentParser()
parser.add_argument('--pdf', action='store_true')
args = parser.parse_args()
FILE = 'pdf' if args.pdf else 'png'
def main():
    try:
        os.mkdir('./plots')
        os.mkdir('./plots/bars')
        os.mkdir('./plots/results')
        os.mkdir('./plots/examples')
    except:
        pass
    # set_font_sizes(16, 18, 20)
    # plot_baseline_example()
    # return
    dataset_visualisations()
    cholec_visualisations()
    set_font_sizes(16, 18, 20)
    visualise_results()
    set_font_sizes()
    # result plots
    results = load_results('./results.json')
    for dataset in ["UCF101-24", "Cholec80", "Breakfast"]:
        plot_result_bar(results, dataset, ["'full-video' inputs", "'random-noise' inputs"], ['full video', 'random'])
        plot_result_bar(results, dataset, ["'video-segments' inputs", "'frame-indices' inputs"], ['video segments', 'indices'])
    plot_result_bar(results, "Bars", ["'full-video' inputs", "'video-segments' inputs"], ['full video', 'video segments'])
    # average index baseline
    set_font_sizes(16, 18, 20)
    # plot_baselines()
    # plot_baseline_example()
    set_font_sizes()
    # dataset statistics
    plot_dataset_lengths()
    # syntethic dataset example
    plot_synthetic(4, [0, 15, 35, 58, 71])
    # example progress predictions
    set_font_sizes(16, 18, 20)
    visualise_results()

    # dataset_statistics()
    # plot_dataset_lengths()
    set_font_sizes(9, 10, 12)
    surgery_duration()

if __name__ == '__main__':
    main()
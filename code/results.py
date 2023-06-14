from typing import List
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-v0_8-paper")

results = {
    "UCF101-24": {
        "ResNet152": {
            "normal": 25.8,
            "segments": 25.8,
            "indices": 14.1,
        },
        "ResNet152-LSTM": {
            "normal": 24.4,
            "random": 17.1,
            "segments": 24.5,
            "indices": 18.0,
        },
        "UTE": {
            "normal": 24.7,
            "segments": 24.7,
            "indices": 14.1,
        },
        "ProgressNet": {
            "normal": 13.6,
            "random": 17.4,
            "segments": 26.6,
            "indices": 19.2,
        },
        "RSDNet": {
            "normal": 24.5,
            "random": 17.3,
            "segments": 25.1,
            "indices": 18.3,
        },
        "Static 0.5": {
            "normal": 25.0,
            "random": 25.0,
            "segments": 25.0,
            "indices": 25.0,
        },
        "Random": {
            "normal": 33.3,
            "random": 33.3,
            "segments": 33.3,
            "indices": 33.3,
        },
        "Average Index": {
            "normal": 14.2,
            "random": 14.2,
            "segments": 14.2,
            "indices": 14.2,
        },
    },
    "Cholec80 (sampled)": {
        "ResNet152": {
            "normal": 17.9,
            "segments": 17.9,
            "indices": 12.3,
        },
        "ResNet152-LSTM": {  # TODO: Waiting
            "normal": 10.2,
            "random": 11.9,
            "segments": 14.4,
            "indices": 12.6,
        },
        "UTE": {  # TODO: Waiting
            "normal": 17.1,
            "segments": 17.1,
            "indices": 12.1,
        },
        "ProgressNet": {  # TODO: Run 2
            "normal": 12.0,
            "random": 13.5,
            "segments": 0,
            "indices": 0,
        },
        "RSDNet": {
            "normal": 10.9,
            "random": 12.7,
            "segments": 17.6,
            "indices": 19.3,
        },
        "Static 0.5": {
            "normal": 25.0,
            "random": 25.0,
            "segments": 25.0,
            "indices": 25.0,
        },
        "Random": {
            "normal": 33.3,
            "random": 33.3,
            "segments": 33.3,
            "indices": 33.3,
        },
        "Average Index": {
            "normal": 11.9,
            "random": 11.9,
            "segments": 11.9,
            "indices": 11.9,
        },
    },
    # vvvvvvvv
    "Cholec80": {
        "ResNet152": {
            "normal": 17.9,
            "indices": 12.3,
        },
        "ResNet152-LSTM": {"normal": 0, "random": 0, "segments": 0, "indices": 0},
        "UTE": {
            "normal": 0,
            "indices": 0,
        },
        "RSDNet": {
            "normal": 0,
            "random": 0,
            "segments": 0,
            "indices": 0,
        },
        "Static 0.5": {
            "normal": 25.0,
            "random": 25.0,
            "segments": 25.0,
            "indices": 25.0,
        },
        "Random": {
            "normal": 33.3,
            "random": 33.3,
            "segments": 33.3,
            "indices": 33.3,
        },
        "Average Index": {
            "normal": 11.9,
            "random": 11.9,
            "segments": 11.9,
            "indices": 11.9,
        },
    },
}
colours = {
    "ResNet152": "red",
    "ResNet152-LSTM": "red",
    "UTE": "red",
    "ProgressNet": "red",
    "RSDNet": "red",
    "Static 0.5": "red",
    "Random": "red",
    "Average Index": "red",
}

BAR_WIDTH = 0.5
SPACING = 1.5
COLORS = ["r", "g", "b"]

BAR_XS = np.array([0, 1, 3, 4, 5, 7, 8, 9])


def compare(dataset: str, modes: List[str]):
    data = [[] for mode in modes]
    networks = list(results[dataset].keys())
    for network in results[dataset]:
        for i, mode in enumerate(modes):
            if mode in results[dataset][network]:
                data[i].append(results[dataset][network][mode])
            else:
                data[i].append(0)

    length = len(data[0])
    for i, (values, mode) in enumerate(zip(data, modes)):
        bar_xs = BAR_XS * SPACING + i * BAR_WIDTH
        plt.bar(bar_xs, values, width=BAR_WIDTH, label=mode)
    xticks = BAR_XS * SPACING + BAR_WIDTH * 0.5

    plt.axhline(y=results[dataset]["Average Index"]["normal"], linestyle=':', color='grey')
    plt.xticks(xticks, networks, rotation=90)
    plt.ylabel("L1 Loss")
    plt.title(f'{dataset} - {" vs ".join(modes)}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./plots/{dataset}_{'_'.join(modes)}.png")
    plt.clf()


def main():
    for dataset in ['UCF101-24', 'Cholec80 (sampled)']:
        compare(dataset, ["normal", "random"])
        compare(dataset, ["segments", "indices"])


if __name__ == "__main__":
    main()

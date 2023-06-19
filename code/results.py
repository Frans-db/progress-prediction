from typing import List
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use("seaborn-v0_8-paper")

results = {
    "UCF101-24": {
        "ResNet152": {
            "sequences": 25.8,
            "segments": 25.8,
            "indices": 14.1,
        },
        "ResNet152-LSTM": {
            "sequences": 24.4,
            "random": 17.1,
            "segments": 24.5,
            "indices": 18.0,
        },
        "UTE": {
            "sequences": 24.7,
            "segments": 24.7,
            "indices": 14.1,
        },
        "ProgressNet": {
            "sequences": 13.6,
            "random": 17.4,
            "segments": 26.6,
            "indices": 19.2,
        },
        "RSDNet": {
            "sequences": 24.5,
            "random": 17.3,
            "segments": 25.1,
            "indices": 18.3,
        },
        "Static 0.5": {
            "sequences": 25.0,
            "random": 25.0,
            "segments": 25.0,
            "indices": 25.0,
        },
        "Random": {
            "sequences": 33.3,
            "random": 33.3,
            "segments": 33.3,
            "indices": 33.3,
        },
        "Average Index": {
            "sequences": 14.2,
            "random": 14.2,
            "segments": 14.2,
            "indices": 14.2,
        },
    },
    # "Cholec80 (sampled)": {
    #     "ResNet152": {
    #         "sequences": 17.9,
    #         "segments": 17.9,
    #         "indices": 12.3,
    #     },
    #     "ResNet152-LSTM": {
    #         "sequences": 10.2,
    #         "random": 11.9,
    #         "segments": 14.4,
    #         "indices": 12.6,
    #     },
    #     "UTE": {
    #         "sequences": 16.0,
    #         "segments": 16.0,
    #         "indices": 11.7,
    #     },
    #     "ProgressNet": {
    #         "sequences": 12.0,
    #         "random": 13.5,
    #         "segments": 14.0,
    #         "indices": 13.5,
    #     },
    #     "RSDNet": {
    #         "sequences": 10.9,
    #         "random": 12.7,
    #         "segments": 17.6,
    #         "indices": 19.3,
    #     },
    #     "Static 0.5": {
    #         "sequences": 25.0,
    #         "random": 25.0,
    #         "segments": 25.0,
    #         "indices": 25.0,
    #     },
    #     "Random": {
    #         "sequences": 33.3,
    #         "random": 33.3,
    #         "segments": 33.3,
    #         "indices": 33.3,
    #     },
    #     "Average Index": {
    #         "sequences": 11.9,
    #         "random": 11.9,
    #         "segments": 11.9,
    #         "indices": 11.9,
    #     },
    # },
    "Cholec80": {
        "ResNet152": {
            "sequences": 17.9,
            "segments": 17.9,
            "indices": 12.3,
        },
        "ResNet152-LSTM": {
            "sequences": 9.8,
            "random": 12.2,
            "segments": 12.4,
            "indices": 12.5,
        },
        "UTE": {
            "sequences": 17.2,
            "segments": 17.2,
            "indices": 11.7,
        },
        "ProgressNet": {
            "sequences": 21.9,
            "random": 19.2,
            "segments": 26.8,
            "indices": 19.1,
        },
        "RSDNet": {
            "sequences": 11.0,
            "random": 14.1,
            "segments": 14.6,
            "indices": 19.4,
        },
        "Static 0.5": {
            "sequences": 25.0,
            "random": 25.0,
            "segments": 25.0,
            "indices": 25.0,
        },
        "Random": {
            "sequences": 33.3,
            "random": 33.3,
            "segments": 33.3,
            "indices": 33.3,
        },
        "Average Index": {
            "sequences": 11.9,
            "random": 11.9,
            "segments": 11.9,
            "indices": 11.9,
        },
    },
    "Breakfast": {
        "ResNet152": {
            "sequences": 27.1,
            "segments": 27.1,
            "indices": 16.7,
        },
        "ResNet152-LSTM": {
            "sequences": 31.1,
            "random": 19.0,
            "segments": 26.4,
            "indices": 20.0,
        },
        "UTE": {
            "sequences": 25.3,
            "segments": 25.3,
            "indices": 16.4,
        },
        "ProgressNet": {
            "sequences": 21.9,
            "random": 19.2,
            "segments": 26.8,
            "indices": 19.1,
        },
        "RSDNet": {
            "sequences": 30.1,
            "random": 19.2,
            "segments": 27.2,
            "indices": 24.2,
        },
        "Static 0.5": {
            "sequences": 25.0,
            "random": 25.0,
            "segments": 25.0,
            "indices": 25.0,
        },
        "Random": {
            "sequences": 33.3,
            "random": 33.3,
            "segments": 33.3,
            "indices": 33.3,
        },
        "Average Index": {
            "sequences": 16.7,
            "random": 16.7,
            "segments": 16.7,
            "indices": 16.7,
        },
    },
    "Bars": {
        "ResNet18": {
            "sequences": 2.8,
            "segments": 2.8,
        },
        "ResNet18-LSTM": {
            "sequences": 2.6,
            "segments": 3.3,
        },
        "UTE": {"sequences": 1.3, "segments": 1.3},
        "ProgressNet": {"sequences": 2.6, "segments": 2.8},
        "RSDNet": {
            "sequences": 2.7,
            "segments": 3.3,
        },
        "Static 0.5": {"sequences": 25.0, "segments": 25.0},
        "Random": {"sequences": 33.5, "segments": 33.5},
        "Average Index": {"sequences": 12.9, "segments": 12.9},
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

# [, , , , , , , (0.5490196078431373, 0.5490196078431373, 0.5490196078431373), (0.8, 0.7254901960784313, 0.4549019607843137), (0.39215686274509803, 0.7098039215686275, 0.803921568627451)]

BAR_WIDTH = 0.5
SPACING = 1.5
MODE_COLOURS = {
    "sequences": (0.2980392156862745, 0.4470588235294118, 0.6901960784313725),
    "random": (0.8666666666666667, 0.5176470588235295, 0.3215686274509804),
    "segments": (0.3333333333333333, 0.6588235294117647, 0.40784313725490196),
    "indices": (0.7686274509803922, 0.3058823529411765, 0.3215686274509804),
}
BASELINES = ["Average Index", "Static 0.5", "Random"]
LINEWIDTH = 1


def compare(dataset: str, modes: List[str]):
    plt.figure(figsize=(7.2, 4.2))
    data = [[] for mode in modes]
    networks = [key for key in results[dataset] if key not in BASELINES][::-1]
    xs_indices = np.array([0, 1, 2, 4, 5])
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
        plt.barh(bar_xs, values, height=BAR_WIDTH, label=mode, color=MODE_COLOURS[mode])
    xticks = xs_indices * SPACING + BAR_WIDTH * 0.5

    plt.axvline(
        x=results[dataset]["Average Index"]["sequences"],
        linestyle="-",
        label="Average Index",
        color=(0.8, 0.7254901960784313, 0.4549019607843137),
        linewidth=LINEWIDTH,
    )
    plt.axvline(
        x=results[dataset]["Static 0.5"]["sequences"],
        linestyle="-",
        label="Static 0.5",
        color=(0.8549019607843137, 0.5450980392156862, 0.7647058823529411),
        linewidth=LINEWIDTH,
    )
    plt.axvline(
        x=results[dataset]["Random"]["sequences"],
        linestyle="-",
        label="Random",
        color=(0.5058823529411764, 0.4470588235294118, 0.7019607843137254),
        linewidth=LINEWIDTH,
    )
    plt.text(
        results[dataset]["Average Index"]["sequences"] - 1,
        4.3,
        str(results[dataset]["Average Index"]["sequences"]),
    ).set_bbox({"facecolor": "white", "edgecolor": "white"})
    plt.text(
        results[dataset]["Static 0.5"]["sequences"] - 1,
        4.3,
        str(results[dataset]["Static 0.5"]["sequences"]),
    ).set_bbox({"facecolor": "white", "edgecolor": "white"})
    plt.text(
        results[dataset]["Random"]["sequences"] - 1,
        4.3,
        str(results[dataset]["Random"]["sequences"]),
    ).set_bbox({"facecolor": "white", "edgecolor": "white"})
    plt.yticks(xticks, networks)
    plt.xlabel("L1 Loss")
    plt.title(f'{dataset} - {" vs ".join(modes)}')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f"./plots/results/{dataset}_{'_'.join(modes)}.pdf")
    plt.savefig(f"./plots/results/{dataset}_{'_'.join(modes)}.png")
    plt.clf()


def main():
    for dataset in ["UCF101-24", "Cholec80", "Breakfast"]:
        compare(dataset, ["sequences", "random"])
        compare(dataset, ["segments", "indices"])
    compare("Bars", ["sequences", "segments"])


if __name__ == "__main__":
    main()

from typing import List
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use("seaborn-v0_8-paper")

results = {
    "UCF101-24": {
        "ResNet 2D": {
            "full video": 25.8,
            "video segments": 25.8,
            "indices": 14.1,
        },
        "ResNet\n-LSTM": {
            "full video": 24.4,
            "random": 17.1,
            "video segments": 24.5,
            "indices": 18.0,
        },
        "UTE": {
            "full video": 24.7,
            "video segments": 24.7,
            "indices": 14.1,
        },
        "ProgressNet": {
            "full video": 13.6,
            "random": 17.4,
            "video segments": 26.6,
            "indices": 19.2,
        },
        "RSDNet": {
            "full video": 24.5,
            "random": 17.3,
            "video segments": 25.1,
            "indices": 18.3,
        },
        "Static 0.5": {
            "full video": 25.0,
            "random": 25.0,
            "video segments": 25.0,
            "indices": 25.0,
        },
        "Random": {
            "full video": 33.3,
            "random": 33.3,
            "video segments": 33.3,
            "indices": 33.3,
        },
        "Average Index": {
            "full video": 14.2,
            "random": 14.2,
            "video segments": 14.2,
            "indices": 14.2,
        },
    },
    # "Cholec80 (sampled)": {
    #     "ResNet152": {
    #         "full video": 17.9,
    #         "video segments": 17.9,
    #         "indices": 12.3,
    #     },
    #     "ResNet152\n-LSTM": {
    #         "full video": 10.2,
    #         "random": 11.9,
    #         "video segments": 14.4,
    #         "indices": 12.6,
    #     },
    #     "UTE": {
    #         "full video": 16.0,
    #         "video segments": 16.0,
    #         "indices": 11.7,
    #     },
    #     "ProgressNet": {
    #         "full video": 12.0,
    #         "random": 13.5,
    #         "video segments": 14.0,
    #         "indices": 13.5,
    #     },
    #     "RSDNet": {
    #         "full video": 10.9,
    #         "random": 12.7,
    #         "video segments": 17.6,
    #         "indices": 19.3,
    #     },
    #     "Static 0.5": {
    #         "full video": 25.0,
    #         "random": 25.0,
    #         "video segments": 25.0,
    #         "indices": 25.0,
    #     },
    #     "Random": {
    #         "full video": 33.3,
    #         "random": 33.3,
    #         "video segments": 33.3,
    #         "indices": 33.3,
    #     },
    #     "Average Index": {
    #         "full video": 11.9,
    #         "random": 11.9,
    #         "video segments": 11.9,
    #         "indices": 11.9,
    #     },
    # },
    "Cholec80": {
        "ResNet 2D": {
            "full video": 17.9,
            "video segments": 17.9,
            "indices": 12.3,
        },
        "ResNet\n-LSTM": {
            "full video": 9.8,
            "random": 12.2,
            "video segments": 12.4,
            "indices": 12.5,
        },
        "UTE": {
            "full video": 17.2,
            "video segments": 17.2,
            "indices": 11.7,
        },
        "ProgressNet": {
            "full video": 21.9,
            "random": 19.2,
            "video segments": 26.8,
            "indices": 19.1,
        },
        "RSDNet": {
            "full video": 11.0,
            "random": 14.1,
            "video segments": 14.6,
            "indices": 19.4,
        },
        "Static 0.5": {
            "full video": 25.0,
            "random": 25.0,
            "video segments": 25.0,
            "indices": 25.0,
        },
        "Random": {
            "full video": 33.3,
            "random": 33.3,
            "video segments": 33.3,
            "indices": 33.3,
        },
        "Average Index": {
            "full video": 11.9,
            "random": 11.9,
            "video segments": 11.9,
            "indices": 11.9,
        },
    },
    "Breakfast": {
        "ResNet 2D": {
            "full video": 27.1,
            "video segments": 27.1,
            "indices": 16.7,
        },
        "ResNet\n-LSTM": {
            "full video": 31.1,
            "random": 19.0,
            "video segments": 26.4,
            "indices": 20.0,
        },
        "UTE": {
            "full video": 25.3,
            "video segments": 25.3,
            "indices": 16.4,
        },
        "ProgressNet": {
            "full video": 21.9,
            "random": 19.2,
            "video segments": 26.8,
            "indices": 19.1,
        },
        "RSDNet": {
            "full video": 30.1,
            "random": 19.2,
            "video segments": 27.2,
            "indices": 24.2,
        },
        "Static 0.5": {
            "full video": 25.0,
            "random": 25.0,
            "video segments": 25.0,
            "indices": 25.0,
        },
        "Random": {
            "full video": 33.3,
            "random": 33.3,
            "video segments": 33.3,
            "indices": 33.3,
        },
        "Average Index": {
            "full video": 16.7,
            "random": 16.7,
            "video segments": 16.7,
            "indices": 16.7,
        },
    },
    "Bars": {
        "ResNet 2D": {
            "full video": 2.8,
            "video segments": 2.8,
        },
        "ResNet\n-LSTM": {
            "full video": 2.6,
            "video segments": 3.3,
        },
        "UTE": {"full video": 1.3, "video segments": 1.3},
        "ProgressNet": {"full video": 2.6, "video segments": 2.8},
        "RSDNet": {
            "full video": 2.7,
            "video segments": 3.3,
        },
        "Static 0.5": {"full video": 25.0, "video segments": 25.0},
        "Random": {"full video": 33.5, "video segments": 33.5},
        "Average Index": {"full video": 12.9, "video segments": 12.9},
    },
}
colours = {
    "ResNet 2D": "red",
    "ResNet\n-LSTM": "red",
    "UTE": "red",
    "ProgressNet": "red",
    "RSDNet": "red",
    "Static 0.5": "red",
    "Random": "red",
    "Average Index": "red",
}

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

plt.rcParams['axes.spines.left'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.bottom'] = False
plt.rcParams['axes.axisbelow'] = True

def compare(dataset: str, modes: List[str]):  
    plt.figure(figsize=(7.2, 4.2))

    data = [[] for mode in modes]
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
    plt.savefig(f"./plots/results/{dataset}_{'_'.join(modes).replace(' ', '_')}.pdf")
    # plt.savefig(f"./plots/results/{dataset}_{'_'.join(modes)}.png")
    plt.clf()


def main():
    for dataset in ["UCF101-24", "Cholec80", "Breakfast"]:
        compare(dataset, ["full video", "random"])
        compare(dataset, ["video segments", "indices"])
    compare("Bars", ["full video", "video segments"])


if __name__ == "__main__":
    main()

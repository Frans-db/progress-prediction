import os
import random
import math
import torch
import torch.nn as nn
from typing import List, Tuple
from copy import copy
import statistics

root = "/home/frans/Datasets/cholec80/"


def save_fold(data: List[Tuple[str, int]], fold_name: str) -> None:
    names = [video[0] for video in data]
    t1 = names[:27]
    t2 = names[27 : 27 + 27]
    t12 = names[: 27 + 27]
    v = names[27 + 27 : 27 + 27 + 6]
    e = names[27 + 27 + 6 : 27 + 27 + 6 + 20]
    with open(os.path.join(root, "splitfiles", f"t1_{fold_name}.txt"), "w+") as f:
        f.write("\n".join(t1))
    with open(os.path.join(root, "splitfiles", f"t2_{fold_name}.txt"), "w+") as f:
        f.write("\n".join(t2))
    with open(os.path.join(root, "splitfiles", f"t12_{fold_name}.txt"), "w+") as f:
        f.write("\n".join(t12))
    with open(os.path.join(root, "splitfiles", f"v_{fold_name}.txt"), "w+") as f:
        f.write("\n".join(v))
    with open(os.path.join(root, "splitfiles", f"e_{fold_name}.txt"), "w+") as f:
        f.write("\n".join(e))


def random_folds(data: List[Tuple[str, int]], start_index: int) -> None:
    for i in range(4):
        random.shuffle(data)
        save_fold(data, str(start_index + i))


def proper_fold(data: List[Tuple[str, int]], start_index: int) -> None:
    random.shuffle(data)
    for i in range(4):
        save_fold(data, f"p{start_index+0}")
        data = data[20:] + data[:20]


def main() -> None:
    random.seed(42)

    video_names = sorted(os.listdir(os.path.join(root, "rgb-images")))
    video_lengths = []
    for video_name in video_names:
        video_path = os.path.join(os.path.join(root, "rgb-images", video_name))
        num_frames = len(os.listdir(video_path))
        video_lengths.append(num_frames)
    video_data = list(zip(video_names, video_lengths))

    print("--- random 1 ---")
    random_folds(video_data, 0)
    print("--- random 2 ---")
    random_folds(video_data, 4)
    print("--- random 3 ---")
    random_folds(video_data, 8)
    print("--- proper 1 ---")
    proper_fold(video_data, 0)
    print("--- proper 2 ---")
    proper_fold(video_data, 4)


if __name__ == "__main__":
    main()

import os
import random
import math
import torch
import torch.nn as nn
from torchvision import models
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
    with open(os.path.join(root, "splitfiles", f"all_{fold_name}.txt"), "w+") as f:
        f.write("\n".join(sorted(names)))
    with open(os.path.join(root, "splitfiles", f"t1_{fold_name}.txt"), "w+") as f:
        f.write("\n".join(sorted(t1)))
    with open(os.path.join(root, "splitfiles", f"t2_{fold_name}.txt"), "w+") as f:
        f.write("\n".join(sorted(t2)))
    with open(os.path.join(root, "splitfiles", f"t12_{fold_name}.txt"), "w+") as f:
        f.write("\n".join(sorted(t12)))
    with open(os.path.join(root, "splitfiles", f"v_{fold_name}.txt"), "w+") as f:
        f.write("\n".join(sorted(v)))
    with open(os.path.join(root, "splitfiles", f"e_{fold_name}.txt"), "w+") as f:
        f.write("\n".join(sorted(e)))


def random_folds(data: List[Tuple[str, int]], start_index: int) -> None:
    for i in range(4):
        random.shuffle(data)
        save_fold(data, str(start_index + i))


def proper_fold(data: List[Tuple[str, int]], start_index: int) -> None:
    random.shuffle(data)
    for i in range(4):
        save_fold(data, f"{start_index+i}")
        data = data[20:] + data[:20]

def create_cholec80_folds():
    video_names = sorted(os.listdir(os.path.join(root, "rgb-images")))
    video_lengths = []
    for video_name in video_names:
        video_path = os.path.join(os.path.join(root, "rgb-images", video_name))
        num_frames = len(os.listdir(video_path))
        video_lengths.append(num_frames)
    video_data = list(zip(video_names, video_lengths))

    print("--- proper 2 ---")
    proper_fold(video_data, 0)

def create_networks():
    resnet = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
    resnet.fc = nn.Identity()
    torch.save(resnet.state_dict(), './resnet152.pth')

    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    resnet.fc = nn.Identity()
    torch.save(resnet.state_dict(), './resnet18.pth')

    vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
    torch.save(vgg16.state_dict(), './vgg16.pth')

    vgg11 = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1).features
    torch.save(vgg11.state_dict(), './vgg11.pth')

def main() -> None:
    random.seed(42)
    # create_cholec80_folds()
    create_networks()




if __name__ == "__main__":
    main()

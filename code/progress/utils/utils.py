import argparse
import uuid
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

"""
TODO:
utils is not a very clear name, extract all functionality into properly named modules
"""

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)

    parser.add_argument('--model', type=str, default='conv3d')

    parser.add_argument('--dataset', type=str, default='toy')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=1)

    parser.add_argument('--num_segments', type=int, default=1)
    parser.add_argument('--frames_per_segment', type=int, default=10)
    parser.add_argument('--sample_every', type=int, default=1)

    args = parser.parse_args()
    if args.name is None:
        args.name = uuid.uuid4()
    if args.seed is None:
        args.seed = random.randint(0, 1_000_000_000)
    return args


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
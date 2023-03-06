import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from networks import RSDNet
from datasets import RSDDataset, rsd_collate

"""
implementation of https://arxiv.org/abs/1802.03243
"""

"""
Implementation is done in 3 steps:
1. Resnet is trained to output progress
    - Replace last layer with a 1 node linear layer and a sigmoid activation
2. Resnet w/ LSTM
    - Last layer of resnet is removed
"""

def main():
    dataset = RSDDataset('/home/frans/Datasets/toy', 'rgb-images', 'splitfiles/trainlist01.txt')

    print(dataset.get_max_video_frame_length())
    print(dataset.get_max_video_length())

if __name__ == '__main__':
    main()
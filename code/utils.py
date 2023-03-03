import os
import torch
import random
import numpy as np
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    # experiment
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    # directories
    parser.add_argument('--experiment_directory', type=str, default='experiments')
    parser.add_argument('--model_directory', type=str, default='models')
    parser.add_argument('--log_directory', type=str, default='logs')
    parser.add_argument('--figures_directory', type=str, default='figures')
    # figures
    parser.add_argument('--figures', action='store_true')
    parser.add_argument('--figure_every', type=int, default=1)
    # dataset
    parser.add_argument('--data_root', type=str, default='/home/frans/Datasets/')
    parser.add_argument('--dataset', type=str, default='ucf24')
    parser.add_argument('--data_type', type=str, default='rgb-images')
    parser.add_argument('--splitfile_dir', type=str, default='splitfiles')
    parser.add_argument('--annotation_file', type=str, default='pyannot.pkl')
    parser.add_argument('--train_split_file', type=str, default='trainlist01.txt')
    parser.add_argument('--test_split_file', type=str, default='testlist01.txt')
    # progressnet
    parser.add_argument('--embed_size', type=int, default=2048)
    # model loading
    parser.add_argument('--model_name', type=str, default='010.pth')

    return parser.parse_args()

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def create_directory(directory: str):
    if not os.path.isdir(directory):
        os.mkdir(directory)
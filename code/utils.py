import os
import torch
import random
import numpy as np
from os.path import join
from args import parse_arguments
import logging

def setup_directores(args):
    # create paths & directories
    annotation_path = join(args.splitfile_dir, args.annotation_file)
    train_splitfile_path = join(args.splitfile_dir, args.train_split_file)
    test_splitfile_path = join(args.splitfile_dir, args.test_split_file)
    dataset_directory = join(args.data_root, args.dataset)
    experiment_directory = join(dataset_directory, args.experiment_directory, args.experiment_name)
    model_directory = join(experiment_directory, args.model_directory)
    figures_directory = join(experiment_directory, args.figures_directory)
    log_directory = join(experiment_directory, args.log_directory)
    log_path = join(log_directory, 'eval.log' if args.eval else 'train.log')
    # create directories
    create_directory(experiment_directory)
    create_directory(log_directory)
    create_directory(model_directory)
    create_directory(figures_directory)

    return {
        'dataset_directory': dataset_directory,
        'model_directory': model_directory,
        'figures_directory': figures_directory,
        'log_path': log_path,
        'annotation_path': annotation_path,
        'train_splitfile_path': train_splitfile_path,
        'test_splitfile_path': test_splitfile_path
    }

def setup():
    args = parse_arguments()
    set_seeds(args.seed)
    device = get_device()
    dirs = setup_directores(args)
    # setup logging
    logging.basicConfig(
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(dirs['log_path']),
            logging.StreamHandler()
        ]
    )

    return args, dirs, device

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def create_directory(directory: str):
    if not os.path.isdir(directory):
        os.mkdir(directory)
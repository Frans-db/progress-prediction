import os

from .progress_dataset import ProgressDataset

def get_datasets(args):
    train_root = os.path.join(args.data_root, args.train_set)
    test_root = os.path.join(args.data_root, args.test_set)
    return ProgressDataset(train_root, args.data_type, args.train_split), ProgressDataset(test_root, args.data_type, args.test_split)
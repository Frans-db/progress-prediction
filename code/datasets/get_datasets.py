import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch

from .progress_dataset import ProgressFeatureDataset, ProgressVideoDataset, ProgressCategoryDataset
from .boundingbox_dataset import BoundingBoxDataset
from .augmentations import Indices, Ones, Randoms
from .augmentations import Subsection, Subsample


def get_datasets(args):
    # TODO: augmentations & data modifier
    transform = get_transform(args)
    data_transform = get_data_transform(args)
    sample_transform = get_sample_transform(args)
    train_transform = {
        'transform': transform,
        'data_transform': data_transform,
        'sample_transform': sample_transform
    }
    test_transform = {
        'transform': transform,
        'data_transform': data_transform,
    }
    train_set = get_dataset(args, args.train_split, transform=train_transform)
    test_set = get_dataset(args, args.test_split, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=1,
                              num_workers=2, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1,
                             num_workers=2, shuffle=False)

    return train_set, test_set, train_loader, test_loader


def get_dataset(args, split_file: str, transform=None):
    root = os.path.join(args.data_root, args.dataset)

    if args.bounding_boxes:
        return BoundingBoxDataset(root, args.data_type, split_file, 'pyannot.pkl', transform=transform)
    elif 'rgb-images' in args.data_type:
        return ProgressVideoDataset(root, args.data_type, split_file, transform=transform)
    elif args.category_directory is not None:
        return ProgressCategoryDataset(root, args.data_type, args.category_directory, args.num_categories, split_file, transform=transform)
    else:
        return ProgressFeatureDataset(root, args.data_type, split_file, transform=transform)


def get_transform(args):
    transform = []
    if args.data_type == 'rgb-images':
        transform.append(transforms.ToTensor())
        if args.resize and args.resize > 0:
            if args.antialias:
                transform.append(transforms.Resize((args.resize, args.resize), antialias=True))
            else:
                transform.append(transforms.Resize((args.resize, args.resize)))
    return transforms.Compose(transform)


def get_data_transform(args):
    transform = []
    if args.data_modifier == 'indices':
        transform.append(Indices(args.data_modifier_value))
    elif args.data_modifier == 'ones':
        transform.append(Ones())
    elif args.data_modifier == 'randoms':
        transform.append(Randoms())

    return transforms.Compose(transform)


def get_sample_transform(args):
    return transforms.Compose([
        Subsection(p=args.subsection_chance),
        Subsample(p=args.subsample_chance)
    ])

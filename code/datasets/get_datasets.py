import os
import torchvision.transforms as transforms

from .progress_dataset import ProgressFeatureDataset, ProgressVideoDataset
from .boundingbox_dataset import BoundingBoxDataset
from .augmentations import Indices, Ones, Randoms
from .augmentations import Subsection, Subsample

def get_datasets(args):
    # TODO: augmentations & data modifier
    transform = get_transform(args)
    sample_transform = get_sample_transform(args)

    train_set = get_dataset(args, args.train_set, args.train_split, transform=transform, sample_transform=sample_transform)
    test_set = get_dataset(args, args.test_set, args.test_split, transform=transform)

    return train_set, test_set

def get_dataset(args, dataset: str, split_file: str, transform = None, sample_transform = None):
    root = os.path.join(args.data_root, dataset)

    if args.bounding_boxes:
        return BoundingBoxDataset(root, args.data_type, split_file, 'pyannot.pkl', transform=transform, sample_transform=sample_transform) # TODO (if needed): UCF24 dataset for features in txt
    elif 'rgb-images' in args.data_type:
        return ProgressVideoDataset(root, args.data_type, split_file, transform=transform, sample_transform=sample_transform)
    else:
        return ProgressFeatureDataset(root, args.data_type, split_file, transform=transform, sample_transform=sample_transform)

def get_transform(args):
    transform = []
    if args.data_type == 'rgb-images':
        transform.append(transforms.ToTensor())

    # TODO
    # if args.data_modifier == 'indices':
    #     transform.append(Indices(args.indices_normalization))
    # elif args.data_modifier == 'ones':
    #     transform.append(Ones())
    # elif args.data_modifier == 'randoms':
    #     transform.append(Randoms())

    return transforms.Compose(transform)

def get_sample_transform(args):
    return transforms.Compose([
        Subsection(p=args.subsection_chance),
        Subsample(p=args.subsample_chance)
    ])
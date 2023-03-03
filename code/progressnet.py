import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from os.path import join
import logging

from utils import parse_arguments, get_device, set_seeds, create_directory
from datasets import BoundingBoxDataset, bounding_box_collate
from datasets.transforms import ImglistToTensor
from networks import ProgressNet
from losses import bo_weight

"""
implementation of https://arxiv.org/abs/1705.01781
"""

def train(network, batch, l1_loss, l2_loss, device, optimizer=None):
    video_names, frames, boxes, labels, lengths = batch
    frames = frames.to(device)
    boxes = boxes.to(device)
    labels = labels.to(device)
    if optimizer:
        optimizer.zero_grad
    predictions = network(frames, boxes, lengths)
    # progress is in range (0, 1], but batch is zero-padded
    # we can use this to fill our loss with 0s for padded values
    predictions.masked_fill_(labels == 0, 0.0)

    bo = bo_weight(p, p_hat)(labels, predictions)
    l1 = l1_loss(predictions, labels)
    l2 = l2_loss(predictions, labels)
    count = lengths.sum()
    if optimizer:
        loss = l1 * bo
        loss = loss.sum() / count
        loss.backward()
        optimizer.step()

    return l1, l2, bo, count


def main():
    # TODO: a lot of this first part is repeated, perhaps it could be extracted
    args = parse_arguments()
    set_seeds(args.seed)
    device = get_device()
    # create paths & directories
    annotation_path = join(args.splitfile_dir, args.annotation_file)
    train_splitfile_path = join(args.splitfile_dir, args.train_split_file)
    test_splitfile_path = join(args.splitfile_dir, args.test_split_file)
    experiment_directory = join(args.data_root, args.dataset, args.experiment_directory, args.experiment_name)
    log_directory = join(experiment_directory, args.log_directory)
    model_directory = join(experiment_directory, args.model_directory)
    figures_directory = join(experiment_directory, args.figures_directory)
    # create directories
    create_directory(experiment_directory)
    create_directory(log_directory)
    create_directory(model_directory)
    create_directory(figures_directory)

    # setup logging
    logging.basicConfig(
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(logging_path),
            logging.StreamHandler()
        ]
    )
    # create datasets
    train_set = BoundingBoxDataset(args.data_root, arg.data_type, annotation_path, train_splitfile_path, lambda x: f'{(x+1):05d}.png')
    test_set = BoundingBoxDataset(args.data_root, arg.data_type, annotation_path, test_splitfile_path, lambda x: f'{(x+1):05d}.png')
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, collate_fn=bounding_box_collate)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=bounding_box_collate)

    network = ProgressNet().to(device)
    # TODO: model loading
    l1_loss = nn.L1Loss(reduction='none')
    l2_loss = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(network.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        train_bo_loss, test_bo_loss = 0.0, 0.0
        train_l1_loss, test_l1_loss = 0.0, 0.0
        train_l2_loss, test_l2_loss = 0.0, 0.0
        train_count, test_count = 0, 0
        net.train()
        for batch in train_loader:
            l1, l2, bo_weight, count = train(network, batch, l1_loss, l2_loss, device, optimizer=optimizer)
            train_bo_loss += (l1 * bo_weight).sum().item()
            train_l1_loss += l1.sum().item()
            train_l2_loss += l2_loss.sum().item()
            train_count += count.item()
        net.eval()
        for batch in test_loader:
            l1, l2, bo_weight, count = train(network, batch, l1_loss, l2_loss, device)
            test_bo_loss += (l1 * bo_weight).sum().item()
            test_l1_loss += l1.sum().item()
            test_l2_loss += l2_loss.sum().item()
            test_count += count.item()
        print(f'[{epoch:03d} train] avg bo loss {(train_bo_loss / train_count):03d} avg l1 loss {(train_l1_loss / train_count):03d} avg l2 loss {(train_l2_loss / train_count):03d}')
        print(f'[{epoch:03d} test] avg bo loss {(test_bo_loss / train_count):03d} avg l1 loss {(test_l1_loss / train_count):03d} avg l2 loss {(test_l2_loss / train_count):03d}')
if __name__ == '__main__':
    main()
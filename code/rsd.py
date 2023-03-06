import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from os.path import join
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import setup
from datasets import RSDDataset, rsd_collate
from datasets.transforms import ImglistToTensor
from networks import RSDNet

"""
implementation of https://arxiv.org/abs/1705.01781
"""

def train(network, batch, smooth_l1_criterion, l1_criterion, l2_criterion, device, optimizer=None):
    video_names, frames, rsd_values, progress_values, lengths = batch
    frames = frames.to(device)
    rsd_values = rsd_values.to(device)
    progress_values = progress_values.to(device)
    if optimizer:
        optimizer.zero_grad()
    rsd_predictions, progress_predictions = network(frames, lengths)
    # progress is in range (0, 1], but batch is zero-padded
    # we can use this to fill our loss with 0s for padded values
    mask = (progress_values != 0).int().to(device)

    rsd_predictions = rsd_predictions * mask
    progress_predictions = progress_predictions * mask

    rsd_loss = smooth_l1_criterion(rsd_predictions, rsd_values)
    progress_loss = smooth_l1_criterion(progress_predictions, progress_values)
    loss = rsd_loss + progress_loss

    progress_l1_loss = l1_criterion(progress_predictions, progress_values)
    progress_l2_loss = l2_criterion(progress_predictions, progress_values)

    count = lengths.sum()
    if optimizer:
        loss = loss.sum() / count
        loss.backward()
        optimizer.step()

    return rsd_predictions, progress_predictions, loss, rsd_loss, progress_loss, progress_l1_loss, progress_l2_loss, count

def main():
    args, dirs, device = setup()

    # create datasets
    train_set = RSDDataset(dirs['dataset_directory'], args.data_type, dirs['train_splitfile_path'], transform=ImglistToTensor(dim=0))
    test_set = RSDDataset(dirs['dataset_directory'], args.data_type, dirs['test_splitfile_path'], transform=ImglistToTensor(dim=0))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, collate_fn=rsd_collate)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, collate_fn=rsd_collate)

    # load model
    net = RSDNet().to(device)
    if args.model_name:
        model_path = join(dirs['model_directory'], args.model_name)
        net.load_state_dict(torch.load(model_path))

    # criterions & optimizer
    smooth_l1_criterion = nn.SmoothL1Loss(reduction='none')
    l1_criterion = nn.L1Loss(reduction='none')
    l2_criterion = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

    logging.info(f'[{args.experiment_name}] starting experiment')
    for epoch in range(args.epochs):
        train_loss, train_rsd_loss, train_progress_loss, train_progress_l1_loss, train_progress_l2_loss, train_count = 0.0, 0.0, 0.0, 0.0, 0.0, 0
        test_loss, test_rsd_loss, test_progress_loss, test_progress_l1_loss, test_progress_l2_loss, test_count = 0.0, 0.0, 0.0, 0.0, 0.0, 0

        if not args.eval:
            net.train()
            for batch in tqdm(train_loader, leave=False):
                rsd_predictions, progress_predictions, loss, rsd_loss, progress_loss, progress_l1_loss, progress_l2_loss, count = train(net, batch, smooth_l1_criterion, l1_criterion, l2_criterion, device, optimizer=optimizer)
                train_loss += loss.sum().item()
                train_rsd_loss += loss.sum().item()
                train_progress_loss += progress_loss.sum().item()
                train_progress_l1_loss += progress_l1_loss.sum().item()
                train_progress_l2_loss += progress_l2_loss.sum().item()
                train_count += count.item()

            logging.info(f'[{epoch:03d} train]        avg loss {(train_loss / train_count):.4f}, avg rsd loss {(train_rsd_loss / train_count):.4f}, avg progress loss {(train_progress_loss / train_count):.4f}, avg progress l1 loss {(train_progress_l1_loss / train_count):.4f}, avg progress l2 loss {(train_progress_l2_loss / train_count):.4f}')
        
            if epoch % args.save_every == 0 and epoch > 0:
                model_name = f'{epoch:03d}.pth'
                model_path = join(dirs['model_directory'], model_name)
                logging.info(f'[{epoch:03d}] saving model {model_name}')
                torch.save(net.state_dict(), model_path)

        net.eval()
        for batch_index, batch in tqdm(enumerate(test_loader), leave=False, total=len(test_loader)):
            do_figure = args.figures and args.batch_size == 1 and batch_index % args.figure_every == 0

            rsd_predictions, progress_predictions, loss, rsd_loss, progress_loss, progress_l1_loss, progress_l2_loss, count = train(net, batch, smooth_l1_criterion, l1_criterion, l2_criterion, device, optimizer=optimizer)
            train_l1_loss += l1_loss.sum().item()
            train_l2_loss += l2_loss.sum().item()
            train_count += count.item()
        for model_name in networks:
            logging.info(f'[{epoch:03d} test {model_name}] avg bo loss {(networks[model_name]["bo_loss"] / networks[model_name]["count"]):.4f}, avg l1 loss {(networks[model_name]["l1_loss"] / networks[model_name]["count"]):.4f}, avg l2 loss {(networks[model_name]["l2_loss"] / networks[model_name]["count"]):.4f}')
        
        if args.eval:
            break # only 1 epoch for evaluation


if __name__ == '__main__':
    main()
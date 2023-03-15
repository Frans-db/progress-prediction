import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
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

def train(network, batch, smooth_l1_criterion, l1_criterion, l2_criterion, device, optimizer=None, train=True):
    video_names, frames, rsd_values, progress_values, lengths = batch
    frames = frames.to(device)
    rsd_values = rsd_values.to(device)
    progress_values = progress_values.to(device)
    if optimizer:
        optimizer.zero_grad()
    rsd_predictions, progress_predictions = network(frames, lengths)

    if train:
        rsd_values = rsd_values / 5
    else:
        rsd_predictions = rsd_predictions * 5
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
    rsd_l1_loss = l1_criterion(rsd_predictions, rsd_values)
    rsd_l2_loss = l2_criterion(rsd_predictions, rsd_values)

    count = lengths.sum()
    if optimizer:
        avg_loss = loss.sum() / count
        avg_loss.backward()
        optimizer.step()

    return {
        'rsd_predictions': rsd_predictions,
        'progress_predictions': progress_predictions,
        'loss': loss,
        'rsd_loss': rsd_loss,
        'progress_loss': progress_loss,
        'rsd_l1_loss': rsd_l1_loss,
        'rsd_l2_loss': rsd_l2_loss,
        'progress_l1_loss': progress_l1_loss,
        'progress_l2_loss': progress_l2_loss,
        'count': count
    }

def main():
    args, dirs, device = setup()

    # create datasets
    train_set = RSDDataset(dirs['dataset_directory'], args.data_type, dirs['train_splitfile_path'], mode='minutes')
    test_set = RSDDataset(dirs['dataset_directory'], args.data_type, dirs['test_splitfile_path'], mode='minutes')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, collate_fn=rsd_collate)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, collate_fn=rsd_collate)

    net = RSDNet(p_dropout=args.dropout_chance).to(device)
    if args.model_name:
        model_path = join(dirs['model_directory'], args.model_name)
        net.load_state_dict(torch.load(model_path))

    # criterions & optimizer
    smooth_l1_criterion = nn.SmoothL1Loss(reduction='none')
    l1_criterion = nn.L1Loss(reduction='none')
    l2_criterion = nn.MSELoss(reduction='none')
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_lr_every, gamma=args.lr_decay)

    logging.info(f'[{args.experiment_name}] starting experiment')
    for epoch in range(args.epochs):
        train_results = {
            'loss': 0.0,
            'rsd_loss': 0.0,
            'progress_loss': 0.0,
            'rsd_l1_loss': 0.0,
            'rsd_l2_loss': 0.0,
            'progress_l1_loss': 0.0,
            'progress_l2_loss': 0.0,
            'count': 0
        }
        test_results = train_results.copy()

        if not args.eval:
            net.train()
            for batch in tqdm(train_loader, leave=False):
                batch_result = train(net, batch, smooth_l1_criterion, l1_criterion, l2_criterion, device, optimizer=optimizer)
                for key in batch_result:
                    if key in train_results:
                        train_results[key] += batch_result[key].sum().item()

            log_string = f'[{epoch:03d} train]'
            for key in train_results:
                if key == 'count': 
                    continue
                log_string += f' {key} {(train_results[key] / train_results["count"]):.3f},'
            logging.info(log_string)

            if epoch % args.save_every == 0 and epoch > 0:
                model_name = f'{epoch:03d}.pth'
                model_path = join(dirs['model_directory'], model_name)
                logging.info(f'[{epoch:03d}] saving model {model_name}')
                torch.save(net.state_dict(), model_path)

        net.eval()
        for batch_index, batch in tqdm(enumerate(test_loader), leave=False, total=len(test_loader)):
            do_figure = args.figures and args.batch_size == 1 and batch_index % args.figure_every == 0

            batch_result = train(net, batch, smooth_l1_criterion, l1_criterion, l2_criterion, device)
            for key in batch_result:
                if key in train_results:
                    test_results[key] += batch_result[key].sum().item()

        log_string = f'[{epoch:03d} test]'
        for key in test_results:
            if key == 'count': 
                continue
            log_string += f' {key} {(test_results[key] / test_results["count"]):.3f},'
        logging.info(log_string)
        scheduler.step()


        if args.eval:
            break # only 1 epoch for evaluation


if __name__ == '__main__':
    main()
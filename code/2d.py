import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from os.path import join
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import setup
from datasets import ImageDataset
from datasets.transforms import ImglistToTensor
from networks import RSDNet

"""
implementation of https://arxiv.org/abs/1705.01781
"""

def train(network, batch, smooth_l1_criterion, l1_criterion, l2_criterion, device, optimizer=None):
    frames, labels = batch
    frames = frames.to(device)
    labels = labels.to(device)

    if optimizer:
        optimizer.zero_grad()
    predictions = network(frames)
    loss = smooth_l1_criterion(predictions.squeeze(), labels)
    l1_loss = l1_criterion(predictions.squeeze(), labels)
    l2_loss = l2_criterion(predictions.squeeze(), labels)

    if optimizer:
        avg_loss = loss.sum() / frames.shape[0]
        avg_loss.backward()
        optimizer.step()

    return predictions, loss, l1_loss, l2_loss, frames.shape[0]

def main():
    args, dirs, device = setup()

    # create datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224))
    ])
    train_set = ImageDataset(dirs['dataset_directory'], args.data_type, dirs['train_splitfile_path'], transform=transform)
    test_set = ImageDataset(dirs['dataset_directory'], args.data_type, dirs['test_splitfile_path'], transform=transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    # load model
    net = models.resnet152().to(device)
    net.fc = nn.Sequential(
        nn.Linear(2048, 1),
        nn.Sigmoid()
    ).to(device)

    if args.model_name:
        model_path = join(dirs['model_directory'], args.model_name)
        logging.info(f'[{epoch:03d}] saving model {model_path}')
        net.load_state_dict(torch.load(model_path))

    # criterions & optimizer
    smooth_l1_criterion = nn.SmoothL1Loss(reduction='none')
    l1_criterion = nn.L1Loss(reduction='none')
    l2_criterion = nn.MSELoss(reduction='none')
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_lr_every, gamma=args.lr_decay)

    logging.info(f'[{args.experiment_name}] starting experiment')
    for epoch in range(args.epochs):
        train_loss, train_l1_loss, train_l2_loss, train_count = 0.0, 0.0, 0.0, 0
        test_loss, test_l1_loss, test_l2_loss, test_count = 0.0, 0.0, 0.0, 0

        if not args.eval:
            net.train()
            for batch in tqdm(train_loader, leave=False):
                predictions, loss, l1_loss, l2_loss, count = train(net, batch, smooth_l1_criterion, l1_criterion, l2_criterion, device, optimizer=optimizer)
                train_loss += loss.sum().item()
                train_l1_loss += l1_loss.sum().item()
                train_l2_loss += l2_loss.sum().item()
                train_count += count

            logging.info(f'[{epoch:03d} train] avg loss {(train_loss / train_count):.4f}, avg l1 loss {(train_l1_loss / train_count):.4f}, avg l2 loss {(train_l2_loss / train_count):.4f}')

            if epoch % args.save_every == 0 and epoch > 0:
                model_name = f'{epoch:03d}_basenet.pth'
                model_path = join(dirs['model_directory'], model_name)
                logging.info(f'[{epoch:03d}] saving model {model_name}')
                torch.save(net.state_dict(), model_path)

        net.eval()
        for batch_index, batch in tqdm(enumerate(test_loader), leave=False, total=len(test_loader)):
            predictions, loss, l1_loss, l2_loss, count = train(net, batch, smooth_l1_criterion, l1_criterion, l2_criterion, device, optimizer=optimizer)
            test_loss += loss.sum().item()
            test_l1_loss += l1_loss.sum().item()
            test_l2_loss += l2_loss.sum().item()
            test_count += count

        logging.info(f'[{epoch:03d} test] avg loss {(test_loss / test_count):.4f}, avg l1 loss {(test_l1_loss / test_count):.4f}, avg l2 loss {(test_l2_loss / test_count):.4f}')
        scheduler.step()

        if args.eval:
            break # only 1 epoch for evaluation


if __name__ == '__main__':
    main()
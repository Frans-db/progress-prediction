import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from os.path import join
import logging
from tqdm import tqdm

from utils import get_device, set_seeds, create_directory
from args import parse_arguments
from datasets import BoundingBoxDataset, bounding_box_collate
from datasets.transforms import ImglistToTensor
from networks import ProgressNet, LSTMRelativeNet
from networks import RandomNet, StaticNet, RelativeNet
from losses import bo_weight

"""
implementation of https://arxiv.org/abs/1705.01781
"""

def train(network, batch, l1_criterion, l2_criterion, device, optimizer=None):
    video_names, frames, boxes, labels, lengths = batch
    frames = frames.to(device)
    boxes = boxes.to(device)
    labels = labels.to(device)
    if optimizer:
        optimizer.zero_grad()
    predictions = network(frames, boxes, lengths)
    # progress is in range (0, 1], but batch is zero-padded
    # we can use this to fill our loss with 0s for padded values
    predictions.masked_fill_(labels == 0, 0.0)

    bo_loss = bo_weight(labels, predictions)
    l1_loss = l1_criterion(predictions, labels)
    l2_loss = l2_criterion(predictions, labels)
    count = lengths.sum()
    if optimizer:
        loss = l1_loss * bo_loss
        loss = loss.sum() / count
        loss.backward()
        optimizer.step()

    return predictions, l1_loss, l2_loss, bo_loss, count

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

def main():
    # TODO: a lot of this first part is repeated, perhaps it could be extracted
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

    # create datasets
    if 'toy' in args.dataset:
        name_fn = lambda x: f'img_{(x):05d}.png'
    else:
        name_fn = lambda x: f'{(x+1):05d}.jpg'
    train_set = BoundingBoxDataset(dirs['dataset_directory'], args.data_type, dirs['annotation_path'], dirs['train_splitfile_path'], name_fn, transform=ImglistToTensor(dim=0))
    test_set = BoundingBoxDataset(dirs['dataset_directory'], args.data_type, dirs['annotation_path'], dirs['test_splitfile_path'], name_fn, transform=ImglistToTensor(dim=0))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, collate_fn=bounding_box_collate)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, collate_fn=bounding_box_collate)

    # load model
    # if args.model_type == 'progressnet':
    net = ProgressNet(embed_size=args.embed_size, p_dropout=args.dropout_chance).to(device)
    # elif args.model_type == 'relativenet':
    #     net = LSTMRelativeNet(device, train_set.get_average_tube_frame_length()).to(device)
    if args.model_name:
        model_path = join(dirs['model_directory'], args.model_name)
        net.load_state_dict(torch.load(model_path))

    # setup dict for evaluation/testing
    networks = {}
    loss_dict = {'bo_loss': 0.0, 'l1_loss': 0.0, 'l2_loss': 0.0, 'count': 0}
    networks['network'] = loss_dict.copy()
    networks['network']['net'] = net
    if args.random:
        networks['random'] = loss_dict.copy()
        networks['random']['net'] = RandomNet(device)
    if args.static:
        networks['static'] = loss_dict.copy()
        networks['static']['net'] = StaticNet(device)
    if args.relative:
        networks['relative'] = loss_dict.copy()
        networks['relative']['net'] = RelativeNet(device, train_set.get_average_tube_frame_length())

    l1_criterion = nn.L1Loss(reduction='none')
    l2_criterion = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

    logging.info(f'[{args.experiment_name}] starting experiment')
    for epoch in range(args.epochs):
        train_bo_loss, train_l1_loss, train_l2_loss, train_count = 0.0, 0.0, 0.0, 0

        if not args.eval:
            net.train()
            for batch in tqdm(train_loader, leave=False):
                predictions, l1_loss, l2_loss, bo_weight, count = train(net, batch, l1_criterion, l2_criterion, device, optimizer=optimizer)
                train_bo_loss += (l1_loss * bo_weight).sum().item()
                train_l1_loss += l1_loss.sum().item()
                train_l2_loss += l2_loss.sum().item()
                train_count += count.item()

            logging.info(f'[{epoch:03d} train] avg bo loss {(train_bo_loss / train_count):.4f}, avg l1 loss {(train_l1_loss / train_count):.4f}, avg l2 loss {(train_l2_loss / train_count):.4f}')
        
            if epoch % args.save_every == 0 and epoch > 0:
                model_name = f'{epoch:03d}.pth'
                model_path = join(dirs['model_directory'], model_name)
                logging.info(f'[{epoch:03d}] saving model {model_name}')
                torch.save(net.state_dict(), model_path)

        net.eval()
        for batch in tqdm(test_loader, leave=False):
            for model_name in networks:
                predictions, l1_loss, l2_loss, bo_weight, count = train(networks[model_name]['net'], batch, l1_criterion, l2_criterion, device)
                networks[model_name]['bo_loss'] += (l1_loss * bo_weight).sum().item()
                networks[model_name]['l1_loss'] += l1_loss.sum().item()
                networks[model_name]['l2_loss'] += l2_loss.sum().item()
                networks[model_name]['count'] += count.item()

                if args.figures and args.batch_size == 1:
                    pass

            if args.figures and args.batch_size == 1:
                pass

        for model_name in networks:
            logging.info(f'[{epoch:03d} test {model_name}] avg bo loss {(networks[model_name]["bo_loss"] / networks[model_name]["count"]):.4f}, avg l1 loss {(networks[model_name]["l1_loss"] / networks[model_name]["count"]):.4f}, avg l2 loss {(networks[model_name]["l2_loss"] / networks[model_name]["count"]):.4f}')
        
        # TODO: Figures
        if args.eval:
            break # only 1 epoch for evaluation


if __name__ == '__main__':
    main()
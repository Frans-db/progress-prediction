import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as transforms
import argparse
import os
import matplotlib.pyplot as plt
import wandb
import random
import numpy as np

from progress_dataset import ProgressDataset
from ucf_dataset import UCFDataset
# from networks import ProgressNet, PooledProgressNet, RNNProgressNet, init_weights
# from networks import SimpleProgressNet, SpatialProgressNet, WeirdProgressNet, WeirderProgressNet
from networks import init_weights
from networks import TinyProgressNet, OracleProgressNet, TinyLSTMNet
from networks import SeparateSingleLSTM, SeparateDoubleLSTM
from augmentations import Subsection, Subsample, Removal

COLORS = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'lime', 'darkblue']

def collate_fn(batch):
    video_names, frames, progress = zip(*batch)
    lengths = torch.Tensor([sample.shape[0] for sample in frames])

    padded_frames = pad_sequence(frames, batch_first=True)
    padded_progress = pad_sequence(progress, batch_first=True)

    return video_names, padded_frames, padded_progress, lengths

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Experiment seed')
    # wandb
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb')
    parser.add_argument('--group', type=str, default='default', help='Used to group charts together in wandb')
    # plots
    parser.add_argument('--plots', action='store_true', help='Enable plots')
    parser.add_argument('--plot_every', type=int, default=25, help='How often to plot')
    parser.add_argument('--plot_directory', type=str, default='plots')
    # network
    parser.add_argument('--network', type=str, default='progressnet', help='Network to use: progressnet / pooled_progressnet')
    # data
    parser.add_argument('--dataset', type=str, default='toy', help='Which dataset to use')
    # forecasting
    parser.add_argument('--delta_t', type=int, default=10, help='Number of timesteps in the future to forecast')
    # training
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-3, help='Learning rate')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of iterations')
    parser.add_argument('--test_every', type=int, default=100, help='On which iterations to test')
    parser.add_argument('--augmentations', nargs='+', default='', help='List of augmentations: subsection, subsample, removal')
    parser.add_argument('--p_dropout', type=float, default=0.5, help='Dropout chance of progressnet')
    parser.add_argument('--losses', nargs='+', default='progress', help='List of extra losses: forecast, embedding')


    return parser.parse_args()

def get_sample_transform(augmentations):
    sample_transform_list = []
    if 'subsection' in augmentations:
        sample_transform_list.append(Subsection())
    if 'subsample' in augmentations:
        sample_transform_list.append(Subsample())
    if 'removal' in augmentations:
        sample_transform_list.append(Removal())
    return transforms.Compose(sample_transform_list)
    
def train(network, batch, l1_criterion, l2_criterion, args, device, optimizer=None):
    # batch data
    video_names, frames, progress, lengths = batch
    forecasted_progress = torch.ones_like(progress)
    forecasted_progress[:, :-args.delta_t] = progress[:, args.delta_t:]
    # forward pass
    predictions, forecasted_predictions, embeddings, forecasted_embeddings = network(frames.to(device))
    # calculate adj. predictions
    B, S = forecasted_predictions.shape
    indices = torch.arange(1, S+1).repeat(B, 1)
    timesteps = torch.clamp(indices + args.delta_t, max=S+1)
    adj_predictions = (forecasted_predictions / timesteps.to(device)) * indices.to(device)
    # loss calculations
    # TODO: Mask when B > 0    
    progress = progress.to(device)
    l1_loss = l1_criterion(predictions, progress)
    l2_loss = l2_criterion(predictions, progress)
    l1_adjusted_loss = l1_criterion(adj_predictions, progress)
    l2_adjusted_loss = l2_criterion(adj_predictions, progress)
    forecasted_progress = forecasted_progress.to(device)
    l1_forecast_loss = l1_criterion(forecasted_predictions, forecasted_progress.to(device))
    l2_forecast_loss = l2_criterion(forecasted_predictions, forecasted_progress.to(device))
    l2_embedding_loss = l2_criterion(embeddings[:, args.delta_t:, :], forecasted_embeddings[:, :-args.delta_t, :])
    # optimizer
    if optimizer:
        optimizer.zero_grad()
        loss = l2_loss.clone()
        if 'forecast' in args.losses:
            loss += l2_forecast_loss
        if 'embedding' in args.losses:
            loss += l2_embedding_loss
        loss.backward()

        optimizer.step()

    return {
        'video_names': video_names,
        'predictions': predictions.cpu().detach(),
        'adjusted_predictions': adj_predictions.cpu().detach(),
        'forecasted_predictions': forecasted_predictions.cpu().detach(),
        'progress': progress.cpu().detach(),
        'forecasted_progress': forecasted_progress.cpu().detach(),
        'l1_loss': l1_loss.item(),
        'l1_adjusted_loss': l1_adjusted_loss.item(),
        'l1_forecast_loss': l1_forecast_loss.item(),
        'l2_loss': l2_loss.item(),
        'l2_adjusted_loss': l2_adjusted_loss.item(),
        'l2_forecast_loss': l2_forecast_loss.item(),
        'l2_embedding_loss': l2_embedding_loss.item(),
        'count': lengths.sum().item()
    }

def get_empty_results() -> dict:
    return {
        'l1_loss': 0.0,
        'l1_adjusted_loss': 0.0,
        'l1_forecast_loss': 0.0,
        'l2_loss': 0.0,
        'l2_adjusted_loss': 0.0,
        'l2_forecast_loss': 0.0,
        'l2_embedding_loss': 0.0,
        'count': 0
    }

def update_results(results: dict, batch_results: dict) -> None:
    for key in results:
        if key in batch_results:
            results[key] += batch_results[key]

def wandb_log(results: dict, iteration: int, prefix: str) -> None:
    wandb.log({
        f'{prefix}_l1_loss': results['l1_loss'] / results['count'],
        f'{prefix}_l2_loss': results['l2_loss'] / results['count'],

        f'{prefix}_l1_forecast_loss': results['l1_forecast_loss'] / results['count'],
        f'{prefix}_l2_forecast_loss': results['l2_forecast_loss'] / results['count'],

        f'{prefix}_l1_adjusted_loss': results['l1_adjusted_loss'] / results['count'],
        f'{prefix}_l2_adjusted_loss': results['l2_adjusted_loss'] / results['count'],

        f'{prefix}_l2_embedding_loss': results['l2_embedding_loss'] / results['count'],

        'count': results['count'],
        'iteration': iteration,
    })

def get_network(args) -> nn.Module:
    # if args.network == 'progressnet':
    #     progressnet = ProgressNet(p_dropout=args.p_dropout)
    # elif args.network == 'pooled_progressnet':
    #     progressnet = PooledProgressNet(p_dropout=args.p_dropout)
    # elif args.network == 'rnn_progressnet':
    #     progressnet = RNNProgressNet(p_dropout=args.p_dropout)
    # elif args.network == 'simple_progressnet':
    #     progressnet = SimpleProgressNet(p_dropout=args.p_dropout)
    # elif args.network == 'spatial_progressnet':
    #     progressnet = SpatialProgressNet(p_dropout=args.p_dropout)
    # elif args.network == 'weird_progressnet':
    #     progressnet = WeirdProgressNet(p_dropout=args.p_dropout, delta_t=args.delta_t)
    # elif args.network == 'weirder_progressnet':
    #     progressnet = WeirderProgressNet(p_dropout=args.p_dropout, delta_t=args.delta_t)
    if args.network == 'tiny_progressnet':
        return TinyProgressNet()
    elif args.network == 'oracle_progressnet':
        return OracleProgressNet(delta_t=args.delta_t)
    elif args.network == 'separate_single_lstm':
        return SeparateSingleLSTM()
    elif args.network == 'separate_double_lstm':
        return SeparateDoubleLSTM()
        
    raise Exception(f'Network {args.network} does not exist')

def main():
    device = get_device()
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not args.no_wandb:
        wandb.init(
            project='mscfransdeboer',
            config={
                'seed': args.seed,
                'network': args.network,
                'dataset': args.dataset,
                'augmentations': args.augmentations,
                'losses': args.losses,
                'delta_t': args.delta_t,
                'batch_size': args.batch_size,
                'lr': args.lr,
                'iterations': args.iterations,
                'test_every': args.test_every,
                'p_dropout': args.p_dropout,
                'group': args.group,
            }
        )

    if args.plots:
        os.mkdir(f'./plots/{args.plot_directory}')

    root = f'/home/frans/Datasets/{args.dataset}'
    data_type = 'rgb-images'

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    sample_transform = get_sample_transform(args.augmentations)
    if args.dataset == 'ucf24':
        trainset = UCFDataset(root, data_type, 'splitfiles/trainlist01.txt', 'splitfiles/pyannot.pkl', transform=transform, sample_transform=sample_transform)
        testset = UCFDataset(root, data_type, 'splitfiles/testlist01.txt', 'splitfiles/pyannot.pkl', transform=transform, sample_transform=sample_transform)
    else:
        trainset = ProgressDataset(root, data_type, 'splitfiles/trainlist01.txt', transform=transform, sample_transform=sample_transform)
        testset = ProgressDataset(root, data_type, 'splitfiles/testlist01.txt', transform=transform)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, num_workers=2, shuffle=True, collate_fn=collate_fn)
    testloader = DataLoader(testset, batch_size=args.batch_size, num_workers=2, shuffle=False, collate_fn=collate_fn)

    progressnet = get_network(args).to(device)
    progressnet.apply(init_weights)

    optimizer = optim.Adam(progressnet.parameters(), lr=args.lr)
    l1_criterion = nn.L1Loss(reduction='sum')
    l2_criterion = nn.MSELoss(reduction='sum')

    iteration = 0
    train_results = get_empty_results()
    test_results = get_empty_results()
    done = False
    while not done:
        # train iteration
        for batch in trainloader:
            batch_result = train(progressnet, batch, l1_criterion, l2_criterion, args, device, optimizer=optimizer)
            if not args.no_wandb:
                wandb_log(batch_result, iteration, 'train')
            update_results(train_results, batch_result)

            # test iteration
            if iteration % args.test_every == 0:
                for i,batch in enumerate(testloader):
                    batch_result = train(progressnet, batch, l1_criterion, l2_criterion, args, device)
                    update_results(test_results, batch_result)
                    if args.plots and i % args.plot_every == 0:
                        iterator = zip(
                            batch_result['video_names'],
                            batch_result['progress'], 
                            batch_result['predictions'],
                            batch_result['adjusted_predictions'], 
                            batch_result['forecasted_predictions']
                        )
                        for video_name, progress, predictions, adj_predictions, forecasted_predictions in iterator:
                            S = predictions.shape[0]
                            indices = range(S)
                            forecasted_indices = range(args.delta_t, S + args.delta_t)
                            plt.plot(indices, progress, label='progress')
                            plt.plot(indices, predictions, label='predicted progress')
                            plt.plot(forecasted_indices, forecasted_predictions, label='forecasted predictions')
                            plt.plot(indices, adj_predictions, label='adjusted progress')
                            
                            if 'toy' in args.dataset:
                                action_labels = testset.get_action_labels(video_name)
                                for j, label in enumerate(action_labels):
                                    plt.axvspan(j-0.5, j+0.5, facecolor=COLORS[label], alpha=0.2, zorder=-1)
                            plt.legend(loc='best')
                            plt.xlabel('Frame')
                            plt.ylabel('Progress (%)')
                            plt.title(f'{args.dataset} - {video_name}')
                            plt.savefig(f'./plots/{args.plot_directory}/{iteration}_{i}.png')
                            plt.clf()

                if not args.no_wandb:
                    wandb_log(test_results, iteration, 'test')
                    wandb_log(train_results, iteration, 'avg_train')
                train_results = get_empty_results()
                test_results = get_empty_results()
            
            iteration += 1
            if iteration > args.iterations:
                done = True
                break

    wandb.finish()

if __name__ == '__main__':
    main()
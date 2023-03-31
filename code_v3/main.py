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
from network import ProgressNet, UnrolledProgressNet
from augmentations import Subsection, Subsample, Removal

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
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_wandb', action='store_true')
    # plots
    parser.add_argument('--plots', action='store_true')
    parser.add_argument('--plot_every', type=int, default=25)
    # data
    parser.add_argument('--dataset', type=str, default='toy')
    parser.add_argument('--augmentations', nargs='+', default='')
    # forecasting
    parser.add_argument('--forecast', action='store_true')
    parser.add_argument('--delta_t', type=int, default=10)
    # training
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--iterations', type=int, default=1001)
    parser.add_argument('--test_every', type=int, default=100)


    return parser.parse_args()

def get_sample_transform(augmentations: list[str]):
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
    predictions, forecasted_predictions = network(frames.to(device))
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
    # optimizer
    if optimizer:
        optimizer.zero_grad()
        if args.forecast:
            (l2_loss + l2_forecast_loss).backward()
        else:
            l2_loss.backward()
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
        'iteration': iteration,
    })

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
                'dataset': args.dataset,
                'augmentations': args.augmentations,
                'forecast': args.forecast,
                'delta_t': args.delta_t,
                'batch_size': args.batch_size,
                'lr': args.lr,
                'iterations': args.iterations,
                'test_every': args.test_every,
            }
        )

    root = f'/home/frans/Datasets/{args.dataset}'
    data_type = 'rgb-images'

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    sample_transform = get_sample_transform(args.augmentations)
    trainset = ProgressDataset(root, data_type, 'splitfiles/trainlist01.txt', transform=transform)
    testset = ProgressDataset(root, data_type, 'splitfiles/testlist01.txt', transform=transform)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, num_workers=2, shuffle=True, collate_fn=collate_fn)
    testloader = DataLoader(testset, batch_size=args.batch_size, num_workers=2, shuffle=False, collate_fn=collate_fn)

    progressnet = UnrolledProgressNet(p_dropout=0.5).to(device)
    progressnet.apply(ProgressNet.init_weights)

    optimizer = optim.Adam(progressnet.parameters(), lr=args.lr)
    l1_criterion = nn.L1Loss(reduction='sum')
    l2_criterion = nn.MSELoss(reduction='sum')

    iteration = 0
    train_results = get_empty_results()
    test_results = get_empty_results()
    while iteration < args.iterations:
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
                            plt.plot(indices, adj_predictions, label='adjusted progress')
                            plt.plot(forecasted_indices, forecasted_predictions, label='forecasted predictions')
                            action_labels = testset.get_action_labels(video_name)
                            for j, label in enumerate(action_labels):
                                plt.axvspan(j-0.5, j+0.5, facecolor=['r', 'g', 'b', 'c', 'm', 'y'][label], alpha=0.2, zorder=-1)
                            plt.legend(loc='best')
                            plt.xlabel('Frame')
                            plt.ylabel('Progress (%)')
                            plt.savefig(f'./plots/{iteration}_{i}.png')
                            plt.clf()

                if not args.no_wandb:
                    wandb_log(test_results, iteration, 'test')
                    wandb_log(train_results, iteration, 'avg_train')
                train_results = get_empty_results()
                test_results = get_empty_results()
            
            # update & log to wandb
            iteration += 1

    # for epoch in range(args.epochs):
    #     train_loss, train_forecast_loss, train_count = 0.0, 0.0, 0
    #     test_loss, test_forecast_loss, test_count = 0.0, 0.0, 0
    #     network.train()
    #     for video_names, frames, progress, lengths in trainloader:
    #         predictions, forecasted_predictions = network(frames.to(device))

    #         forecasted_progress = torch.ones_like(progress)
    #         forecasted_progress[:, :-DELTA_T] = progress[:, DELTA_T:]

    #         loss = l2_criterion(predictions, progress.to(device))
    #         forecast_loss = l2_criterion(forecasted_predictions, forecasted_progress.to(device))

    #         optimizer.zero_grad()
    #         (loss + forecast_loss).backward()
    #         optimizer.step()

    #         train_loss += loss.item()
    #         train_forecast_loss += forecast_loss.item()
    #         train_count += lengths.sum().item()


    #     os.mkdir(f'./plots/{epoch}')
    #     network.eval()
    #     for i, (video_names, frames, progress, lengths) in enumerate(testloader):
    #         predictions, forecasted_predictions = network(frames.to(device))
    #         forecasted_progress = torch.ones_like(progress)
    #         forecasted_progress[:, :-DELTA_T] = progress[:, DELTA_T:]

    #         loss = l2_criterion(predictions, progress.to(device))
    #         forecast_loss = l2_criterion(forecasted_predictions, forecasted_progress.to(device))

    #         test_loss += loss.item()
    #         test_forecast_loss += forecast_loss.item()
    #         test_count += lengths.sum().item()

    #         if i % 10 == 0:
                # action_labels = testset.get_action_labels(video_names[0])
                # for j, label in enumerate(action_labels):
                #     plt.axvspan(j-0.5, j+0.5, facecolor=['r', 'g', 'b', 'c', 'm', 'y'][label], alpha=0.2, zorder=-1)
    #             plt.plot(predictions.detach().cpu().squeeze(), label='predicted progress')
    #             forecasted_predictions = forecasted_predictions.detach().cpu().squeeze().tolist()
    #             plt.plot([i+DELTA_T for i,_ in enumerate(forecasted_predictions)], forecasted_predictions, label='forecasted progress')
    #             plt.plot(progress.detach().cpu().squeeze(), label='progress')

    #             plt.legend(loc='best')
    #             plt.title(f'Sample {video_names[0]} - Δt={DELTA_T}')
    #             plt.savefig(f'./plots/{epoch}/sample_{i}.png')
    #             plt.clf()

    #     print(f'[{epoch:03d} train] l2 loss {(train_loss / train_count):.4f} forecast loss  {(train_forecast_loss / train_count):.4f}')
    #     print(f'[{epoch:03d} test]  l2 loss {(test_loss / test_count):.4f} forecast loss  {(test_forecast_loss / train_count):.4f}')
if __name__ == '__main__':
    main()
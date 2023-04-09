import argparse
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
import os

from datasets import ProgressDataset
from networks import SequentialLSTM


# util functions

def get_device(device: str) -> torch.device:
    if torch.cuda.is_available() and device == 'cuda':
        return torch.device('cuda')
    return torch.device('cpu')


def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.uniform_(m.bias, a=0, b=0)
    elif isinstance(m, nn.LSTM):
        nn.init.xavier_uniform_(m.weight_hh_l0)
        nn.init.xavier_uniform_(m.weight_ih_l0)
        nn.init.uniform_(m.bias_ih_l0, a=0, b=0)
        nn.init.uniform_(m.bias_hh_l0, a=0, b=0)


def collate_fn(batch):
    video_names, embeddings, progress = zip(*batch)
    lengths = torch.Tensor([sample.shape[0] for sample in embeddings])

    padded_embeddings = pad_sequence(embeddings, batch_first=True)
    padded_progress = pad_sequence(progress, batch_first=True)

    return video_names, padded_embeddings, padded_progress, lengths

# init functions

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # experiment
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_root', type=str,
                        default='/home/frans/Datasets/')
    # network
    parser.add_argument('--network', type=str, default='sequential_lstm')
    parser.add_argument('--data_embedding_size', type=int, default=15)
    parser.add_argument('--forecasting_hidden_size', type=int, default=7)
    parser.add_argument('--lstm_hidden_size', type=int, default=1)
    # wandb config
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable Weights & Biases')
    parser.add_argument('--wandb_group', type=str, default='default',
                        help='Uses to group together results in Weights & Biases')
    # datasets
    parser.add_argument('--train_set', type=str, default='toy')
    parser.add_argument('--test_set', type=str, default='toy')
    parser.add_argument('--data_type', type=str, default='pooled/small')
    # training
    parser.add_argument('--iterations', type=int, default=1_500)
    parser.add_argument('--learning_rate', type=float, default=3e-3)
    parser.add_argument('--losses', nargs='+', default='progress')
    parser.add_argument('--augmentations', nargs='+', default='')
    # testing
    parser.add_argument('--test_every', type=int, default=100)
    # forecasting
    parser.add_argument('--delta_t', type=int, default=10)

    return parser.parse_args()


def get_network(args: argparse.Namespace, device: torch.device) -> nn.Module:
    if args.network == 'sequential_lstm':
        return SequentialLSTM(
            args.data_embedding_size,
            args.forecasting_hidden_size,
            args.lstm_hidden_size,
            device
        )


def init(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not args.no_wandb:
        wandb.init(
            project='mscfransdeboer',
            config={
                'seed': args.seed,
                'network': args.network,
                'group': args.wandb_group,
                'train_set': args.train_set,
                'test_set': args.test_set,
                'data_type': args.data_type,
                'iterations': args.iterations,
                'learning_rate': args.learning_rate,
                'losses': args.losses,
                'augmentations': args.augmentations,
                'delta_t': args.delta_t,
            }
        )

# wandb functions

def get_empty_result() -> dict:
    return {
        'l1_loss': 0.0,
        'l2_loss': 0.0,
        'l1_forecast_loss': 0.0,
        'l2_forecast_loss': 0.0,
        'l2_embedding_loss': 0.0,
        'count': 0
    }


def update_result(result: dict, batch_result: dict) -> None:
    for key in result:
        if key in batch_result:
            result[key] += batch_result[key]


def wandb_log(result: dict, iteration: int, prefix: str) -> None:
    wandb.log({
        f'{prefix}_l1_loss': result['l1_loss'] / result['count'],
        f'{prefix}_l2_loss': result['l2_loss'] / result['count'],

        f'{prefix}_l1_forecast_loss': result['l1_forecast_loss'] / result['count'],
        f'{prefix}_l2_forecast_loss': result['l2_forecast_loss'] / result['count'],

        f'{prefix}_l2_embedding_loss': result['l2_embedding_loss'] / result['count'],

        'count': result['count'],
        'iteration': iteration,
    })


def train(batch, network, args, device, optimizer=None):
    # criterions
    l1_criterion = nn.L1Loss(reduction='sum')
    l2_criterion = nn.MSELoss(reduction='sum')
    # batch data
    video_names, embeddings, progress, lengths = batch
    embeddings = embeddings.to(device)
    progress = progress.to(device)
    forecasted_progress = torch.ones_like(progress, device=device)
    forecasted_progress[:, :-args.delta_t] = progress[:, args.delta_t]
    # forward pass
    predictions, forecasted_predictions, forecasted_embeddings = network(
        embeddings)
    # loss calculations
    l1_loss = l1_criterion(predictions, progress)
    l2_loss = l2_criterion(predictions, progress)
    l1_forecast_loss = l1_criterion(
        forecasted_predictions, forecasted_progress)
    l2_forecast_loss = l2_criterion(
        forecasted_predictions, forecasted_progress)
    l2_embedding_loss = l2_criterion(
        forecasted_embeddings[:, :-args.delta_t, :], embeddings[:, args.delta_t:, :])
    # optimizer
    if optimizer:
        optimizer.zero_grad()
        loss = torch.zeros(1, device=device)
        if 'progress' in args.losses:
            loss += l2_loss
        if 'forecast' in args.losses:
            loss += l2_forecast_loss
        if 'embedding' in args.losses:
            loss += l2_embedding_loss
        loss.backward()
        optimizer.step()

    return {
        'video_names': video_names,
        'predictions': predictions.cpu().detach(),
        'forecasted_predictions': forecasted_predictions.cpu().detach(),
        'progress': progress.cpu().detach(),
        'forecasted_progress': forecasted_progress.cpu().detach(),
        'l1_loss': l1_loss.item(),
        'l2_loss': l2_loss.item(),
        'l1_forecast_loss': l1_forecast_loss.item(),
        'l2_forecast_loss': l2_forecast_loss.item(),
        'l2_embedding_loss': l2_embedding_loss.item(),
        'count': lengths.sum().item()
    }


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    init(args)

    train_root = os.path.join(args.data_root, args.train_set)
    test_root = os.path.join(args.data_root, args.test_set)
    # TODO: Sample transform
    trainset = ProgressDataset(
        train_root, args.data_type, 'splitfiles/trainlist01.txt')
    testset = ProgressDataset(
        test_root, args.data_type, 'splitfiles/testlist01.txt')
    trainloader = DataLoader(trainset, batch_size=1,
                             num_workers=4, shuffle=True, collate_fn=collate_fn)
    testloader = DataLoader(testset, batch_size=1,
                            num_workers=4, shuffle=False, collate_fn=collate_fn)

    progressnet = get_network(args, device).to(device)
    progressnet.apply(init_weights)

    optimizer = optim.Adam(progressnet.parameters(), lr=args.learning_rate)

    iteration = 0
    train_result = get_empty_result()
    test_result = get_empty_result()
    done = False
    while not done:
        for batch in trainloader:
            # train step
            batch_result = train(batch, progressnet, args,
                                 device, optimizer=optimizer)
            if not args.no_wandb:
                wandb_log(batch_result, iteration, 'train')
            update_result(train_result, batch_result)

            # test iteration
            if iteration % args.test_every == 0:
                for batch in testloader:
                    batch_result = train(batch, progressnet, args, device)
                    update_result(test_result, batch_result)

                if not args.no_wandb:
                    wandb_log(test_result, iteration, 'test')
                    wandb_log(train_result, iteration, 'avg_train')
                train_result = get_empty_result()
                test_result = get_empty_result()

            # update iteration & check if done
            iteration += 1
            if iteration > args.iterations:
                done = True
                break

    wandb.finish()


if __name__ == '__main__':
    main()

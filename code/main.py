from torch import nn, optim
from torch.utils.data import DataLoader
import argparse
import wandb
import os

from networks import Linear
from datasets import FeatureDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--data_root", type=str, default="/home/frans/Datasets")
    parser.add_argument('--dataset', type=str, default='cholec80')
    parser.add_argument("--data_type", type=str, default="features/i3d_embeddings")
    parser.add_argument("--train_split", type=str, default="t12_0.txt")
    parser.add_argument("--test_split", type=str, default="e_0.txt")

    parser.add_argument('--wandb_name', type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=float, default=0.1)
    parser.add_argument('--lr_decay_every', type=int, default=503 * 30)

    parser.add_argument('--feature_dim', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=20)

    parser.add_argument('--iterations', type=int, default=503 * 60)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--test_every', type=int, default=200)

    return parser.parse_args()

def train(network, batch, device, optimizer=None):
    l2_loss = nn.MSELoss(reduction='sum')
    name, data, progress = batch
    data = data.to(device)
    B, _ = data.shape
    predicted_progress = network(data).reshape(B)
    progress = progress.to(device)
    if optimizer:
        optimizer.zero_grad()
        l2_loss(predicted_progress, progress).backward()
        optimizer.step()
    
    return {
        'l2_loss': l2_loss(predicted_progress, progress),
        'count': B,
    }


def main():
    args = parse_args()
    data_root = os.path.join(args.data_root, args.dataset)

    wandb.init(
        project='ute',
        name=args.wandb_name,
        config={
            'dataset': args.dataset,
            'data_type': args.data_type,
            'train_split': args.train_split,
            'test_split': args.test_split,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'lr_decay': args.lr_decay,
            'lr_decay_every': args.lr_decay_every,
            'feature_dim': args.feature_dim,
            'embed_dim': args.embed_dim,
            'iterations': args.iterations
        }
    )

    # TODO: Get Dataset
    # - Feature Dataset (flat / no flat)
    # - Image Dataset (flat / no flat)
    train_set = FeatureDataset(data_root, args.data_type, args.train_split, flat=True)
    test_set = FeatureDataset(data_root, args.data_type, args.test_split, flat=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=4, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=4, shuffle=False)

    network = Linear(args.feature_dim, args.embed_dim)
    optimizer = optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_every, args.lr_decay)

    print('--- Network ---')
    print(network)
    print('--- Datasets ---')
    print(f'Train {len(train_set)} ({len(train_loader)})')
    print(f'Test {len(test_set)} ({len(test_loader)})')



if __name__ == "__main__":
    main()

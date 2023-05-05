import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import os
from typing import List
import argparse
import statistics

def load_splitfile(path: str):
    with open(path) as f:
        names = f.readlines()
    return [name.strip() for name in names]

activity_lengths = {
    'coffee': 587,
    'cereals': 704,
    'tea': 716,
    'milk': 949,
    'juice': 1491,
    'sandwich': 1535,
    'scrambledegg': 3117,
    'friedegg': 3120,
    'salat': 3429,
    'pancake': 5969
}

class FeatureDataset(Dataset):
    def __init__(
        self,
        root: str,
        data_dir: str,
        splitfile: str,
        transform=None,
    ) -> None:
        super().__init__()
        self.transform = transform
        split_path = os.path.join(root, "splitfiles", splitfile)
        splitnames = load_splitfile(split_path)
        self.data = self._get_data(os.path.join(root, data_dir), splitnames)

    @staticmethod
    def _get_data(
        root: str,
        splitnames: List[str],
    ) -> List[str]:
        data = []
        lengths = []
        for video_name in splitnames:
            path = os.path.join(root, f"{video_name}.txt")
            with open(path) as f:
                video_data = f.readlines()

            video_data = torch.FloatTensor(
                [list(map(float, row.split(" "))) for row in video_data]
            )
            S, F = video_data.shape
            video_data = torch.arange(0, S).reshape(S, 1).repeat(1, F) / 5969
            lengths.append(S)
            progress = torch.arange(1, S + 1) / S
            for i, (embedding, p) in enumerate(zip(video_data, progress)):
                data.append((f"{video_name}_{i}", embedding, p))
        print(statistics.mean(lengths), statistics.stdev(lengths))
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class MLP(nn.Module):
    def __init__(self, feature_dim: int = 64, embed_dim: int = 20) -> None:
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, embed_dim * 2)
        self.fc2 = nn.Linear(2 * embed_dim, embed_dim)
        self.fc_last = nn.Linear(embed_dim, 1)
        # self._init_weights()

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc_last(x)
        return x

    def embedded(self, x):
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--activity', type=str, default='coffee')
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)

    return parser.parse_args()

def train(network, batch, device, optimizer=None):
    criterion = nn.MSELoss()
    l2 = nn.MSELoss(reduction='sum')
    name, features, progress = batch
    predicted_progress = network(features.to(device)).reshape(features.shape[0])
    progress = progress.to(device)
    if optimizer:
        optimizer.zero_grad()
        criterion(predicted_progress, progress).backward()
        optimizer.step()

    return {
        'l2_loss': l2(predicted_progress, progress).item(),
        'count': features.shape[0],
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parse_args()

    torch.random.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    network = MLP().to(device)

    trainset = FeatureDataset('/home/frans/Datasets/breakfast', 'features/dense_trajectories', f'all_{args.activity}.txt')
    testset = FeatureDataset('/home/frans/Datasets/breakfast', 'features/dense_trajectories', f'all_{args.activity}.txt')
    trainloader = DataLoader(trainset, batch_size=256, num_workers=4, shuffle=True)
    testloader = DataLoader(testset, batch_size=256, num_workers=4, shuffle=False)

    optimizer = optim.Adam(network.parameters(), lr=args.lr)

    iterations = 10_000
    iteration = 0
    done = False

    train_result = {'l2_loss': 0, 'count': 0}
    test_result = {'l2_loss': 0, 'count': 0}
    while not done:
        for batch in trainloader:
            batch_result = train(network, batch, device, optimizer=optimizer)
            train_result['l2_loss'] += batch_result['l2_loss']
            train_result['count'] += batch_result['count']

            if iteration % 100 == 0 and iteration > 0:
                # print(f'train {iteration} - {train_result["l2_loss"] / train_result["count"]}')
                train_result = {'l2_loss': 0, 'count': 0}
            if iteration % 1000 == 0 and iteration > 0:
                network.eval()
                with torch.no_grad():
                    for batch in testloader:
                        batch_result = train(network, batch, device)
                        test_result['l2_loss'] += batch_result['l2_loss']
                        test_result['count'] += batch_result['count']
                # print(f'test {iteration} - {test_result["l2_loss"] / test_result["count"]}')
                test_result = {'l2_loss': 0, 'count': 0}
                network.train()
            
            iteration += 1
            if iteration > iterations:
                done = True
                break

    network.eval()
    test_result = {'l2_loss': 0, 'count': 0}
    with torch.no_grad():
        for batch in testloader:
            batch_result = train(network, batch, device)
            test_result['l2_loss'] += batch_result['l2_loss']
            test_result['count'] += batch_result['count']
    print(f'test {iteration} - {test_result["l2_loss"] / test_result["count"]}')
    test_result = {'l2_loss': 0, 'count': 0}
    network.train()

if __name__ == "__main__":
    main()

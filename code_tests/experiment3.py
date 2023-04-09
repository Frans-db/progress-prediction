import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import wandb

from dataset import FileDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
COLORS = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'lime', 'darkblue']

def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.uniform_(m.bias, a=0, b=0)
    elif isinstance(m, nn.LSTM):
        nn.init.xavier_uniform_(m.weight_hh_l0)
        nn.init.xavier_uniform_(m.weight_ih_l0)
        nn.init.uniform_(m.bias_ih_l0, a=0, b=0)
        nn.init.uniform_(m.bias_hh_l0, a=0, b=0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--hidden_size', type=int)
    parser.add_argument('--forecasting_embedding_size', type=int)
    parser.add_argument('--delta_t', type=int)

    parser.add_argument('--group', type=str)

    return parser.parse_args()

class Network(nn.Module):
    def __init__(self, data_embedding_size: int, forecasting_embedding_size: int, hidden_size: int):
        super(Network, self).__init__()
        self.hidden_size = hidden_size
        self.hn, self.cn = (torch.zeros(1, 1, device=device), torch.zeros(1, self.hidden_size, device=device))

        self.forecasting_head = nn.Sequential(
            nn.Linear(15, forecasting_embedding_size),
            nn.Linear(forecasting_embedding_size, 15)
        )
        self.lstm = nn.LSTM(data_embedding_size, self.hidden_size, 1, batch_first=True, proj_size=1 if self.hidden_size > 1 else 0)

    def forward(self, x):
        B, S, _ = x.shape
        
        forecasts = self.forecasting_head(x.reshape(-1, 15)).reshape(1, -1, 15)

        progress = torch.zeros(B, S, 1, device=device)
        forecasted_progress = torch.zeros_like(progress, device=device)
        hn, cn = (self.hn, self.cn)
        for i in range(S):
            item = x[:, i, :]
            forecasted_item = forecasts[:, i, :]

            item_progress, (hn, cn) = self.lstm(item, (hn, cn))
            forecasted_item_progress, _ = self.lstm(forecasted_item, (hn, cn))

            progress[:, i, :] = item_progress
            forecasted_progress[:, i, :] = forecasted_item_progress

        return progress.reshape(B, S), forecasted_progress.reshape(B, S), forecasts

def main():
    args = parse_args()

    wandb.init(
        project='experiments',
        config={
            'group': args.group
        }
    )

    trainset = FileDataset(f'/home/frans/Datasets/toy_speed', 'pooled/small', 'trainlist01.txt')
    trainloader = DataLoader(trainset, batch_size=1, num_workers=4)
    
    testset = FileDataset(f'/home/frans/Datasets/toy_speedier', 'pooled/small', 'testlist01.txt')
    testloader = DataLoader(testset, batch_size=1, num_workers=4)

    network = Network(trainset.embedding_size, args.forecasting_embedding_size, args.hidden_size).to(device)
    network.apply(init_weights)
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(network.parameters(), lr=3e-3)

    iteration = 0
    for epoch in range(args.epochs):
        train_loss, test_loss = 0.0, 0.0
        train_count, test_count = 0, 0
        for _, data, progress in trainloader:
            B, S, _ = data.shape # B,S,15

            predictions, forecasted_predictions, forecasted_embeddings = network(data.to(device))
            optimizer.zero_grad()
            loss = criterion(predictions, progress.to(device))
            forecasted_loss = criterion(forecasted_predictions[:, :-args.delta_t], progress.to(device)[:, args.delta_t:])
            wandb.log({
                'iteration': iteration,
                'train_loss': (loss / S).item()
            })
            ((loss) / S).backward()
            optimizer.step()

            train_loss += loss.item()
            train_count += S
            iteration += 1

        for i, (video_names, data, progress) in enumerate(testloader):
            B, S, _ = data.shape
            video_name = video_names[0]

            predictions, forecasted_predictions, forecasted_embeddings = network(data.to(device))
            loss = criterion(predictions, progress.to(device))

            test_loss += loss.item()
            test_count += S

            # if epoch == args.epochs - 1:
            #     action_labels = testset.get_action_labels(video_name)
            #     for j, label in enumerate(action_labels):
            #         plt.axvspan(j-0.5, j+0.5, facecolor=COLORS[label], alpha=0.2, zorder=-1)
            #     xs = list(range(S))
            #     forecasted_xs = list(map(lambda x: x + args.delta_t, xs))
            #     plt.plot(xs, progress.reshape(S).detach(), label='progress')
            #     plt.plot(xs, predictions.cpu().reshape(S).detach(), label='predicted')
            #     plt.plot(forecasted_xs, forecasted_predictions.cpu().reshape(S).detach(), label='forecasted')
            #     plt.title(f'Sample {i}')
            #     plt.savefig(f'./plots/{i}.png')
            #     plt.clf()

        wandb.log({
            'iteration': iteration,
            'test_loss': test_loss / test_count
        })
        print(f'[{epoch:03d}] train {(train_loss / train_count):.4f} test {(test_loss / test_count):.4f}')




if __name__ == '__main__':
    main()
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse


from dataset import FileDataset

"""
Using just LSTM memory to predict progress. 
Network input is spatial pyramid pooling of video frames
"""

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

    return parser.parse_args()

class Network(nn.Module):
    def __init__(self, embedding_size: int, hidden_size: int):
        super(Network, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(embedding_size, self.hidden_size, 1, batch_first=True, proj_size=1 if self.hidden_size > 1 else 0)

    def forward(self, x):
        B, S, _ = x.shape
        x,_ = self.lstm(x)
        return x

def main():
    args = parse_args()

    trainset = FileDataset(f'/home/frans/Datasets/{args.dataset}', 'pooled/small', 'trainlist01.txt')
    trainloader = DataLoader(trainset, batch_size=1, num_workers=4)
    
    testset = FileDataset(f'/home/frans/Datasets/{args.dataset}', 'pooled/small', 'testlist01.txt')
    testloader = DataLoader(testset, batch_size=1, num_workers=4)

    network = Network(trainset.embedding_size, args.hidden_size).to(device)
    network.apply(init_weights)
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(network.parameters(), lr=3e-3)

    for epoch in range(args.epochs):
        train_loss, test_loss = 0.0, 0.0
        train_count, test_count = 0, 0
        for _, data, progress in trainloader:
            B, S, _ = data.shape # B,S,15

            predictions = network(data.to(device)).reshape(B, S)
            optimizer.zero_grad()
            loss = criterion(predictions, progress.to(device))
            (loss / S).backward()
            optimizer.step()

            train_loss += loss.item()
            train_count += S

        for i, (video_names, data, progress) in enumerate(testloader):
            B, S, _ = data.shape
            video_name = video_names[0]

            predictions = network(data.to(device)).reshape(B, S)
            loss = criterion(predictions, progress.to(device))

            test_loss += loss.item()
            test_count += S

            if epoch == args.epochs - 1:
                action_labels = testset.get_action_labels(video_name)
                for j, label in enumerate(action_labels):
                    plt.axvspan(j-0.5, j+0.5, facecolor=COLORS[label], alpha=0.2, zorder=-1)
                plt.plot(progress.reshape(S).detach(), label='progress')
                plt.plot(predictions.cpu().reshape(S).detach(), label='predicted')
                plt.title(f'Sample {i}')
                plt.savefig(f'./plots/{i}.png')
                plt.clf()

        print(f'[{epoch:03d}] train {(train_loss / train_count):.4f} test {(test_loss / test_count):.4f}')




if __name__ == '__main__':
    main()
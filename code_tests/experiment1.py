import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

"""
Using just LSTM memory to predict progress. 
Network input is all ones
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 10_000

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_steps', type=int)

    return parser.parse_args()

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.lstm = nn.LSTM(1, 1, 1)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        return x

def main():
    args = parse_args()

    outputs = torch.arange(1, args.num_steps + 1) / args.num_steps
    inputs = torch.ones_like(outputs)

    inputs = inputs.unsqueeze(dim=1).to(device)
    outputs = outputs.unsqueeze(dim=1).to(device)
    network = Network().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(network.parameters(), lr=1e-3)

    for epoch in tqdm(range(epochs)):
        predictions = network(inputs)
        optimizer.zero_grad()
        loss = criterion(predictions, outputs)
        loss.backward()
        optimizer.step()

    plt.plot(outputs.squeeze().detach().cpu(), label='progress')
    plt.plot(predictions.squeeze().detach().cpu(), label='predictions')
    plt.title(f'Num Steps {args.num_steps}')
    plt.legend(loc='best')
    plt.savefig(f'./plot_{args.num_steps}.png')

if __name__ == '__main__':
    main()
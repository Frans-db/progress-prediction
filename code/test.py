import torch
import torch.nn as nn

def main():
    steps = 1000
    target = torch.FloatTensor([1])
    l1_criterion = nn.L1Loss()
    l2_criterion = nn.MSELoss()
    for i in range(steps):
        prediction = torch.FloatTensor([(i + 1) / steps])
        l1_loss = l1_criterion(prediction, target).item()
        l2_loss = l2_criterion(prediction, target).item()
        print(i, l2_loss, l1_loss)

if __name__ == '__main__':
    main()
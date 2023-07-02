import torch.nn as nn
import torch

l1_criterion = nn.L1Loss()
l2_criterion = nn.MSELoss()
iterations = 10000
target = torch.FloatTensor([1])
for i in range(iterations):
    prediction = torch.FloatTensor([(i+1) / iterations])
    l1_loss = l1_criterion(prediction * 100, target * 100)
    l2_loss = l2_criterion(prediction, target)
    print(l1_loss.item(), l2_loss.item())
    if l2_loss.item() < 0.045:
        break

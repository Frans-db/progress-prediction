import torch
import torch.nn as nn


loss = nn.L1Loss()
inputs = torch.FloatTensor([1, 1, 1])
outputs = torch.FloatTensor([2, 2, 2])
print(loss(inputs, outputs))
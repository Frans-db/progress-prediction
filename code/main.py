import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.optim as optim
import math

from dataset import VideoFrameDataset, ProgressDataset
from dataset.transforms import ImglistToTensor, SwapDimensions
from networks import S3D, S2D

def bo_loss(p, p_hat):
    l,u = 0,1
    m = (l + u) / 2
    r = (u - l) / 2

    real_part = (p - m) * (r * math.sqrt(2))
    predicted_part = (p_hat - m) * (r * math.sqrt(2))
    
    energies = torch.clamp(torch.square(real_part) + torch.square(predicted_part), min=1)
    losses = energies * (p - p_hat).abs()
    return losses.mean()

    
def main():
    num_segments = 1
    frames_per_segment = 10
    num_frames = num_segments * frames_per_segment

    dataset = ProgressDataset(
            './data/toy',
            num_segments=num_segments,
            frames_per_segment=frames_per_segment,
            transform=transforms.Compose([
                ImglistToTensor(),
                SwapDimensions()
            ]),
            )      

    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=1)

    net = S3D()
    criterion = bo_loss
    optimizer = optim.Adam(net.parameters())

    for epoch in range(3):
        epoch_loss = 0
        for i,(videos, labels) in enumerate(dataloader):
            num_samples = videos.shape[0]
            labels = torch.stack(labels).reshape(num_samples, num_frames).float()
            
            outputs = net(videos).float()

            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
        print(f'[{epoch:2d}] loss: {epoch_loss:.3f}')

    for i in range(20):
        video, labels = dataset[i]
        video = video.reshape(3, num_frames, 42, 42)
        predictions = net(video.unsqueeze(0)).squeeze()
        print(predictions)

        plt.plot(predictions.detach().numpy(), label='Predicted')
        plt.plot(labels, label='Actual')
        plt.title(f'Predicted vs Actul Completion Percentage - Video {i:5d}')
        plt.legend(loc='best')
        plt.show()
                
if __name__ == '__main__':
    main()
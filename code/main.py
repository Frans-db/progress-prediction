import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm

from dataset import VideoFrameDataset, ProgressDataset
from dataset.transforms import ImglistToTensor, SwapDimensions
from networks import S3D, S2D



def main():
    num_segments = 1
    frames_per_segment = 15
    num_frames = num_segments * frames_per_segment

    dataset = ProgressDataset(
            './data/toy_1',
            num_segments=num_segments,
            frames_per_segment=frames_per_segment,
            transform=transforms.Compose([
                ImglistToTensor(),
                SwapDimensions()
            ]),
            )         
    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=1)

    net = S3D()
    criterion = nn.MSELoss()
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
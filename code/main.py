import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm

from video_dataset import VideoFrameDataset, ImglistToTensor

class Network(nn.Module):
    """
    Very basic network to play around with the data
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(3, 16, 5, padding=(2, 0, 0), stride=(1, 2, 2)) # (90,38,38)
        self.conv2 = nn.Conv3d(16, 32, 5, padding=(2, 0, 0), stride=(1, 2, 2)) # (90,38,38)
        self.conv3 = nn.Conv3d(32, 64, (5, 8, 8), padding=(2, 0, 0)) # (90,38,38)
        self.conv4 = nn.Conv3d(64, 1, 1)

    def forward(self, x):
        num_samples, num_frames = x.shape[0], x.shape[2]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.reshape(num_samples, num_frames)
        return x
        
def main():
    dataset = VideoFrameDataset(
            './data/MNIST/toy', 
            './data/MNIST/annotations.txt',
            num_segments=1,
            frames_per_segment=90,
            imagefile_template='img_{:05d}.png',
            transform=transforms.Compose([
                ImglistToTensor(),
            ])
        )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=3,
        shuffle=True,
        num_workers=1
    )
    net = Network()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001)
    for epoch in range(10):
        epoch_loss = 0
        for i,(videos, labels) in tqdm(enumerate(dataloader)):
            num_samples = videos.shape[0]
            videos = videos.reshape(num_samples, 3, 90, 42, 42)
            labels = torch.stack(labels).reshape(num_samples, 90)
            
            outputs = net(videos)

            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            optimizer.step()
        print(f'[{epoch:2d}] loss: {epoch_loss:.3f}')

    for (video, labels) in dataset:
        video = video.reshape(3, 90, 42, 42)
        predictions = net(video.unsqueeze(0)).squeeze()
        print(predictions)
        xs = [i for i in range(90)]
        plt.plot(xs, predictions.detach().numpy())
        plt.plot(labels)
        plt.show()
        break
                
if __name__ == '__main__':
    main()
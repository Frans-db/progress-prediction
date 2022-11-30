import torch
import torch.nn as nn
import torch.nn.functional as F

class S3D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(3, 32, (1, 3, 3)) # (20,38,38)
        self.pool1 = nn.AvgPool3d((1, 2, 2))
        
        self.conv_2_1 = nn.Conv3d(32, 64, (1, 3, 3), stride=(1, 2, 2))
        self.conv_2_2 = nn.Conv3d(64, 64, (7, 1, 1), padding=(3, 0, 0))

        self.conv_3_1 = nn.Conv3d(64, 128, (1, 3, 3), stride=(1, 2, 2))
        self.conv_3_2 = nn.Conv3d(128, 128, (7, 1, 1), padding=(3, 0, 0))

        self.conv_4_1 = nn.Conv3d(128, 256, (1, 4, 4), stride=(1, 2, 2))
        self.conv_4_2 = nn.Conv3d(256, 256, (7, 1, 1), padding=(3, 0, 0))     

        self.conv5 = nn.Conv3d(256, 1, (1, 1, 1))

    def forward(self, x):
        num_samples, num_frames = x.shape[0], x.shape[2]

        x = self.conv1(x)
        x = self.pool1(x)
        x = F.relu(x)

        x = self.conv_2_1(x)
        x = self.conv_2_2(x)
        x = F.relu(x)

        x = self.conv_3_1(x)
        x = self.conv_3_2(x)
        x = F.relu(x)

        x = self.conv_4_1(x)
        x = self.conv_4_2(x)
        x = F.relu(x)

        x = self.conv5(x)
        x = torch.sigmoid(x)

        x = x.reshape(num_samples, num_frames)
        return x
    
import torch
import torch.nn.functional as F
from torch import nn


class Linear(nn.Module):
    def __init__(self, feature_dim: int, embed_dim: int, dropout_chance: float) -> None:
        super().__init__()
        self.embedding_dropout = nn.Dropout(p=dropout_chance)
        self.fc1 = nn.Linear(feature_dim, embed_dim * 2)
        self.fc2 = nn.Linear(2 * embed_dim, embed_dim)
        self.fc_last = nn.Linear(embed_dim, 1)
        # self._init_weights()

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.embedding_dropout(x)
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc_last(x)
        return x

    def embed(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

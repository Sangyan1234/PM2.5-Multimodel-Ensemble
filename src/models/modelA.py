import torch
import torch.nn as nn

class ModelA(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(16, 64, 3, padding=1)
        self.out = nn.Conv2d(64, 16, 1)

    def forward(self, x):
        x = x[:, -1]
        x = self.conv(x)
        return self.out(x)

import torch
import torch.nn as nn

class ModelB(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(16, 64, 3, padding=1)
        self.out = nn.Conv2d(64, 16, 1)

    def forward(self, x):
        x = x[:, -1]
        return self.out(self.conv(x))

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(oc, oc, 3, padding=1),
            nn.GELU()
        )

    def forward(self, x):
        return self.net(x)

class ModelA(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = ConvBlock(16, 64)
        self.out = nn.Conv2d(64, 16, 1)

    def forward(self, x, last_pm):
        x = x[:, -1]
        x = self.conv(x)
        return self.out(x)

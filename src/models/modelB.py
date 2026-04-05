import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvGRUCell(nn.Module):
    def __init__(self, in_ch, h_ch):
        super().__init__()
        self.hc = h_ch
        self.rz = nn.Conv2d(in_ch+h_ch, 2*h_ch, 3, padding=1)
        self.can = nn.Conv2d(in_ch+h_ch, h_ch, 3, padding=1)

    def forward(self, x, h):
        rz = torch.sigmoid(self.rz(torch.cat([x,h],1)))
        r, z = rz.chunk(2,1)
        can = torch.tanh(self.can(torch.cat([x, r*h],1)))
        return (1-z)*h + z*can

    def init(self, B, H, W, dev):
        return torch.zeros(B, self.hc, H, W, device=dev)


class ModelB(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Conv2d(16, 32, 3, padding=1)

        self.gru1 = ConvGRUCell(32, 32)
        self.gru2 = ConvGRUCell(32, 64)

        self.conv = nn.Sequential(
            nn.Conv2d(64,128,3,padding=1),
            nn.GELU(),
            nn.Conv2d(128,64,3,padding=1),
            nn.GELU()
        )

        self.out = nn.Conv2d(64,16,1)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, last_pm):
        B,T,F,H,W = x.shape
        h1 = self.gru1.init(B,H,W,x.device)
        h2 = self.gru2.init(B,H,W,x.device)

        for t in range(T):
            xt = self.inp(x[:,t])
            h1 = self.gru1(xt, h1)
            h2 = self.gru2(h1, h2)

        x = self.conv(h2)
        delta = self.out(x)

        return delta + self.alpha * last_pm.unsqueeze(1)

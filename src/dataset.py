import torch
from torch.utils.data import Dataset

class PM25Dataset(Dataset):
    def __init__(self, data, idx):
        self.data = data
        self.idx = idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        m, s = self.idx[i]
        arr = self.data[m]

        x = arr[s:s+10]
        y = arr[s+10:s+26, 0]

        return torch.tensor(x).float(), torch.tensor(y).float()

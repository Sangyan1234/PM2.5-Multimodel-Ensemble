import torch
from torch.utils.data import Dataset

class PM25Dataset(Dataset):
    def __init__(self, idx, data):
        self.idx = idx
        self.data = data

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        m, s = self.idx[i]
        arr = self.data[m]

        x = arr[s:s+10]
        lp = arr[s+9,0]
        y = arr[s+10:s+26,0]

        return (
            torch.tensor(x).float(),
            torch.tensor(y).float(),
            torch.tensor(lp).float()
        )

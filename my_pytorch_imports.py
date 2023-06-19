import torch
from torch.utils.data import Dataset
class DatasetFromNp(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = data
        self.target = target
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        # Extending 1channel image to be put to three channels
        x = torch.from_numpy(x)
        x = torch.unsqueeze(x, 0)
        x = x.expand(3, -1, -1)
        y = self.target[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.target)

import torch
from torch.utils.data import Dataset


class APIDataset(Dataset):
    def __init__(self, x, normal_key_api_sequence, abnormal_key_api_sequence, y):
        self.x = torch.tensor(x, dtype=torch.long)
        self.normal_key_api_sequence = torch.tensor(normal_key_api_sequence, dtype=torch.long)
        self.abnormal_key_api_sequence = torch.tensor(abnormal_key_api_sequence, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.normal_key_api_sequence[idx], self.abnormal_key_api_sequence[idx], self.y[idx]

#  1. Implement Custom Dataset

import torch
import numpy as np
from base import BaseDataLoader
from torch.utils.data import Dataset

class FunctionsDataset(Dataset):
    def __init__(self, n_samples=100, function='linear'):
        self.n_samples = n_samples
        self.function = function

        # generate values of x randomly between 0 and 2pi
        self.x = np.random.uniform(0, 2 * np.pi, self.n_samples)

        # generate values of y based on function type and noise (epsilon between -1 and 1)
        epsilon = np.random.uniform(-1, 1, self.n_samples)
        if self.function == 'linear':
            self.y = 1.5 * self.x + 0.3 + epsilon
        elif self.function == 'quadratic':
            self.y = 2 * (self.x ** 2) + 0.5 * self.x + 0.3 + epsilon
        elif self.function == 'harmonic':
            self.y = 0.5 * (self.x ** 2) + 5 * np.sin(self.x) + 3 * np.cos(3 * self.x) + 2 + epsilon
        else:
            raise ValueError(f"Unknown function type: {self.function}. Function options only are linear, quadratic, and harmonic")

        # normalize x and y
        self.x = (self.x - np.mean(self.x)) / np.std(self.x)
        self.y = (self.y - np.mean(self.y)) / np.std(self.y)

        self.x = torch.tensor(self.x, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(self.y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class FunctionDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, function='linear', n_samples=100):
        self.dataset = FunctionDataset(n_samples=n_samples, function=function)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
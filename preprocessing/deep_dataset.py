import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from preprocessing.physics_transform import apply_physics_transform


class IMSDataset(Dataset):
    def __init__(self, data_dir, target_length=1024):
        self.data_dir = Path(data_dir)
        self.target_length = target_length

        self.windows, self.labels = self._prepare_data()

    def _load_windows(self):
        files = sorted(self.data_dir.iterdir())
        windows = []

        for file in files:
            data = np.loadtxt(file)
            windows.append(data)

        windows = np.stack(windows)
        return windows

    def _downsample(self, windows):
        factor = windows.shape[1] // self.target_length
        return windows[:, ::factor, :]

    def _normalize(self, windows):
        mean = windows.mean(axis=(0, 1), keepdims=True)
        std = windows.std(axis=(0, 1), keepdims=True) + 1e-8
        return (windows - mean) / std

    def _create_labels(self, n):
        labels = np.linspace(0, 1, n)
        stages = np.zeros_like(labels, dtype=int)

        stages[(labels >= 0.25) & (labels < 0.50)] = 1
        stages[(labels >= 0.50) & (labels < 0.75)] = 2
        stages[labels >= 0.75] = 3

        return stages

    def _prepare_data(self):
        windows = self._load_windows()
        windows = self._downsample(windows)

        windows = apply_physics_transform(windows)
        windows = self._normalize(windows)

        labels = self._create_labels(windows.shape[0])

        return windows.astype(np.float32), labels.astype(np.int64)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = torch.tensor(self.windows[idx])
        y = torch.tensor(self.labels[idx])

        x = x.permute(1, 0)

        return x, y

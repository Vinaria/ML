import numpy as np
import typing


class MinMaxScaler:
    def fit(self, data: np.ndarray) -> None:
        self.mins = data.min(axis=0)
        self.maxs = data.max(axis=0)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mins) / (self.maxs - self.mins)


class StandardScaler:
    def fit(self, data: np.ndarray) -> None:
        self.means = data.mean(axis=0)
        self.stds = data.std(axis=0)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.means) / self.stds

import numpy as np
from nanograd.tensor import Tensor


class DataLoader:
    """Simple data loader with batching and shuffling.

    Args:
        X: input data as numpy array
        y: labels as numpy array
        batch_size: number of samples per batch
        shuffle: whether to shuffle at the start of each epoch
    """

    def __init__(self, X, y, batch_size=32, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = len(X)

    def __iter__(self):
        indices = np.arange(self.n)
        if self.shuffle:
            np.random.shuffle(indices)

        for start in range(0, self.n, self.batch_size):
            idx = indices[start:start + self.batch_size]
            yield Tensor(self.X[idx]), self.y[idx]

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size

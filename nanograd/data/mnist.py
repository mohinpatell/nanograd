"""MNIST dataset loading.

Downloads the dataset from the web if not already cached locally.
Returns numpy arrays ready for training.
"""

import os
import gzip
import struct
import urllib.request
import numpy as np

MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _download(filename, cache_dir):
    filepath = os.path.join(cache_dir, filename)
    if not os.path.exists(filepath):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(MNIST_URL + filename, filepath)
    return filepath


def _read_images(filepath):
    with gzip.open(filepath, 'rb') as f:
        magic, n, rows, cols = struct.unpack('>IIII', f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(n, rows * cols).astype(np.float32) / 255.0


def _read_labels(filepath):
    with gzip.open(filepath, 'rb') as f:
        magic, n = struct.unpack('>II', f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)


def load_mnist(cache_dir=None):
    """Load MNIST dataset.

    Returns:
        (X_train, y_train, X_test, y_test)
        X: float32 arrays of shape (N, 784), normalized to [0, 1]
        y: uint8 arrays of shape (N,), values 0-9
    """
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".nanograd", "data")
    os.makedirs(cache_dir, exist_ok=True)

    X_train = _read_images(_download(FILES["train_images"], cache_dir))
    y_train = _read_labels(_download(FILES["train_labels"], cache_dir))
    X_test = _read_images(_download(FILES["test_images"], cache_dir))
    y_test = _read_labels(_download(FILES["test_labels"], cache_dir))

    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_mnist()
    print(f"Train: {X_train.shape}, {y_train.shape}")
    print(f"Test:  {X_test.shape}, {y_test.shape}")
    print(f"Labels: {np.unique(y_train)}")
    print(f"Pixel range: [{X_train.min()}, {X_train.max()}]")

"""Train a CNN on MNIST.

Architecture:
  Conv2d(1, 8, 3, padding=1) -> ReLU -> MaxPool(2)    # 28x28 -> 14x14
  Conv2d(8, 16, 3, padding=1) -> ReLU -> MaxPool(2)   # 14x14 -> 7x7
  Flatten -> Linear(16*7*7, 64) -> ReLU -> Linear(64, 10)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
from nanograd import Tensor
from nanograd.nn import (
    Conv2d, Linear, ReLU, MaxPool2d, Flatten,
    Sequential, CrossEntropyLoss, Dropout,
)
from nanograd.optim import Adam
from nanograd.data import DataLoader, load_mnist

np.random.seed(42)

print("Loading MNIST...")
X_train, y_train, X_test, y_test = load_mnist()

# reshape to (N, 1, 28, 28) for conv layers
X_train = X_train.reshape(-1, 1, 28, 28)
X_test = X_test.reshape(-1, 1, 28, 28)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# model
model = Sequential(
    Conv2d(1, 8, kernel_size=3, padding=1),   # -> (N, 8, 28, 28)
    ReLU(),
    MaxPool2d(2),                              # -> (N, 8, 14, 14)
    Conv2d(8, 16, kernel_size=3, padding=1),   # -> (N, 16, 14, 14)
    ReLU(),
    MaxPool2d(2),                              # -> (N, 16, 7, 7)
    Flatten(),                                 # -> (N, 784)
    Linear(16 * 7 * 7, 64),
    ReLU(),
    Dropout(0.3),
    Linear(64, 10),
)

criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-3)

num_params = sum(p.data.size for p in model.parameters())
print(f"Model parameters: {num_params:,}")
print()

# use a subset for faster training (CNN is slow in pure numpy)
TRAIN_SIZE = 10000
X_sub = X_train[:TRAIN_SIZE]
y_sub = y_train[:TRAIN_SIZE]

EPOCHS = 5
BATCH_SIZE = 32

train_loader = DataLoader(X_sub, y_sub, batch_size=BATCH_SIZE)
test_loader = DataLoader(X_test[:2000], y_test[:2000], batch_size=64, shuffle=False)

for epoch in range(EPOCHS):
    model.set_training(True)
    total_loss = 0
    correct = 0
    total = 0
    t0 = time.time()

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()

        logits = model(batch_X)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.data.item() * len(batch_y)
        preds = logits.data.argmax(axis=1)
        correct += (preds == batch_y).sum()
        total += len(batch_y)

    train_loss = total_loss / total
    train_acc = correct / total
    elapsed = time.time() - t0

    # eval
    model.set_training(False)
    test_correct = 0
    test_total = 0
    for batch_X, batch_y in test_loader:
        logits = model(batch_X)
        preds = logits.data.argmax(axis=1)
        test_correct += (preds == batch_y).sum()
        test_total += len(batch_y)

    test_acc = test_correct / test_total

    print(f"Epoch {epoch+1}/{EPOCHS} ({elapsed:.1f}s) | "
          f"train loss: {train_loss:.4f}, train acc: {train_acc:.4f} | "
          f"test acc: {test_acc:.4f}")

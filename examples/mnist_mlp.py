"""Train a simple MLP on MNIST.

Network: 784 -> 128 -> 64 -> 10
Loss: CrossEntropy
Optimizer: Adam with lr=1e-3
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
from nanograd import Tensor
from nanograd.nn import Linear, ReLU, Sequential, CrossEntropyLoss
from nanograd.optim import SGD, Adam
from nanograd.data import DataLoader, load_mnist

np.random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam'])
parser.add_argument('--lr', type=float, default=None)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--batch-size', type=int, default=64)
args = parser.parse_args()

# load data
print("Loading MNIST...")
X_train, y_train, X_test, y_test = load_mnist()
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# model
model = Sequential(
    Linear(784, 128),
    ReLU(),
    Linear(128, 64),
    ReLU(),
    Linear(64, 10),
)

criterion = CrossEntropyLoss()
if args.optimizer == 'adam':
    lr = args.lr or 1e-3
    optimizer = Adam(model.parameters(), lr=lr)
else:
    lr = args.lr or 0.1
    optimizer = SGD(model.parameters(), lr=lr)

num_params = sum(p.data.size for p in model.parameters())
print(f"Model parameters: {num_params:,}")
print(f"Optimizer: {args.optimizer} (lr={lr})")
print()

# training
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size

train_loader = DataLoader(X_train, y_train, batch_size=BATCH_SIZE)
test_loader = DataLoader(X_test, y_test, batch_size=256, shuffle=False)

for epoch in range(EPOCHS):
    # train
    total_loss = 0
    correct = 0
    total = 0

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

    # eval
    test_correct = 0
    test_total = 0
    for batch_X, batch_y in test_loader:
        logits = model(batch_X)
        preds = logits.data.argmax(axis=1)
        test_correct += (preds == batch_y).sum()
        test_total += len(batch_y)

    test_acc = test_correct / test_total

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"train loss: {train_loss:.4f}, train acc: {train_acc:.4f} | "
          f"test acc: {test_acc:.4f}")

"""XOR sanity check — needs a hidden layer since it's not linearly separable."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from nanograd import Tensor
from nanograd.nn import Linear, ReLU, Sequential, MSELoss
from nanograd.optim import SGD

np.random.seed(42)

# XOR dataset
X = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32))
y = Tensor(np.array([[0], [1], [1], [0]], dtype=np.float32))

# model: 2 -> 8 -> 1
model = Sequential(
    Linear(2, 8),
    ReLU(),
    Linear(8, 1),
)

criterion = MSELoss()
optimizer = SGD(model.parameters(), lr=0.1)

print(model)
print(f"Parameters: {sum(p.data.size for p in model.parameters())}")
print()

# train
for epoch in range(1000):
    optimizer.zero_grad()

    pred = model(X)
    loss = criterion(pred, y)
    loss.backward()

    optimizer.step()

    if epoch % 100 == 0:
        print(f"epoch {epoch:4d} | loss: {loss.data.item():.6f}")

# final predictions
print("\nFinal predictions:")
pred = model(X)
for i in range(4):
    inp = X.data[i]
    out = pred.data[i, 0]
    target = y.data[i, 0]
    print(f"  {inp} -> {out:.4f} (target: {target})")

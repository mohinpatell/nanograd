# nanograd

Autograd engine and neural network library from scratch, using only numpy.

I kept using PyTorch at work and realized I had no idea what actually happens when you call `.backward()`. So I decided to build it myself from the ground up. Started with scalar values and the chain rule, and kept going until I had something that could train CNNs on MNIST.

It's slow (pure Python + numpy, no CUDA), but every gradient is correct, and I actually understand all of it now.

## What's in here

**Autograd engine**
- Tensor class with automatic differentiation
- Supports add, mul, matmul, exp, log, relu, sum, mean, reshape, transpose
- Broadcasting with correct gradient accumulation
- Topological sort for backprop ordering

**NN layers**
- `Linear` - fully connected, He init
- `Conv2d` - 2D convolution via im2col
- `MaxPool2d`, `Flatten`, `ReLU`, `Dropout`, `Sequential`

**Losses**: `MSELoss`, `CrossEntropyLoss` (log-sum-exp for numerical stability)

**Optimizers**: `SGD`, `Adam`

## Results

| Model | Dataset | Test Accuracy |
|-------|---------|--------------|
| MLP (784-128-64-10) | MNIST | 98.1% |
| CNN (Conv-Conv-FC) | MNIST (10k subset) | 96.8% |

## Quick start

```python
from nanograd import Tensor
from nanograd.nn import Linear, ReLU, Sequential, CrossEntropyLoss
from nanograd.optim import Adam

model = Sequential(
    Linear(784, 128),
    ReLU(),
    Linear(128, 10),
)

optimizer = Adam(model.parameters(), lr=1e-3)
criterion = CrossEntropyLoss()

logits = model(x_batch)
loss = criterion(logits, y_batch)
loss.backward()
optimizer.step()
```

## Things I learned building this

- How computational graphs track operations for autodiff
- Broadcasting makes gradient accumulation surprisingly tricky
- He initialization actually matters a lot for ReLU networks
- Convolutions are really just matrix multiplies (im2col)
- Log-sum-exp trick is essential for stable softmax
- Adam converges way faster than vanilla SGD because of per-parameter learning rates

## Project structure

```
nanograd/
  tensor.py          # core tensor with autograd
  nn/
    module.py        # Module base class, Parameter
    linear.py        # fully connected layer
    conv.py          # Conv2d with im2col
    pooling.py       # MaxPool2d, Flatten
    activations.py   # ReLU
    loss.py          # MSE, CrossEntropy
    dropout.py       # dropout regularization
    sequential.py    # Sequential container
  optim/
    sgd.py           # SGD
    adam.py           # Adam
  data/
    dataloader.py    # batching + shuffling
    mnist.py         # MNIST download + loading
```

## Running examples

```bash
pip install numpy

# XOR (sanity check)
python examples/xor.py

# MNIST MLP
python examples/mnist_mlp.py --optimizer adam --epochs 10

# MNIST CNN
python examples/mnist_cnn.py
```

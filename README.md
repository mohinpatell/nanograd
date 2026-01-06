# nanograd

A minimal autograd engine and neural network library built from scratch.

I built this to understand how frameworks like PyTorch actually compute gradients. Every operation, every backward pass, every optimizer — written from the ground up with just numpy.

What started as a scalar autograd engine grew into a tensor-based framework that can train CNNs on MNIST. It's not fast (it's pure Python/numpy), but it's correct — and I understand every line of it.

## What's implemented

**Autograd engine**
- Tensor class with automatic differentiation
- Operations: add, mul, matmul, exp, log, relu, sum, mean, reshape, transpose
- Broadcasting support with correct gradient accumulation
- Topological sort for backpropagation ordering

**Neural network layers**
- `Linear` — fully connected layer with He initialization
- `Conv2d` — 2D convolution using im2col
- `MaxPool2d` — max pooling with gradient routing
- `ReLU` — rectified linear unit
- `Dropout` — inverted dropout with train/eval mode
- `Sequential` — layer container
- `Flatten` — reshape for conv-to-linear transition

**Loss functions**
- `MSELoss` — mean squared error
- `CrossEntropyLoss` — with log-sum-exp trick for numerical stability

**Optimizers**
- `SGD` — stochastic gradient descent
- `Adam` — adaptive learning rates with momentum + bias correction

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

# define a model
model = Sequential(
    Linear(784, 128),
    ReLU(),
    Linear(128, 10),
)

# training loop
optimizer = Adam(model.parameters(), lr=1e-3)
criterion = CrossEntropyLoss()

logits = model(x_batch)
loss = criterion(logits, y_batch)
loss.backward()
optimizer.step()
```

## What I learned

- How computational graphs track operations for automatic differentiation
- The chain rule applied to tensor operations (broadcasting makes gradients tricky!)
- Why He initialization matters for ReLU networks
- im2col: how convolutions are really just matrix multiplies
- The log-sum-exp trick for numerically stable softmax
- Why Adam converges faster than SGD (adaptive per-parameter learning rates)

## Project structure

```
nanograd/
  tensor.py          # Core tensor with autograd
  nn/
    module.py        # Module base class, Parameter
    linear.py        # Fully connected layer
    conv.py          # Conv2d with im2col
    pooling.py       # MaxPool2d, Flatten
    activations.py   # ReLU
    loss.py          # MSE, CrossEntropy
    dropout.py       # Dropout regularization
    sequential.py    # Sequential container
  optim/
    sgd.py           # SGD optimizer
    adam.py           # Adam optimizer
  data/
    dataloader.py    # Batching and shuffling
    mnist.py         # MNIST download and loading
```

## Running the examples

```bash
pip install numpy

# XOR (sanity check)
python examples/xor.py

# MNIST MLP
python examples/mnist_mlp.py --optimizer adam --epochs 10

# MNIST CNN
python examples/mnist_cnn.py
```

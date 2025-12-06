# nanograd

Building a tiny autograd engine from scratch to understand how backpropagation actually works under the hood.

I've been using PyTorch at work and realized I don't really understand what happens when you call `.backward()`. This repo is my attempt to build it from the ground up — starting with scalar values and the chain rule, eventually working up to tensors and real neural networks.

## Goals
- Implement automatic differentiation from scratch
- Build neural network layers (Linear, Conv2d, etc.)
- Train on real datasets (MNIST, CIFAR-10)
- No PyTorch/TensorFlow dependency for the core engine (just numpy)

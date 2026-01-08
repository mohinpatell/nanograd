import numpy as np
from nanograd.tensor import Tensor
from nanograd.nn.module import Module


class Dropout(Module):
    """Inverted dropout. No-op during eval."""

    def __init__(self, p=0.5):
        self.p = p
        self.training = True

    def forward(self, x):
        if not self.training:
            return x

        mask = (np.random.rand(*x.shape) > self.p).astype(np.float32)
        scale = 1.0 / (1.0 - self.p)

        out = Tensor(x.data * mask * scale, (x,), 'dropout',
                     requires_grad=x.requires_grad)

        def _backward():
            if x.requires_grad:
                x.grad += mask * scale * out.grad
        out._backward = _backward

        return out

    def __repr__(self):
        return f"Dropout(p={self.p})"

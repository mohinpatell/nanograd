import numpy as np
from nanograd.nn.module import Module, Parameter


class Linear(Module):
    """Fully connected layer: y = x @ W^T + b

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
    """

    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

        # He initialization (good for ReLU networks)
        # scale = sqrt(2 / fan_in)
        scale = np.sqrt(2.0 / in_features)
        self.weight = Parameter(np.random.randn(out_features, in_features).astype(np.float32) * scale)
        self.bias = Parameter(np.zeros(out_features, dtype=np.float32))

    def forward(self, x):
        # x: (batch, in_features) -> (batch, out_features)
        return x @ self.weight.T + self.bias

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features})"

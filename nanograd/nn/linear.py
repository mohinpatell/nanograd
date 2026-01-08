import numpy as np
from nanograd.nn.module import Module, Parameter


class Linear(Module):

    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

        scale = np.sqrt(2.0 / in_features)  # He init
        self.weight = Parameter(np.random.randn(out_features, in_features).astype(np.float32) * scale)
        self.bias = Parameter(np.zeros(out_features, dtype=np.float32))

    def forward(self, x):
        # x: (batch, in_features) -> (batch, out_features)
        return x @ self.weight.T + self.bias

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features})"

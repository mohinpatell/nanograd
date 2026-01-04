import numpy as np
from nanograd.tensor import Tensor
from nanograd.nn.module import Module


class MaxPool2d(Module):
    """2D max pooling.

    Args:
        kernel_size: size of the pooling window
        stride: stride of the pooling (defaults to kernel_size)
    """

    def __init__(self, kernel_size=2, stride=None):
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size

    def forward(self, x):
        N, C, H, W = x.shape
        kh = kw = self.kernel_size
        s = self.stride
        out_h = (H - kh) // s + 1
        out_w = (W - kw) // s + 1

        # extract all pooling windows
        out_data = np.zeros((N, C, out_h, out_w), dtype=np.float32)
        mask = np.zeros_like(x.data)

        for i in range(out_h):
            for j in range(out_w):
                window = x.data[:, :, i*s:i*s+kh, j*s:j*s+kw]
                out_data[:, :, i, j] = window.max(axis=(2, 3))

                # save argmax positions for backward
                max_val = out_data[:, :, i, j][:, :, None, None]
                mask[:, :, i*s:i*s+kh, j*s:j*s+kw] += (window == max_val).astype(np.float32)

        out = Tensor(out_data, (x,), 'maxpool2d',
                     requires_grad=x.requires_grad)

        def _backward():
            if x.requires_grad:
                # distribute gradient only to the max positions
                dx = np.zeros_like(x.data)
                for i in range(out_h):
                    for j in range(out_w):
                        window_mask = mask[:, :, i*s:i*s+kh, j*s:j*s+kw]
                        # normalize mask (in case of ties)
                        window_sum = window_mask.sum(axis=(2, 3), keepdims=True)
                        window_sum = np.maximum(window_sum, 1)  # avoid div by zero
                        dx[:, :, i*s:i*s+kh, j*s:j*s+kw] += (
                            window_mask / window_sum * out.grad[:, :, i:i+1, j:j+1]
                        )
                x.grad += dx
        out._backward = _backward

        return out

    def __repr__(self):
        return f"MaxPool2d(kernel_size={self.kernel_size}, stride={self.stride})"


class Flatten(Module):
    """Flatten all dims except the batch dimension."""

    def forward(self, x):
        batch_size = x.shape[0]
        flat_size = int(np.prod(x.shape[1:]))
        return x.reshape(batch_size, flat_size)

    def __repr__(self):
        return "Flatten()"

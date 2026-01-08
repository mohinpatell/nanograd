import numpy as np
from nanograd.tensor import Tensor
from nanograd.nn.module import Module, Parameter


def _im2col(x, kh, kw, stride=1, padding=0):
    """Patches to columns for conv as matmul."""
    N, C, H, W = x.shape

    if padding > 0:
        x = np.pad(x, ((0,0), (0,0), (padding, padding), (padding, padding)))
        _, _, H, W = x.shape

    out_h = (H - kh) // stride + 1
    out_w = (W - kw) // stride + 1

    cols = np.zeros((N, C, kh, kw, out_h, out_w), dtype=x.dtype)
    for i in range(kh):
        i_max = i + stride * out_h
        for j in range(kw):
            j_max = j + stride * out_w
            cols[:, :, i, j, :, :] = x[:, :, i:i_max:stride, j:j_max:stride]

    # reshape to (N * out_h * out_w, C * kh * kw)
    cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return cols, out_h, out_w


def _col2im(cols, x_shape, kh, kw, stride=1, padding=0):
    """Inverse of im2col."""
    N, C, H, W = x_shape
    H_padded = H + 2 * padding
    W_padded = W + 2 * padding
    out_h = (H_padded - kh) // stride + 1
    out_w = (W_padded - kw) // stride + 1

    cols = cols.reshape(N, out_h, out_w, C, kh, kw).transpose(0, 3, 4, 5, 1, 2)

    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    for i in range(kh):
        i_max = i + stride * out_h
        for j in range(kw):
            j_max = j + stride * out_w
            x_padded[:, :, i:i_max:stride, j:j_max:stride] += cols[:, :, i, j, :, :]

    if padding > 0:
        return x_padded[:, :, padding:-padding, padding:-padding]
    return x_padded


class Conv2d(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        fan_in = in_channels * kernel_size * kernel_size
        scale = np.sqrt(2.0 / fan_in)
        self.weight = Parameter(
            np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float32) * scale
        )
        self.bias = Parameter(np.zeros(out_channels, dtype=np.float32))

    def forward(self, x):
        N = x.shape[0]
        kh = kw = self.kernel_size

        cols, out_h, out_w = _im2col(x.data, kh, kw, self.stride, self.padding)
        W_flat = self.weight.data.reshape(self.out_channels, -1)

        out_data = cols @ W_flat.T + self.bias.data
        out_data = out_data.reshape(N, out_h, out_w, self.out_channels).transpose(0, 3, 1, 2)

        out = Tensor(out_data, (x, self.weight, self.bias), 'conv2d',
                     requires_grad=x.requires_grad or self.weight.requires_grad)

        _cols = cols
        _x_shape = x.shape
        _self = self

        def _backward():
            dout = out.grad.transpose(0, 2, 3, 1).reshape(-1, _self.out_channels)

            if _self.bias.requires_grad:
                _self.bias.grad += dout.sum(axis=0)

            if _self.weight.requires_grad:
                dW = _cols.T @ dout
                _self.weight.grad += dW.T.reshape(_self.weight.shape)

            if x.requires_grad:
                dcols = dout @ W_flat  # (N*oh*ow, C_in*kh*kw)
                x.grad += _col2im(dcols, _x_shape, kh, kw, _self.stride, _self.padding)

        out._backward = _backward

        return out

    def __repr__(self):
        return (f"Conv2d({self.in_channels}, {self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})")

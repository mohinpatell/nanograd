import numpy as np


class Tensor:
    """Tensor with autograd, backed by numpy."""

    def __init__(self, data, _children=(), _op='', requires_grad=False):
        if isinstance(data, (int, float, np.floating, np.integer)):
            data = np.array(data, dtype=np.float32)
        elif isinstance(data, list):
            data = np.array(data, dtype=np.float32)
        elif isinstance(data, np.ndarray):
            data = data.astype(np.float32) if data.dtype != np.float32 else data
        else:
            raise TypeError(f"unsupported data type: {type(data)}")

        self.data = data
        self.grad = None
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

        if requires_grad:
            self.grad = np.zeros_like(self.data)

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    def _make_tensor(self, other):
        if isinstance(other, Tensor):
            return other
        return Tensor(other)

    @staticmethod
    def _unbroadcast(grad, shape):
        """Sum out broadcast dims so grad matches original shape."""
        while len(grad.shape) > len(shape):
            grad = grad.sum(axis=0)
        for i, s in enumerate(shape):
            if s == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    def __add__(self, other):
        other = self._make_tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+',
                     requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += Tensor._unbroadcast(out.grad, self.shape)
            if other.requires_grad:
                other.grad += Tensor._unbroadcast(out.grad, other.shape)
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = self._make_tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*',
                     requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += Tensor._unbroadcast(other.data * out.grad, self.shape)
            if other.requires_grad:
                other.grad += Tensor._unbroadcast(self.data * out.grad, other.shape)
        out._backward = _backward

        return out

    def matmul(self, other):
        other = self._make_tensor(other)
        out = Tensor(self.data @ other.data, (self, other), '@',
                     requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad
        out._backward = _backward

        return out

    def __matmul__(self, other):
        return self.matmul(other)

    def sum(self):
        out = Tensor(self.data.sum(), (self,), 'sum',
                     requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += np.ones_like(self.data) * out.grad
        out._backward = _backward

        return out

    def mean(self):
        n = self.data.size
        out = Tensor(self.data.mean(), (self,), 'mean',
                     requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += np.ones_like(self.data) * (out.grad / n)
        out._backward = _backward

        return out

    def exp(self):
        out = Tensor(np.exp(self.data), (self,), 'exp',
                     requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,), 'log',
                     requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0), (self,), 'relu',
                     requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += (self.data > 0).astype(np.float32) * out.grad
        out._backward = _backward

        return out

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(shape), (self,), 'reshape',
                     requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += out.grad.reshape(self.shape)
        out._backward = _backward

        return out

    def transpose(self, ax0=-2, ax1=-1):
        axes = list(range(len(self.shape)))
        axes[ax0], axes[ax1] = axes[ax1], axes[ax0]
        out = Tensor(self.data.transpose(axes), (self,), 'T',
                     requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += out.grad.transpose(axes)
        out._backward = _backward

        return out

    @property
    def T(self):
        return self.transpose()

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return Tensor(other) + (-self)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def backward(self):
        assert self.data.size == 1, "backward can only be called on scalar tensors"

        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

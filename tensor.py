import numpy as np


class Tensor:
    """A tensor with automatic differentiation support, backed by numpy."""

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
        """Sum out dimensions that were broadcast so grad matches original shape.

        When numpy broadcasts (3,4) + (4,) -> (3,4), the grad coming back is (3,4)
        but we need (4,) for the second operand. So we sum along axis 0.
        """
        # first handle the case where shape has fewer dims (was prepended with 1s)
        while len(grad.shape) > len(shape):
            grad = grad.sum(axis=0)
        # then handle dims that were 1 and got broadcast
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
        """Matrix multiplication: self @ other."""
        other = self._make_tensor(other)
        out = Tensor(self.data @ other.data, (self, other), '@',
                     requires_grad=self.requires_grad or other.requires_grad)

        # for Y = A @ B:
        #   dL/dA = dL/dY @ B^T
        #   dL/dB = A^T @ dL/dY
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


if __name__ == '__main__':
    import torch

    # test relu
    a = Tensor(np.array([-1.0, 2.0, -3.0, 4.0]), requires_grad=True)
    c = a.relu().sum()
    c.backward()

    at = torch.tensor([-1.0, 2.0, -3.0, 4.0], requires_grad=True)
    ct = at.relu().sum()
    ct.backward()
    print(f"relu grad match: {np.allclose(a.grad, at.grad.numpy())}")
    print(f"  ours: {a.grad}, pytorch: {at.grad.numpy()}")

    # test mean
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
    c = a.mean()
    c.backward()

    at = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    ct = at.mean()
    ct.backward()
    print(f"\nmean grad match: {np.allclose(a.grad, at.grad.numpy())}")
    print(f"  ours: {a.grad}")

    # test sub
    a = Tensor([3.0, 4.0], requires_grad=True)
    b = Tensor([1.0, 2.0], requires_grad=True)
    c = (a - b).sum()
    c.backward()
    print(f"\nsub grads: a={a.grad} (expect [1,1]), b={b.grad} (expect [-1,-1])")

    # combined expression: relu(Wx + b).mean()
    W = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
    x = Tensor(np.random.randn(4, 2).astype(np.float32), requires_grad=True)
    b = Tensor(np.random.randn(3, 1).astype(np.float32), requires_grad=True)
    y = (W @ x + b).relu().mean()
    y.backward()

    Wt = torch.tensor(W.data, requires_grad=True)
    xt = torch.tensor(x.data, requires_grad=True)
    bt = torch.tensor(b.data, requires_grad=True)
    yt = (Wt @ xt + bt).relu().mean()
    yt.backward()

    print(f"\nrelu(Wx + b).mean() grads:")
    print(f"  W match: {np.allclose(W.grad, Wt.grad.numpy(), atol=1e-6)}")
    print(f"  x match: {np.allclose(x.grad, xt.grad.numpy(), atol=1e-6)}")
    print(f"  b match: {np.allclose(b.grad, bt.grad.numpy(), atol=1e-6)}")

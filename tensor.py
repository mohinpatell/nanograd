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

    # Test broadcasting: (3,4) + (4,)
    a = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
    b = Tensor(np.random.randn(4).astype(np.float32), requires_grad=True)
    c = (a + b).sum()
    c.backward()

    at = torch.tensor(a.data, requires_grad=True)
    bt = torch.tensor(b.data, requires_grad=True)
    ct = (at + bt).sum()
    ct.backward()

    print("Broadcasting add (3,4) + (4,):")
    print(f"  a.grad match: {np.allclose(a.grad, at.grad.numpy())}")
    print(f"  b.grad match: {np.allclose(b.grad, bt.grad.numpy())}")
    print(f"  b.grad shape: {b.grad.shape} (should be (4,))")

    # Test broadcasting: (3,4) * (1,4)
    a = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
    b = Tensor(np.random.randn(1, 4).astype(np.float32), requires_grad=True)
    c = (a * b).sum()
    c.backward()

    at = torch.tensor(a.data, requires_grad=True)
    bt = torch.tensor(b.data, requires_grad=True)
    ct = (at * bt).sum()
    ct.backward()

    print("\nBroadcasting mul (3,4) * (1,4):")
    print(f"  a.grad match: {np.allclose(a.grad, at.grad.numpy())}")
    print(f"  b.grad match: {np.allclose(b.grad, bt.grad.numpy())}")
    print(f"  b.grad shape: {b.grad.shape} (should be (1,4))")

    # Test scalar broadcast: (3,) + scalar
    a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    c = (a + 5.0).sum()
    c.backward()
    at = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    ct = (at + 5.0).sum()
    ct.backward()
    print(f"\nScalar broadcast: a.grad match: {np.allclose(a.grad, at.grad.numpy())}")

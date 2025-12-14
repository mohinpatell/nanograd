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

    def __add__(self, other):
        other = self._make_tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+',
                     requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = self._make_tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*',
                     requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += other.data * out.grad
            if other.requires_grad:
                other.grad += self.data * out.grad
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
    # test add with sum
    a = Tensor([1, 2, 3], requires_grad=True)
    b = Tensor([4, 5, 6], requires_grad=True)
    c = (a + b).sum()
    c.backward()
    print(f"(a + b).sum() grads: a={a.grad}, b={b.grad}")

    # test mul with sum
    a = Tensor([2, 3], requires_grad=True)
    b = Tensor([4, 5], requires_grad=True)
    c = (a * b).sum()
    c.backward()
    print(f"(a * b).sum() grads: a={a.grad} (expect [4,5]), b={b.grad} (expect [2,3])")

    # test matmul
    # A is (2,3), B is (3,2), result is (2,2)
    A = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    B = Tensor([[1, 0], [0, 1], [1, 1]], requires_grad=True)
    C = (A @ B).sum()
    C.backward()
    print(f"\nA @ B matmul:")
    print(f"A.grad =\n{A.grad}")
    print(f"B.grad =\n{B.grad}")

    # verify against pytorch
    import torch
    At = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, requires_grad=True)
    Bt = torch.tensor([[1, 0], [0, 1], [1, 1]], dtype=torch.float32, requires_grad=True)
    Ct = (At @ Bt).sum()
    Ct.backward()
    print(f"\nPyTorch A.grad =\n{At.grad.numpy()}")
    print(f"PyTorch B.grad =\n{Bt.grad.numpy()}")
    print(f"\nMatch: A={np.allclose(A.grad, At.grad.numpy())}, B={np.allclose(B.grad, Bt.grad.numpy())}")

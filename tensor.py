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
    # element-wise add
    a = Tensor([1, 2, 3], requires_grad=True)
    b = Tensor([4, 5, 6], requires_grad=True)
    c = a + b
    s = Tensor(c.data.sum(), (c,), 'sum', requires_grad=True)

    # manually wire up the sum backward for now
    def _sum_backward():
        c.grad = np.ones_like(c.data) * s.grad if c.requires_grad else None
    s._backward = _sum_backward
    c.requires_grad = True
    c.grad = np.zeros_like(c.data)

    s.backward()
    print(f"a + b = {c.data}")
    print(f"a.grad = {a.grad}")  # should be [1, 1, 1]
    print(f"b.grad = {b.grad}")  # should be [1, 1, 1]

    # element-wise mul
    a = Tensor([2, 3], requires_grad=True)
    b = Tensor([4, 5], requires_grad=True)
    c = a * b  # [8, 15]
    s = Tensor(c.data.sum(), (c,), 'sum', requires_grad=True)
    def _sum_backward2():
        c.grad = np.ones_like(c.data) * s.grad
    s._backward = _sum_backward2
    c.requires_grad = True
    c.grad = np.zeros_like(c.data)

    s.backward()
    print(f"\na * b = {c.data}")
    print(f"a.grad = {a.grad}")  # should be [4, 5]
    print(f"b.grad = {b.grad}")  # should be [2, 3]

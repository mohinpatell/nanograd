import numpy as np


class Tensor:
    """A tensor with automatic differentiation support, backed by numpy."""

    def __init__(self, data, _children=(), _op='', requires_grad=False):
        if isinstance(data, (int, float)):
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
    # basic creation tests
    a = Tensor([1, 2, 3])
    print(f"a = {a}")
    print(f"a.shape = {a.shape}")

    b = Tensor([[1, 2], [3, 4]], requires_grad=True)
    print(f"b = {b}")
    print(f"b.grad = {b.grad}")

    c = Tensor(5.0)
    print(f"scalar: {c}, shape: {c.shape}")

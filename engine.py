class Value:
    """Stores a single scalar value and its gradient."""

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        # d(a+b)/da = 1, d(a+b)/db = 1
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        # d(a*b)/da = b, d(a*b)/db = a
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def backward(self):
        # topological sort — we need to call _backward in reverse order
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one node at a time and apply the chain rule
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other


if __name__ == '__main__':
    # simple test: f = a*b + c, compute gradients
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    f = a * b + c  # f = 2*(-3) + 10 = 4

    f.backward()

    # df/da = b = -3, df/db = a = 2, df/dc = 1
    print(f"f = {f}")
    print(f"a.grad = {a.grad} (expected -3.0)")
    print(f"b.grad = {b.grad} (expected 2.0)")
    print(f"c.grad = {c.grad} (expected 1.0)")

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

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), f'**{other}')

        # d(a^n)/da = n * a^(n-1)
        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        # a / b = a * b^(-1)
        return self * other ** -1

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return Value(other) + (-self)

    def __rtruediv__(self, other):
        return Value(other) * self ** -1


if __name__ == '__main__':
    # test all the new ops
    a = Value(4.0)
    b = Value(2.0)

    # subtraction
    c = a - b  # 2.0
    c.backward()
    print(f"a - b = {c.data} | a.grad={a.grad}, b.grad={b.grad}")

    # reset grads
    a.grad = 0; b.grad = 0

    # division
    d = a / b  # 2.0
    d.backward()
    print(f"a / b = {d.data} | a.grad={a.grad}, b.grad={b.grad}")

    # reset
    a.grad = 0

    # power
    e = a ** 3  # 64.0
    e.backward()
    print(f"a ** 3 = {e.data} | a.grad={a.grad} (expected 48.0)")

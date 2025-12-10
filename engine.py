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

    def exp(self):
        import math
        out = Value(math.exp(self.data), (self,), 'exp')

        # d(e^a)/da = e^a
        def _backward():
            self.grad += out.data * out.grad  # nice: derivative of exp is just exp
        out._backward = _backward

        return out

    def tanh(self):
        import math
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        # d(tanh(a))/da = 1 - tanh(a)^2
        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0, (self,), 'relu')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

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
    import math

    # test exp
    a = Value(2.0)
    b = a.exp()
    b.backward()
    print(f"exp(2) = {b.data:.4f} (expected {math.exp(2):.4f}) | a.grad = {a.grad:.4f}")

    # test tanh
    a = Value(0.5)
    b = a.tanh()
    b.backward()
    print(f"tanh(0.5) = {b.data:.4f} (expected {math.tanh(0.5):.4f}) | a.grad = {a.grad:.4f}")

    # test relu
    a = Value(-2.0)
    b = a.relu()
    b.backward()
    print(f"relu(-2) = {b.data} | a.grad = {a.grad} (expected 0)")

    a = Value(3.0)
    b = a.relu()
    b.backward()
    print(f"relu(3) = {b.data} | a.grad = {a.grad} (expected 1)")

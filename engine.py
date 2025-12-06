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
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        return out

    def __repr__(self):
        return f"Value(data={self.data})"

    # so we can do 2 + Value(3) not just Value(3) + 2
    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other


if __name__ == '__main__':
    a = Value(2.0)
    b = Value(-3.0)
    c = a * b + Value(10.0)
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"c = a*b + 10 = {c}")
    print(f"c._prev = {c._prev}")
    print(f"c._op = {c._op}")

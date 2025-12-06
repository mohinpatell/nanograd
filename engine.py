class Value:
    """Stores a single scalar value and its gradient."""

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        # internal variables for autograd graph
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data})"


# quick test
if __name__ == '__main__':
    a = Value(2.0)
    b = Value(3.0)
    print(a, b)

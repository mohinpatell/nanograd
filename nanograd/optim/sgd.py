class SGD:
    """Stochastic Gradient Descent optimizer.

    Args:
        params: iterable of Parameters to optimize
        lr: learning rate
    """

    def __init__(self, params, lr=0.01):
        self.params = list(params)
        self.lr = lr

    def step(self):
        for p in self.params:
            p.data -= self.lr * p.grad

    def zero_grad(self):
        for p in self.params:
            p.grad *= 0

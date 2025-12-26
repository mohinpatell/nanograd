import numpy as np


class Adam:
    """Adam optimizer — adaptive learning rates with momentum.

    Combines the ideas of momentum (exponential moving average of gradients)
    and RMSprop (exponential moving average of squared gradients).

    Args:
        params: iterable of Parameters to optimize
        lr: learning rate (default: 1e-3)
        betas: coefficients for computing running averages (default: (0.9, 0.999))
        eps: term added for numerical stability (default: 1e-8)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0

        # initialize moment estimates
        self.m = [np.zeros_like(p.data) for p in self.params]  # first moment (mean)
        self.v = [np.zeros_like(p.data) for p in self.params]  # second moment (variance)

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            # update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            # update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (p.grad ** 2)

            # bias correction — important in early steps when estimates are biased toward 0
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # update parameters
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.grad *= 0

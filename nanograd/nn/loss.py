import numpy as np
from nanograd.tensor import Tensor
from nanograd.nn.module import Module


class MSELoss(Module):

    def forward(self, pred, target):
        diff = pred + target * -1  # pred - target
        return (diff * diff).mean()


class CrossEntropyLoss(Module):
    """Takes raw logits (batch, classes) and integer targets (batch,)."""

    def forward(self, logits, targets):
        batch_size = logits.shape[0]

        # shift by max so exp() doesn't overflow
        max_vals = logits.data.max(axis=1, keepdims=True)
        shifted = logits + Tensor(-max_vals)
        exp_shifted = shifted.exp()

        # manual sum along class dim (need to keep dims for broadcast)
        sum_exp = Tensor(exp_shifted.data.sum(axis=1, keepdims=True),
                         (exp_shifted,), 'sum_axis1',
                         requires_grad=exp_shifted.requires_grad)

        def _sum_backward():
            if exp_shifted.requires_grad:
                exp_shifted.grad += np.ones_like(exp_shifted.data) * sum_exp.grad
        sum_exp._backward = _sum_backward
        if exp_shifted.requires_grad:
            sum_exp.requires_grad = True
            sum_exp.grad = np.zeros_like(sum_exp.data)

        log_sum = sum_exp.log()
        log_probs = shifted + log_sum * -1  # (batch, classes)

        if isinstance(targets, Tensor):
            targets_np = targets.data.astype(int)
        else:
            targets_np = np.array(targets, dtype=int)

        correct_log_probs = log_probs.data[np.arange(batch_size), targets_np]
        nll = Tensor(-correct_log_probs.mean(), (log_probs,), 'nll',
                     requires_grad=log_probs.requires_grad)

        def _nll_backward():
            if log_probs.requires_grad:
                grad = np.zeros_like(log_probs.data)
                grad[np.arange(batch_size), targets_np] = -1.0 / batch_size
                log_probs.grad += grad * nll.grad
        nll._backward = _nll_backward
        if log_probs.requires_grad:
            nll.requires_grad = True
            nll.grad = np.zeros_like(nll.data)

        return nll

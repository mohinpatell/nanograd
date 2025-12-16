"""Numerical gradient checking to verify our autograd is correct.

The idea: for each parameter element, perturb it by a small epsilon,
compute the function twice (f(x+eps) and f(x-eps)), and estimate the
gradient as (f(x+eps) - f(x-eps)) / (2*eps). Then compare against
our analytical gradients from backprop.
"""

import numpy as np
from tensor import Tensor


def numerical_gradient(f, tensor, eps=1e-4):
    """Compute numerical gradient of scalar function f with respect to tensor."""
    grad = np.zeros_like(tensor.data)
    it = np.nditer(tensor.data, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        old_val = tensor.data[idx]

        tensor.data[idx] = old_val + eps
        fxph = f().data.item()

        tensor.data[idx] = old_val - eps
        fxmh = f().data.item()

        grad[idx] = (fxph - fxmh) / (2 * eps)
        tensor.data[idx] = old_val
        it.iternext()

    return grad


def check_gradient(f, tensors, eps=1e-4, tol=1e-3):
    """Check analytical gradients against numerical gradients.

    Args:
        f: function that takes no args, uses the tensors via closure, returns scalar Tensor
        tensors: list of Tensors to check gradients for
        eps: perturbation size
        tol: tolerance for gradient match

    Returns True if all gradients match within tolerance.
    """
    # compute analytical gradients
    result = f()
    result.backward()

    # save analytical grads BEFORE numerical check (which calls f() and zeros them)
    analytical_grads = [t.grad.copy() for t in tensors]

    all_ok = True
    for i, t in enumerate(tensors):
        num_grad = numerical_gradient(f, t, eps)
        ana_grad = analytical_grads[i]

        if not np.allclose(num_grad, ana_grad, atol=tol, rtol=tol):
            print(f"FAIL tensor {i}: max diff = {np.max(np.abs(num_grad - ana_grad))}")
            print(f"  numerical: {num_grad}")
            print(f"  analytical: {ana_grad}")
            all_ok = False
        else:
            print(f"OK tensor {i} (shape {t.shape}): max diff = {np.max(np.abs(num_grad - ana_grad)):.2e}")

    return all_ok


if __name__ == '__main__':
    np.random.seed(42)

    # test 1: simple matmul + relu
    print("=== matmul + relu ===")
    W = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
    x = Tensor(np.random.randn(4, 2).astype(np.float32), requires_grad=True)
    b = Tensor(np.random.randn(3, 1).astype(np.float32), requires_grad=True)

    def f1():
        W.grad = np.zeros_like(W.data)
        x.grad = np.zeros_like(x.data)
        b.grad = np.zeros_like(b.data)
        return (W @ x + b).relu().mean()

    assert check_gradient(f1, [W, x, b])

    # test 2: deeper expression
    print("\n=== W2 @ relu(W1 @ x + b1) + b2 ===")
    W1 = Tensor(np.random.randn(5, 3).astype(np.float32), requires_grad=True)
    b1 = Tensor(np.random.randn(5, 1).astype(np.float32), requires_grad=True)
    W2 = Tensor(np.random.randn(2, 5).astype(np.float32), requires_grad=True)
    b2 = Tensor(np.random.randn(2, 1).astype(np.float32), requires_grad=True)
    x = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)

    def f2():
        for t in [W1, b1, W2, b2, x]:
            t.grad = np.zeros_like(t.data)
        h = (W1 @ x + b1).relu()
        return (W2 @ h + b2).mean()

    assert check_gradient(f2, [W1, b1, W2, b2, x])

    print("\nAll gradient checks passed!")

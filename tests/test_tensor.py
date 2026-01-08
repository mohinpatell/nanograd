"""Test tensor autograd against PyTorch."""

import numpy as np
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanograd import Tensor


def test_add_broadcast():
    a = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
    b = Tensor(np.random.randn(4).astype(np.float32), requires_grad=True)
    c = (a + b).sum()
    c.backward()

    at = torch.tensor(a.data, requires_grad=True)
    bt = torch.tensor(b.data, requires_grad=True)
    ct = (at + bt).sum()
    ct.backward()

    assert np.allclose(a.grad, at.grad.numpy())
    assert np.allclose(b.grad, bt.grad.numpy())
    print("test_add_broadcast PASSED")


def test_mul_broadcast():
    a = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
    b = Tensor(np.random.randn(1, 4).astype(np.float32), requires_grad=True)
    c = (a * b).sum()
    c.backward()

    at = torch.tensor(a.data, requires_grad=True)
    bt = torch.tensor(b.data, requires_grad=True)
    ct = (at * bt).sum()
    ct.backward()

    assert np.allclose(a.grad, at.grad.numpy())
    assert np.allclose(b.grad, bt.grad.numpy())
    print("test_mul_broadcast PASSED")


def test_matmul():
    A = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
    B = Tensor(np.random.randn(4, 2).astype(np.float32), requires_grad=True)
    C = (A @ B).sum()
    C.backward()

    At = torch.tensor(A.data, requires_grad=True)
    Bt = torch.tensor(B.data, requires_grad=True)
    Ct = (At @ Bt).sum()
    Ct.backward()

    assert np.allclose(A.grad, At.grad.numpy())
    assert np.allclose(B.grad, Bt.grad.numpy())
    print("test_matmul PASSED")


def test_relu():
    a = Tensor(np.array([-1.0, 2.0, -3.0, 4.0]), requires_grad=True)
    c = a.relu().sum()
    c.backward()

    at = torch.tensor([-1.0, 2.0, -3.0, 4.0], requires_grad=True)
    ct = at.relu().sum()
    ct.backward()

    assert np.allclose(a.grad, at.grad.numpy())
    print("test_relu PASSED")


def test_mean():
    a = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
    c = a.mean()
    c.backward()

    at = torch.tensor(a.data, requires_grad=True)
    ct = at.mean()
    ct.backward()

    assert np.allclose(a.grad, at.grad.numpy())
    print("test_mean PASSED")


def test_combined():
    """relu(Wx + b).mean() - basically a neural net forward pass."""
    W = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
    x = Tensor(np.random.randn(4, 2).astype(np.float32), requires_grad=True)
    b = Tensor(np.random.randn(3, 1).astype(np.float32), requires_grad=True)
    y = (W @ x + b).relu().mean()
    y.backward()

    Wt = torch.tensor(W.data, requires_grad=True)
    xt = torch.tensor(x.data, requires_grad=True)
    bt = torch.tensor(b.data, requires_grad=True)
    yt = (Wt @ xt + bt).relu().mean()
    yt.backward()

    assert np.allclose(W.grad, Wt.grad.numpy(), atol=1e-6)
    assert np.allclose(x.grad, xt.grad.numpy(), atol=1e-6)
    assert np.allclose(b.grad, bt.grad.numpy(), atol=1e-6)
    print("test_combined PASSED")


if __name__ == '__main__':
    np.random.seed(42)
    test_add_broadcast()
    test_mul_broadcast()
    test_matmul()
    test_relu()
    test_mean()
    test_combined()
    print("\nAll tensor tests passed!")

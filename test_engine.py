"""Compare our autograd gradients against PyTorch to verify correctness."""

import torch
from engine import Value


def test_sanity():
    """Basic forward + backward test."""
    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xng, yng = x, y

    x = torch.tensor([-4.0], dtype=torch.float64, requires_grad=True)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # forward pass
    assert abs(yng.data - ypt.item()) < 1e-6, f"forward: {yng.data} != {ypt.item()}"
    # backward pass
    assert abs(xng.grad - xpt.grad.item()) < 1e-6, f"backward: {xng.grad} != {xpt.grad.item()}"
    print("test_sanity PASSED")


def test_more_ops():
    """Test a longer expression with more operations."""
    a = Value(3.0)
    b = Value(-2.0)
    c = a * b
    d = c + a
    e = d ** 2
    f = e / Value(2.0)
    f.backward()

    at = torch.tensor([3.0], dtype=torch.float64, requires_grad=True)
    bt = torch.tensor([-2.0], dtype=torch.float64, requires_grad=True)
    ct = at * bt
    dt = ct + at
    et = dt ** 2
    ft = et / 2.0
    ft.backward()

    assert abs(f.data - ft.item()) < 1e-6
    assert abs(a.grad - at.grad.item()) < 1e-6, f"a.grad: {a.grad} != {at.grad.item()}"
    assert abs(b.grad - bt.grad.item()) < 1e-6, f"b.grad: {b.grad} != {bt.grad.item()}"
    print("test_more_ops PASSED")


def test_exp():
    a = Value(1.5)
    b = a.exp()
    c = b * Value(2.0)
    c.backward()

    at = torch.tensor([1.5], dtype=torch.float64, requires_grad=True)
    bt = at.exp()
    ct = bt * 2.0
    ct.backward()

    assert abs(c.data - ct.item()) < 1e-6
    assert abs(a.grad - at.grad.item()) < 1e-6
    print("test_exp PASSED")


def test_tanh():
    a = Value(0.8)
    b = a.tanh()
    c = b + Value(1.0)
    c.backward()

    at = torch.tensor([0.8], dtype=torch.float64, requires_grad=True)
    bt = at.tanh()
    ct = bt + 1.0
    ct.backward()

    assert abs(c.data - ct.item()) < 1e-6
    assert abs(a.grad - at.grad.item()) < 1e-6
    print("test_tanh PASSED")


def test_relu():
    # positive input
    a = Value(2.0)
    b = a.relu()
    b.backward()

    at = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    bt = at.relu()
    bt.backward()

    assert abs(a.grad - at.grad.item()) < 1e-6

    # negative input
    a = Value(-1.0)
    b = a.relu()
    b.backward()

    at = torch.tensor([-1.0], dtype=torch.float64, requires_grad=True)
    bt = at.relu()
    bt.backward()

    assert abs(a.grad - at.grad.item()) < 1e-6
    print("test_relu PASSED")


if __name__ == '__main__':
    test_sanity()
    test_more_ops()
    test_exp()
    test_tanh()
    test_relu()
    print("\nAll tests passed!")

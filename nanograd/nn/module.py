import numpy as np
from nanograd.tensor import Tensor


class Parameter(Tensor):
    """Tensor with requires_grad=True."""

    def __init__(self, data):
        if isinstance(data, np.ndarray):
            super().__init__(data, requires_grad=True)
        else:
            super().__init__(np.array(data, dtype=np.float32), requires_grad=True)

    def __repr__(self):
        return f"Parameter(shape={self.shape})"


class Module:

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def parameters(self):
        params = []
        for attr in vars(self).values():
            if isinstance(attr, Parameter):
                params.append(attr)
            elif isinstance(attr, Module):
                params.extend(attr.parameters())
            elif isinstance(attr, (list, tuple)):
                for item in attr:
                    if isinstance(item, Parameter):
                        params.append(item)
                    elif isinstance(item, Module):
                        params.extend(item.parameters())
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    def _get_submodules(self):
        modules = []
        for attr in vars(self).values():
            if isinstance(attr, Module):
                modules.append(attr)
                modules.extend(attr._get_submodules())
            elif isinstance(attr, (list, tuple)):
                for item in attr:
                    if isinstance(item, Module):
                        modules.append(item)
                        modules.extend(item._get_submodules())
        return modules

    def set_training(self, mode):
        for m in self._get_submodules():
            if hasattr(m, 'training'):
                m.training = mode
        if hasattr(self, 'training'):
            self.training = mode

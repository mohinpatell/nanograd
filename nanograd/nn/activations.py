from nanograd.nn.module import Module


class ReLU(Module):
    def forward(self, x):
        return x.relu()

    def __repr__(self):
        return "ReLU()"

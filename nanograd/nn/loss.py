from nanograd.nn.module import Module


class MSELoss(Module):
    """Mean Squared Error loss."""

    def forward(self, pred, target):
        diff = pred + target * -1  # pred - target
        return (diff * diff).mean()

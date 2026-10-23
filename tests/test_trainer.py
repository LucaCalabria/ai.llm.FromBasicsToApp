import torch
from torch import nn

from trainer.trainer import Trainer


class LossTrackingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.forward_states = []

    def forward(self, xb, yb):
        self.forward_states.append((self.training, torch.is_grad_enabled()))
        logits = self.weight.expand(xb.shape)
        loss = self.weight * 2
        return logits, loss


# Ensure train_step enables training mode, performs an optimizer update, and returns the loss.
def test_train_step_updates_parameters_and_returns_loss():
    model = LossTrackingModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(model, optimizer)
    xb = torch.zeros((2, 3))
    yb = torch.zeros((2, 3))

    loss = trainer.train_step(xb, yb)

    assert loss == 2.0
    assert model.forward_states == [(True, True)]
    assert torch.isclose(model.weight, torch.tensor(0.8))


# Ensure evaluate uses evaluation mode without tracking gradients or changing parameters.
def test_evaluate_disables_gradients_and_does_not_update_parameters():
    model = LossTrackingModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(model, optimizer)
    xb = torch.zeros((2, 3))
    yb = torch.zeros((2, 3))
    original_weight = model.weight.detach().clone()

    loss = trainer.evaluate(xb, yb)

    assert loss == 2.0
    assert model.forward_states == [(False, False)]
    assert torch.equal(model.weight, original_weight)

import torch
import copy

from torch.optim import SGD


def check_parameter_updates(model, inputs):
    model.train()
    params_before = copy.deepcopy([p for p in model.parameters()])
    optim = SGD(model.parameters(), lr=1.0)
    loss = (model(*inputs)**2).sum()
    loss.backward()
    optim.step()
    for before, after in zip(params_before, model.parameters()):
        assert not torch.allclose(before, after)


class DummyHead:
    def __init__(self, feature_size):
        self.feature_size = feature_size

    def __call__(self, *args):
        return torch.cat(args, dim=1)

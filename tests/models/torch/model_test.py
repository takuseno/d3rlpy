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
        assert not torch.allclose(
            before, after), 'tensor with shape of {} is not updated.'.format(
                after.shape)


class DummyHead(torch.nn.Module):
    def __init__(self, feature_size, action_size=None, concat=False):
        super().__init__()
        self.feature_size = feature_size
        self.action_size = action_size
        self.concat = concat

    def __call__(self, *args):
        if self.concat:
            h = torch.cat([args[0][:, :-args[1].shape[1]], args[1]], dim=1)
            return h
        return args[0]

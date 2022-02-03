import copy

import numpy as np
import torch
from torch.optim import SGD


def check_parameter_updates(model, inputs=None, output=None):
    model.train()
    params_before = copy.deepcopy([p for p in model.parameters()])
    optim = SGD(model.parameters(), lr=10.0)
    if output is None:
        if hasattr(model, "compute_error"):
            output = model.compute_error(*inputs)
        else:
            output = model(*inputs)
    if isinstance(output, (list, tuple)):
        loss = 0.0
        for y in output:
            loss += (y**2).sum()
    else:
        loss = (output**2).sum()
    loss.backward()
    optim.step()
    for before, after in zip(params_before, model.parameters()):
        assert not torch.allclose(
            before, after
        ), "tensor with shape of {} is not updated.".format(after.shape)


def ref_huber_loss(a, b):
    abs_diff = np.abs(a - b).reshape((-1,))
    l2_diff = ((a - b) ** 2).reshape((-1,))
    huber_diff = np.zeros_like(abs_diff)
    huber_diff[abs_diff < 1.0] = 0.5 * l2_diff[abs_diff < 1.0]
    huber_diff[abs_diff >= 1.0] = abs_diff[abs_diff >= 1.0] - 0.5
    return np.mean(huber_diff)


def ref_quantile_huber_loss(a, b, taus, n_quantiles):
    abs_diff = np.abs(a - b).reshape((-1,))
    l2_diff = ((a - b) ** 2).reshape((-1,))
    huber_diff = np.zeros_like(abs_diff)
    huber_diff[abs_diff < 1.0] = 0.5 * l2_diff[abs_diff < 1.0]
    huber_diff[abs_diff >= 1.0] = abs_diff[abs_diff >= 1.0] - 0.5
    huber_diff = huber_diff.reshape(-1, n_quantiles, n_quantiles)
    delta = np.array((b - a) < 0.0, dtype=np.float32)
    element_wise_loss = np.abs(taus - delta) * huber_diff
    return element_wise_loss.sum(axis=2).mean(axis=1)


class DummyEncoder(torch.nn.Module):
    def __init__(self, feature_size, action_size=None, concat=False):
        super().__init__()
        self.feature_size = feature_size
        self.observation_shape = (feature_size,)
        self.action_size = action_size
        self.concat = concat

    def __call__(self, *args):
        if self.concat:
            h = torch.cat([args[0][:, : -args[1].shape[1]], args[1]], dim=1)
            return h
        return args[0]

    def get_feature_size(self):
        return self.feature_size

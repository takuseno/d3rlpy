import copy
from typing import Any, Sequence

import numpy as np
import torch
from torch.optim import SGD

from d3rlpy.models.torch import ActionOutput, QFunctionOutput
from d3rlpy.models.torch.encoders import Encoder, EncoderWithAction
from d3rlpy.types import NDArray


def check_parameter_updates(
    model: torch.nn.Module, inputs: Any = None, output: Any = None
) -> None:
    model.train()
    params_before = copy.deepcopy(list(model.parameters()))
    optim = SGD(model.parameters(), lr=1000.0)
    if output is None:
        if hasattr(model, "compute_error"):
            output = model.compute_error(*inputs)
        else:
            output = model(*inputs)
            if isinstance(output, ActionOutput):
                mu = output.squashed_mu
                logstd = output.logstd
                output = mu
                if logstd is not None:
                    output = output + logstd
            elif isinstance(output, QFunctionOutput):
                output = output.q_value
    if isinstance(output, (list, tuple)):
        loss = 0.0
        for y in output:
            loss += (y**2).sum()
    else:
        loss = (output**2).sum()
    loss.backward()  # type: ignore
    optim.step()
    for before, after in zip(params_before, model.parameters()):
        assert not torch.allclose(
            before, after
        ), f"tensor with shape of {after.shape} is not updated."


def ref_huber_loss(a: NDArray, b: NDArray) -> float:
    abs_diff = np.abs(a - b).reshape((-1,))
    l2_diff = ((a - b) ** 2).reshape((-1,))
    huber_diff = np.zeros_like(abs_diff)
    huber_diff[abs_diff < 1.0] = 0.5 * l2_diff[abs_diff < 1.0]
    huber_diff[abs_diff >= 1.0] = abs_diff[abs_diff >= 1.0] - 0.5
    return float(np.mean(huber_diff))


def ref_quantile_huber_loss(
    a: NDArray, b: NDArray, taus: NDArray, n_quantiles: int
) -> NDArray:
    abs_diff = np.abs(a - b).reshape((-1,))
    l2_diff = ((a - b) ** 2).reshape((-1,))
    huber_diff = np.zeros_like(abs_diff)
    huber_diff[abs_diff < 1.0] = 0.5 * l2_diff[abs_diff < 1.0]
    huber_diff[abs_diff >= 1.0] = abs_diff[abs_diff >= 1.0] - 0.5
    huber_diff = huber_diff.reshape((-1, n_quantiles, n_quantiles))
    delta = np.array((b - a) < 0.0, dtype=np.float32)
    element_wise_loss = np.abs(taus - delta) * huber_diff
    return element_wise_loss.sum(axis=2).mean(axis=1)  # type: ignore


class DummyEncoder(Encoder):
    def __init__(self, feature_size: int):
        super().__init__()
        self.feature_size = feature_size
        self._observation_shape = (feature_size,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    @property
    def observation_shape(self) -> Sequence[int]:
        return self._observation_shape

    def get_feature_size(self) -> int:
        return self.feature_size


class DummyEncoderWithAction(EncoderWithAction):
    def __init__(self, feature_size: int, action_size: int):
        super().__init__()
        self.feature_size = feature_size
        self._observation_shape = (feature_size,)
        self._action_size = action_size

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.cat([x[:, : -action.shape[1]], action], dim=1)

    def get_feature_size(self) -> int:
        return self.feature_size

    @property
    def observation_shape(self) -> Sequence[int]:
        return self._observation_shape

    @property
    def action_size(self) -> int:
        return self._action_size

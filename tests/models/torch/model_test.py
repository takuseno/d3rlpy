import copy
from typing import Any

import numpy as np
import torch
from torch.optim import SGD

from d3rlpy.models import EncoderFactory, register_encoder_factory
from d3rlpy.models.torch import ActionOutput, QFunctionOutput
from d3rlpy.models.torch.encoders import Encoder, EncoderWithAction
from d3rlpy.types import Float32NDArray, NDArray, Shape, TorchObservation


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
    delta: Float32NDArray = np.array((b - a) < 0.0, dtype=np.float32)
    element_wise_loss = np.abs(taus - delta) * huber_diff
    return element_wise_loss.sum(axis=2).mean(axis=1)  # type: ignore


class DummyEncoder(Encoder):
    def __init__(self, input_shape: Shape):
        super().__init__()
        self.input_shape = input_shape
        self.dummy_parameter = torch.nn.Parameter(
            torch.rand(1, self.get_feature_size())
        )

    def forward(self, x: TorchObservation) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            y = x.view(x.shape[0], -1)
        else:
            batch_size = x[0].shape[0]
            y = torch.cat([_x.view(batch_size, -1) for _x in x], dim=-1)
        return y + self.dummy_parameter

    def get_feature_size(self) -> int:
        if isinstance(self.input_shape[0], int):
            return int(np.cumprod(self.input_shape)[-1])
        else:
            return sum([np.cumprod(shape)[-1] for shape in self.input_shape])


class DummyEncoderWithAction(EncoderWithAction):
    def __init__(self, input_shape: Shape, action_size: int):
        super().__init__()
        self.input_shape = input_shape
        self._action_size = action_size
        self.dummy_parameter = torch.nn.Parameter(
            torch.rand(1, self.get_feature_size())
        )

    def forward(
        self, x: TorchObservation, action: torch.Tensor
    ) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            y = x.view(x.shape[0], -1)
        else:
            batch_size = x[0].shape[0]
            y = torch.cat([_x.view(batch_size, -1) for _x in x], dim=-1)
        return torch.cat([y, action], dim=-1) + self.dummy_parameter

    def get_feature_size(self) -> int:
        if isinstance(self.input_shape[0], int):
            feature_size = int(np.cumprod(self.input_shape)[-1])
        else:
            feature_size = sum(
                [np.cumprod(shape)[-1] for shape in self.input_shape]
            )
        return feature_size + self._action_size


class DummyEncoderFactory(EncoderFactory):
    def create(self, observation_shape: Shape) -> DummyEncoder:
        return DummyEncoder(observation_shape)

    def create_with_action(
        self,
        observation_shape: Shape,
        action_size: int,
        discrete_action: bool = False,
    ) -> DummyEncoderWithAction:
        return DummyEncoderWithAction(observation_shape, action_size)

    @staticmethod
    def get_type() -> str:
        return "dummy"


register_encoder_factory(DummyEncoderFactory)

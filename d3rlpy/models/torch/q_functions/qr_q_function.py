import dataclasses
from typing import Optional, Union

import torch
from torch import nn

from ....torch_utility import get_batch_size, get_device
from ....types import TorchObservation
from ..encoders import Encoder, EncoderWithAction
from .base import (
    ContinuousQFunction,
    ContinuousQFunctionForwarder,
    DiscreteQFunction,
    DiscreteQFunctionForwarder,
    QFunctionOutput,
    TargetOutput,
)
from .utility import (
    compute_quantile_loss,
    compute_reduce,
    pick_quantile_value_by_action,
    pick_value_by_action,
)

__all__ = [
    "DiscreteQRQFunction",
    "ContinuousQRQFunction",
    "ContinuousQRQFunctionForwarder",
    "DiscreteQRQFunctionForwarder",
    "QuantileTargetOutput",
]


def _make_taus(n_quantiles: int, device: torch.device) -> torch.Tensor:
    steps = torch.arange(n_quantiles, dtype=torch.float32, device=device)
    taus = ((steps + 1).float() / n_quantiles).view(1, -1)
    taus_dot = (steps.float() / n_quantiles).view(1, -1)
    return (taus + taus_dot) / 2.0


@dataclasses.dataclass(frozen=True)
class QuantileTargetOutput(TargetOutput):
    quantile: torch.Tensor

    def add(self, v: torch.Tensor) -> "QuantileTargetOutput":
        assert v.shape == (self.q_value.shape[0], 1)
        return QuantileTargetOutput(
            q_value=self.q_value + v,
            quantile=(
                (self.quantile + v.unsqueeze(dim=1))
                if self.quantile.ndim == 3
                else (self.quantile + v)
            ),
        )


class DiscreteQRQFunction(DiscreteQFunction):
    _action_size: int
    _encoder: Encoder
    _n_quantiles: int
    _fc: nn.Linear

    def __init__(
        self,
        encoder: Encoder,
        hidden_size: int,
        action_size: int,
        n_quantiles: int,
    ):
        super().__init__()
        self._encoder = encoder
        self._action_size = action_size
        self._n_quantiles = n_quantiles
        self._fc = nn.Linear(hidden_size, action_size * n_quantiles)

    def forward(self, x: TorchObservation) -> QFunctionOutput:
        quantiles = self._fc(self._encoder(x))
        quantiles = quantiles.view(-1, self._action_size, self._n_quantiles)
        return QFunctionOutput(
            q_value=quantiles.mean(dim=2),
            quantiles=quantiles,
            taus=_make_taus(self._n_quantiles, device=get_device(x)),
        )

    @property
    def encoder(self) -> Encoder:
        return self._encoder


class DiscreteQRQFunctionForwarder(DiscreteQFunctionForwarder):
    _q_func: DiscreteQRQFunction
    _n_quantiles: int

    def __init__(self, q_func: DiscreteQRQFunction, n_quantiles: int):
        self._q_func = q_func
        self._n_quantiles = n_quantiles

    def compute_expected_q(self, x: TorchObservation) -> torch.Tensor:
        return self._q_func(x).q_value

    def compute_error(
        self,
        observations: TorchObservation,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        target: TargetOutput,
        terminals: torch.Tensor,
        gamma: Union[float, torch.Tensor] = 0.99,
        reduction: str = "mean",
    ) -> torch.Tensor:
        assert isinstance(target, QuantileTargetOutput)
        batch_size = get_batch_size(observations)
        assert target.quantile.shape == (batch_size, self._n_quantiles)

        # extraect quantiles corresponding to act_t
        output = self._q_func(observations)
        all_quantiles = output.quantiles
        taus = output.taus
        assert all_quantiles is not None and taus is not None
        quantiles = pick_quantile_value_by_action(all_quantiles, actions)

        loss = compute_quantile_loss(
            quantiles=quantiles,
            rewards=rewards,
            target=target.quantile,
            terminals=terminals,
            taus=taus,
            gamma=gamma,
        )

        return compute_reduce(loss, reduction)

    def compute_target(
        self, x: TorchObservation, action: Optional[torch.Tensor] = None
    ) -> QuantileTargetOutput:
        q_output = self._q_func(x)
        q_value = q_output.q_value
        quantiles = q_output.quantiles
        assert quantiles is not None
        if action is not None:
            q_value = pick_value_by_action(q_value, action, keepdim=True)
            quantiles = pick_quantile_value_by_action(quantiles, action)
        return QuantileTargetOutput(
            q_value=q_value,
            quantile=quantiles,
        )

    def set_q_func(self, q_func: DiscreteQFunction) -> None:
        self._q_func = q_func


class ContinuousQRQFunction(ContinuousQFunction):
    _encoder: EncoderWithAction
    _fc: nn.Linear
    _n_quantiles: int

    def __init__(
        self,
        encoder: EncoderWithAction,
        hidden_size: int,
        n_quantiles: int,
    ):
        super().__init__()
        self._encoder = encoder
        self._fc = nn.Linear(hidden_size, n_quantiles)
        self._n_quantiles = n_quantiles

    def forward(
        self, x: TorchObservation, action: torch.Tensor
    ) -> QFunctionOutput:
        quantiles = self._fc(self._encoder(x, action))
        return QFunctionOutput(
            q_value=quantiles.mean(dim=1, keepdim=True),
            quantiles=quantiles,
            taus=_make_taus(self._n_quantiles, device=get_device(x)),
        )

    @property
    def encoder(self) -> EncoderWithAction:
        return self._encoder


class ContinuousQRQFunctionForwarder(ContinuousQFunctionForwarder):
    _q_func: ContinuousQRQFunction
    _n_quantiles: int

    def __init__(self, q_func: ContinuousQRQFunction, n_quantiles: int):
        self._q_func = q_func
        self._n_quantiles = n_quantiles

    def compute_expected_q(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        return self._q_func(x, action).q_value

    def compute_error(
        self,
        observations: TorchObservation,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        target: TargetOutput,
        terminals: torch.Tensor,
        gamma: Union[float, torch.Tensor] = 0.99,
        reduction: str = "mean",
    ) -> torch.Tensor:
        assert isinstance(target, QuantileTargetOutput)
        batch_size = get_batch_size(observations)
        assert target.quantile.shape == (batch_size, self._n_quantiles)

        output = self._q_func(observations, actions)
        quantiles = output.quantiles
        taus = output.taus
        assert quantiles is not None and taus is not None

        loss = compute_quantile_loss(
            quantiles=quantiles,
            rewards=rewards,
            target=target.quantile,
            terminals=terminals,
            taus=taus,
            gamma=gamma,
        )

        return compute_reduce(loss, reduction)

    def compute_target(
        self, x: TorchObservation, action: torch.Tensor
    ) -> QuantileTargetOutput:
        q_output = self._q_func(x, action)
        quantiles = q_output.quantiles
        assert quantiles is not None
        return QuantileTargetOutput(
            q_value=q_output.q_value,
            quantile=quantiles,
        )

    def set_q_func(self, q_func: ContinuousQFunction) -> None:
        self._q_func = q_func

from typing import Optional

import torch
from torch import nn

from ..encoders import Encoder, EncoderWithAction
from .base import (
    ContinuousQFunction,
    ContinuousQFunctionForwarder,
    DiscreteQFunction,
    DiscreteQFunctionForwarder,
    QFunctionOutput,
)
from .utility import (
    compute_quantile_loss,
    compute_reduce,
    pick_quantile_value_by_action,
)

__all__ = [
    "DiscreteQRQFunction",
    "ContinuousQRQFunction",
    "ContinuousQRQFunctionForwarder",
    "DiscreteQRQFunctionForwarder",
]


def _make_taus(n_quantiles: int, device: torch.device) -> torch.Tensor:
    steps = torch.arange(n_quantiles, dtype=torch.float32, device=device)
    taus = ((steps + 1).float() / n_quantiles).view(1, -1)
    taus_dot = (steps.float() / n_quantiles).view(1, -1)
    return (taus + taus_dot) / 2.0


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

    def forward(self, x: torch.Tensor) -> QFunctionOutput:
        quantiles = self._fc(self._encoder(x))
        quantiles = quantiles.view(-1, self._action_size, self._n_quantiles)
        return QFunctionOutput(
            q_value=quantiles.mean(dim=2),
            quantiles=quantiles,
            taus=_make_taus(self._n_quantiles, device=x.device),
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

    def compute_expected_q(self, x: torch.Tensor) -> torch.Tensor:
        return self._q_func(x).q_value

    def compute_error(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        target: torch.Tensor,
        terminals: torch.Tensor,
        gamma: float = 0.99,
        reduction: str = "mean",
    ) -> torch.Tensor:
        assert target.shape == (observations.shape[0], self._n_quantiles)

        # extraect quantiles corresponding to act_t
        output = self._q_func(observations)
        all_quantiles = output.quantiles
        taus = output.taus
        assert all_quantiles is not None and taus is not None
        quantiles = pick_quantile_value_by_action(all_quantiles, actions)

        loss = compute_quantile_loss(
            quantiles=quantiles,
            rewards=rewards,
            target=target,
            terminals=terminals,
            taus=taus,
            gamma=gamma,
        )

        return compute_reduce(loss, reduction)

    def compute_target(
        self, x: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        quantiles = self._q_func(x).quantiles
        assert quantiles is not None
        if action is None:
            return quantiles
        return pick_quantile_value_by_action(quantiles, action)


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

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> QFunctionOutput:
        quantiles = self._fc(self._encoder(x, action))
        return QFunctionOutput(
            q_value=quantiles.mean(dim=1, keepdim=True),
            quantiles=quantiles,
            taus=_make_taus(self._n_quantiles, device=x.device),
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
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        target: torch.Tensor,
        terminals: torch.Tensor,
        gamma: float = 0.99,
        reduction: str = "mean",
    ) -> torch.Tensor:
        assert target.shape == (observations.shape[0], self._n_quantiles)

        output = self._q_func(observations, actions)
        quantiles = output.quantiles
        taus = output.taus
        assert quantiles is not None and taus is not None

        loss = compute_quantile_loss(
            quantiles=quantiles,
            rewards=rewards,
            target=target,
            terminals=terminals,
            taus=taus,
            gamma=gamma,
        )

        return compute_reduce(loss, reduction)

    def compute_target(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        quantiles = self._q_func(x, action).quantiles
        assert quantiles is not None
        return quantiles

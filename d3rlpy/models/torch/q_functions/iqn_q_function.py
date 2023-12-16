import math
from typing import Optional, Union

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
    "DiscreteIQNQFunction",
    "ContinuousIQNQFunction",
    "DiscreteIQNQFunctionForwarder",
    "ContinuousIQNQFunctionForwarder",
]


def _make_taus(
    batch_size: int, n_quantiles: int, training: bool, device: torch.device
) -> torch.Tensor:
    if training:
        taus = torch.rand(batch_size, n_quantiles, device=device)
    else:
        taus = torch.linspace(
            start=0,
            end=1,
            steps=n_quantiles,
            device=device,
            dtype=torch.float32,
        )
        taus = taus.view(1, -1).repeat(batch_size, 1)
    return taus


def compute_iqn_feature(
    h: torch.Tensor,
    taus: torch.Tensor,
    embed: nn.Linear,
    embed_size: int,
) -> torch.Tensor:
    # compute embedding
    steps = torch.arange(embed_size, device=h.device).float() + 1
    # (batch, quantile, embedding)
    expanded_taus = taus.view(h.shape[0], -1, 1)
    prior = torch.cos(math.pi * steps.view(1, 1, -1) * expanded_taus)
    # (batch, quantile, embedding) -> (batch, quantile, feature)
    phi = torch.relu(embed(prior))
    # (batch, 1, feature) -> (batch,  quantile, feature)
    return h.view(h.shape[0], 1, -1) * phi


class DiscreteIQNQFunction(DiscreteQFunction):
    _action_size: int
    _encoder: Encoder
    _fc: nn.Linear
    _n_quantiles: int
    _n_greedy_quantiles: int
    _embed_size: int
    _embed: nn.Linear

    def __init__(
        self,
        encoder: Encoder,
        hidden_size: int,
        action_size: int,
        n_quantiles: int,
        n_greedy_quantiles: int,
        embed_size: int,
    ):
        super().__init__()
        self._encoder = encoder
        self._action_size = action_size
        self._fc = nn.Linear(hidden_size, self._action_size)
        self._n_quantiles = n_quantiles
        self._n_greedy_quantiles = n_greedy_quantiles
        self._embed_size = embed_size
        self._embed = nn.Linear(embed_size, hidden_size)

    def forward(self, x: torch.Tensor) -> QFunctionOutput:
        h = self._encoder(x)

        if self.training:
            n_quantiles = self._n_quantiles
        else:
            n_quantiles = self._n_greedy_quantiles
        taus = _make_taus(
            batch_size=x.shape[0],
            n_quantiles=n_quantiles,
            training=self.training,
            device=x.device,
        )

        # (batch, quantile, feature)
        prod = compute_iqn_feature(h, taus, self._embed, self._embed_size)
        # (batch, quantile, action) -> (batch, action, quantile)
        quantiles = self._fc(prod).transpose(1, 2)

        return QFunctionOutput(
            q_value=quantiles.mean(dim=2),
            quantiles=quantiles,
            taus=taus,
        )

    @property
    def encoder(self) -> Encoder:
        return self._encoder


class DiscreteIQNQFunctionForwarder(DiscreteQFunctionForwarder):
    _q_func: DiscreteIQNQFunction
    _n_quantiles: int

    def __init__(self, q_func: DiscreteIQNQFunction, n_quantiles: int):
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
        gamma: Union[float, torch.Tensor] = 0.99,
        reduction: str = "mean",
    ) -> torch.Tensor:
        assert target.shape == (observations.shape[0], self._n_quantiles)

        # extraect quantiles corresponding to act_t
        output = self._q_func(observations)
        taus = output.taus
        all_quantiles = output.quantiles
        assert taus is not None and all_quantiles is not None
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


class ContinuousIQNQFunction(ContinuousQFunction, nn.Module):  # type: ignore
    _encoder: EncoderWithAction
    _fc: nn.Linear
    _n_quantiles: int
    _n_greedy_quantiles: int
    _embed_size: int
    _embed: nn.Linear

    def __init__(
        self,
        encoder: EncoderWithAction,
        hidden_size: int,
        n_quantiles: int,
        n_greedy_quantiles: int,
        embed_size: int,
    ):
        super().__init__()
        self._encoder = encoder
        self._fc = nn.Linear(hidden_size, 1)
        self._n_quantiles = n_quantiles
        self._n_greedy_quantiles = n_greedy_quantiles
        self._embed_size = embed_size
        self._embed = nn.Linear(embed_size, hidden_size)

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> QFunctionOutput:
        h = self._encoder(x, action)

        if self.training:
            n_quantiles = self._n_quantiles
        else:
            n_quantiles = self._n_greedy_quantiles
        taus = _make_taus(
            batch_size=x.shape[0],
            n_quantiles=n_quantiles,
            training=self.training,
            device=x.device,
        )

        # element-wise product on feature and phi (batch, quantile, feature)
        prod = compute_iqn_feature(h, taus, self._embed, self._embed_size)
        # (batch, quantile, feature) -> (batch, quantile)
        quantiles = self._fc(prod).view(h.shape[0], -1)

        return QFunctionOutput(
            q_value=quantiles.mean(dim=1, keepdim=True),
            quantiles=quantiles,
            taus=taus,
        )

    @property
    def encoder(self) -> EncoderWithAction:
        return self._encoder


class ContinuousIQNQFunctionForwarder(ContinuousQFunctionForwarder):
    _q_func: ContinuousIQNQFunction
    _n_quantiles: int

    def __init__(self, q_func: ContinuousIQNQFunction, n_quantiles: int):
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
        gamma: Union[float, torch.Tensor] = 0.99,
        reduction: str = "mean",
    ) -> torch.Tensor:
        assert target.shape == (observations.shape[0], self._n_quantiles)

        output = self._q_func(observations, actions)
        taus = output.taus
        quantiles = output.quantiles
        assert taus is not None and quantiles is not None

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

import dataclasses
import math
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
    "DiscreteIQNQFunction",
    "ContinuousIQNQFunction",
    "DiscreteIQNQFunctionForwarder",
    "ContinuousIQNQFunctionForwarder",
    "ImplicitQuantileTargetOutput",
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


@dataclasses.dataclass(frozen=True)
class ImplicitQuantileTargetOutput(TargetOutput):
    quantile: torch.Tensor
    taus: torch.Tensor

    def add(self, v: torch.Tensor) -> "ImplicitQuantileTargetOutput":
        assert v.shape == (self.q_value.shape[0], 1)
        return ImplicitQuantileTargetOutput(
            q_value=self.q_value + v,
            quantile=(
                (self.quantile + v.unsqueeze(dim=1))
                if self.quantile.ndim == 3
                else (self.quantile + v)
            ),
            taus=self.taus,
        )


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

    def forward(self, x: TorchObservation) -> QFunctionOutput:
        h = self._encoder(x)

        if self.training:
            n_quantiles = self._n_quantiles
        else:
            n_quantiles = self._n_greedy_quantiles
        taus = _make_taus(
            batch_size=get_batch_size(x),
            n_quantiles=n_quantiles,
            training=self.training,
            device=torch.device(get_device(x)),
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
        assert isinstance(target, ImplicitQuantileTargetOutput)
        batch_size = get_batch_size(observations)
        assert target.quantile.shape == (batch_size, self._n_quantiles)

        # extraect quantiles corresponding to act_t
        output = self._q_func(observations)
        taus = output.taus
        all_quantiles = output.quantiles
        assert taus is not None and all_quantiles is not None
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
    ) -> ImplicitQuantileTargetOutput:
        q_output = self._q_func(x)
        q_value = q_output.q_value
        quantiles = q_output.quantiles
        assert quantiles is not None and q_output.taus is not None
        if action is not None:
            q_value = pick_value_by_action(q_value, action, keepdim=True)
            quantiles = pick_quantile_value_by_action(quantiles, action)
        return ImplicitQuantileTargetOutput(
            q_value=q_value,
            quantile=quantiles,
            taus=q_output.taus,
        )

    def set_q_func(self, q_func: DiscreteQFunction) -> None:
        self._q_func = q_func


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

    def forward(
        self, x: TorchObservation, action: torch.Tensor
    ) -> QFunctionOutput:
        h = self._encoder(x, action)

        if self.training:
            n_quantiles = self._n_quantiles
        else:
            n_quantiles = self._n_greedy_quantiles
        taus = _make_taus(
            batch_size=get_batch_size(x),
            n_quantiles=n_quantiles,
            training=self.training,
            device=torch.device(get_device(x)),
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
        self, x: TorchObservation, action: torch.Tensor
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
        assert isinstance(target, ImplicitQuantileTargetOutput)
        batch_size = get_batch_size(observations)
        assert target.quantile.shape == (batch_size, self._n_quantiles)

        output = self._q_func(observations, actions)
        taus = output.taus
        quantiles = output.quantiles
        assert taus is not None and quantiles is not None

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
    ) -> ImplicitQuantileTargetOutput:
        q_output = self._q_func(x, action)
        quantiles = q_output.quantiles
        assert quantiles is not None and q_output.taus is not None
        return ImplicitQuantileTargetOutput(
            q_value=q_output.q_value,
            quantile=quantiles,
            taus=q_output.taus,
        )

    def set_q_func(self, q_func: ContinuousQFunction) -> None:
        self._q_func = q_func

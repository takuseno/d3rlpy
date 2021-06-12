import math
from typing import Optional, cast

import torch
import torch.nn as nn

from ..encoders import Encoder, EncoderWithAction
from .base import ContinuousQFunction, DiscreteQFunction
from .qr_q_function import QRQFunction
from .utility import compute_reduce, pick_quantile_value_by_action


class IQNQFunction(QRQFunction):
    _n_greedy_quantiles: int
    _embed_size: int
    _embed: nn.Linear

    def __init__(
        self,
        n_quantiles: int,
        n_greedy_quantiles: int,
        embed_size: int,
        feature_size: int,
    ):
        super().__init__(n_quantiles)
        self._n_greedy_quantiles = n_greedy_quantiles
        self._embed_size = embed_size
        self._embed = nn.Linear(embed_size, feature_size)

    def _make_taus(self, h: torch.Tensor) -> torch.Tensor:
        if self.training:
            taus = torch.rand(h.shape[0], self._n_quantiles, device=h.device)
        else:
            taus = torch.linspace(
                start=0,
                end=1,
                steps=self._n_greedy_quantiles,
                device=h.device,
                dtype=torch.float32,
            )
            taus = taus.view(1, -1).repeat(h.shape[0], 1)
        return taus

    def _compute_last_feature(
        self, h: torch.Tensor, taus: torch.Tensor
    ) -> torch.Tensor:
        # compute embedding
        steps = torch.arange(self._embed_size, device=h.device).float() + 1
        # (batch, quantile, embedding)
        expanded_taus = taus.view(h.shape[0], -1, 1)
        prior = torch.cos(math.pi * steps.view(1, 1, -1) * expanded_taus)
        # (batch, quantile, embedding) -> (batch, quantile, feature)
        phi = torch.relu(self._embed(prior))

        # (batch, 1, feature) -> (batch,  quantile, feature)
        return h.view(h.shape[0], 1, -1) * phi


class DiscreteIQNQFunction(IQNQFunction, DiscreteQFunction):
    _action_size: int
    _encoder: Encoder
    _fc: nn.Linear

    def __init__(
        self,
        encoder: Encoder,
        action_size: int,
        n_quantiles: int,
        n_greedy_quantiles: int,
        embed_size: int,
    ):
        super().__init__(
            n_quantiles,
            n_greedy_quantiles,
            embed_size,
            encoder.get_feature_size(),
        )
        self._encoder = encoder
        self._action_size = action_size
        self._fc = nn.Linear(encoder.get_feature_size(), self._action_size)

    def _compute_quantiles(
        self, h: torch.Tensor, taus: torch.Tensor
    ) -> torch.Tensor:
        # element-wise product on feature and phi (batch, quantile, feature)
        prod = self._compute_last_feature(h, taus)
        # (batch, quantile, feature) -> (batch, action, quantile)
        return cast(torch.Tensor, self._fc(prod)).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._encoder(x)
        taus = self._make_taus(h)
        quantiles = self._compute_quantiles(h, taus)
        return quantiles.mean(dim=2)

    def compute_error(
        self,
        obs_t: torch.Tensor,
        act_t: torch.Tensor,
        rew_tp1: torch.Tensor,
        q_tp1: torch.Tensor,
        ter_tp1: torch.Tensor,
        gamma: float = 0.99,
        reduction: str = "mean",
    ) -> torch.Tensor:
        assert q_tp1.shape == (obs_t.shape[0], self._n_quantiles)

        # extraect quantiles corresponding to act_t
        h = self._encoder(obs_t)
        taus = self._make_taus(h)
        quantiles = self._compute_quantiles(h, taus)
        quantiles_t = pick_quantile_value_by_action(quantiles, act_t)

        loss = self._compute_quantile_loss(
            quantiles_t=quantiles_t,
            rew_tp1=rew_tp1,
            q_tp1=q_tp1,
            ter_tp1=ter_tp1,
            taus=taus,
            gamma=gamma,
        )

        return compute_reduce(loss, reduction)

    def compute_target(
        self, x: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        h = self._encoder(x)
        taus = self._make_taus(h)
        quantiles = self._compute_quantiles(h, taus)
        if action is None:
            return quantiles
        return pick_quantile_value_by_action(quantiles, action)

    @property
    def action_size(self) -> int:
        return self._action_size

    @property
    def encoder(self) -> Encoder:
        return self._encoder


class ContinuousIQNQFunction(IQNQFunction, ContinuousQFunction):
    _action_size: int
    _encoder: EncoderWithAction
    _fc: nn.Linear

    def __init__(
        self,
        encoder: EncoderWithAction,
        n_quantiles: int,
        n_greedy_quantiles: int,
        embed_size: int,
    ):
        super().__init__(
            n_quantiles,
            n_greedy_quantiles,
            embed_size,
            encoder.get_feature_size(),
        )
        self._encoder = encoder
        self._action_size = encoder.action_size
        self._fc = nn.Linear(encoder.get_feature_size(), 1)

    def _compute_quantiles(
        self, h: torch.Tensor, taus: torch.Tensor
    ) -> torch.Tensor:
        # element-wise product on feature and phi (batch, quantile, feature)
        prod = self._compute_last_feature(h, taus)
        # (batch, quantile, feature) -> (batch, quantile)
        return cast(torch.Tensor, self._fc(prod)).view(h.shape[0], -1)

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        h = self._encoder(x, action)
        taus = self._make_taus(h)
        quantiles = self._compute_quantiles(h, taus)
        return quantiles.mean(dim=1, keepdim=True)

    def compute_error(
        self,
        obs_t: torch.Tensor,
        act_t: torch.Tensor,
        rew_tp1: torch.Tensor,
        q_tp1: torch.Tensor,
        ter_tp1: torch.Tensor,
        gamma: float = 0.99,
        reduction: str = "mean",
    ) -> torch.Tensor:
        assert q_tp1.shape == (obs_t.shape[0], self._n_quantiles)

        h = self._encoder(obs_t, act_t)
        taus = self._make_taus(h)
        quantiles_t = self._compute_quantiles(h, taus)

        loss = self._compute_quantile_loss(
            quantiles_t=quantiles_t,
            rew_tp1=rew_tp1,
            q_tp1=q_tp1,
            ter_tp1=ter_tp1,
            taus=taus,
            gamma=gamma,
        )

        return compute_reduce(loss, reduction)

    def compute_target(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        h = self._encoder(x, action)
        taus = self._make_taus(h)
        return self._compute_quantiles(h, taus)

    @property
    def action_size(self) -> int:
        return self._action_size

    @property
    def encoder(self) -> EncoderWithAction:
        return self._encoder

from typing import Optional, Tuple, cast

import torch
import torch.nn as nn

from ..encoders import Encoder, EncoderWithAction
from .base import ContinuousQFunction, DiscreteQFunction
from .iqn_q_function import IQNQFunction
from .utility import compute_reduce, pick_quantile_value_by_action


class FQFQFunction(IQNQFunction):
    _proposal: nn.Linear

    def __init__(self, n_quantiles: int, embed_size: int, feature_size: int):
        super().__init__(n_quantiles, n_quantiles, embed_size, feature_size)
        self._proposal = nn.Linear(feature_size, n_quantiles)

    def _make_fqf_taus(
        self, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        proposals = self._proposal(h.detach())

        # tau_i+1
        log_probs = torch.log_softmax(proposals, dim=1)
        probs = log_probs.exp()
        taus = torch.cumsum(probs, dim=1)

        # tau_i
        pads = torch.zeros(h.shape[0], 1, device=h.device)
        taus_minus = torch.cat([pads, taus[:, :-1]], dim=1)

        # tau^
        taus_prime = (taus + taus_minus) / 2

        # entropy for penalty
        entropies = -(log_probs * probs).sum(dim=1)

        return taus, taus_minus, taus_prime, entropies


class DiscreteFQFQFunction(FQFQFunction, DiscreteQFunction):
    _action_size: int
    _entropy_coeff: float
    _encoder: Encoder
    _fc: nn.Linear

    def __init__(
        self,
        encoder: Encoder,
        action_size: int,
        n_quantiles: int,
        embed_size: int,
        entropy_coeff: float = 0.0,
    ):
        super().__init__(n_quantiles, embed_size, encoder.get_feature_size())
        self._encoder = encoder
        self._action_size = action_size
        self._fc = nn.Linear(encoder.get_feature_size(), self._action_size)
        self._entropy_coeff = entropy_coeff

    def _compute_quantiles(
        self, h: torch.Tensor, taus: torch.Tensor
    ) -> torch.Tensor:
        # element-wise product on feature and phi (batch, quantile, feature)
        prod = self._compute_last_feature(h, taus)
        # (batch, quantile, feature) -> (batch, action, quantile)
        return cast(torch.Tensor, self._fc(prod)).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._encoder(x)
        taus, taus_minus, taus_prime, _ = self._make_fqf_taus(h)
        quantiles = self._compute_quantiles(h, taus_prime.detach())
        weight = (taus - taus_minus).view(-1, 1, self._n_quantiles).detach()
        return (weight * quantiles).sum(dim=2)

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

        # compute quantiles
        h = self._encoder(obs_t)
        taus, _, taus_prime, entropies = self._make_fqf_taus(h)
        quantiles = self._compute_quantiles(h, taus_prime.detach())
        quantiles_t = pick_quantile_value_by_action(quantiles, act_t)

        quantile_loss = self._compute_quantile_loss(
            quantiles_t=quantiles_t,
            rew_tp1=rew_tp1,
            q_tp1=q_tp1,
            ter_tp1=ter_tp1,
            taus=taus_prime.detach(),
            gamma=gamma,
        )

        # compute proposal network loss
        # original paper explicitly separates the optimization process
        # but, it's combined here
        proposal_loss = self._compute_proposal_loss(h, act_t, taus, taus_prime)
        proposal_params = list(self._proposal.parameters())
        proposal_grads = torch.autograd.grad(
            outputs=proposal_loss.mean(),
            inputs=proposal_params,
            retain_graph=True,
        )
        # directly apply gradients
        for param, grad in zip(list(proposal_params), proposal_grads):
            param.grad = 1e-4 * grad

        loss = quantile_loss - self._entropy_coeff * entropies

        return compute_reduce(loss, reduction)

    def _compute_proposal_loss(
        self,
        h: torch.Tensor,
        action: torch.Tensor,
        taus: torch.Tensor,
        taus_prime: torch.Tensor,
    ) -> torch.Tensor:
        q_taus = self._compute_quantiles(h.detach(), taus)
        q_taus_prime = self._compute_quantiles(h.detach(), taus_prime)
        batch_steps = torch.arange(h.shape[0])
        # (batch, n_quantiles - 1)
        q_taus = q_taus[batch_steps, action.view(-1)][:, :-1]
        # (batch, n_quantiles)
        q_taus_prime = q_taus_prime[batch_steps, action.view(-1)]

        # compute gradients
        proposal_grad = 2 * q_taus - q_taus_prime[:, :-1] - q_taus_prime[:, 1:]

        return proposal_grad.sum(dim=1)

    def compute_target(
        self, x: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        h = self._encoder(x)
        _, _, taus_prime, _ = self._make_fqf_taus(h)
        quantiles = self._compute_quantiles(h, taus_prime.detach())
        if action is None:
            return quantiles
        return pick_quantile_value_by_action(quantiles, action)

    @property
    def action_size(self) -> int:
        return self._action_size

    @property
    def encoder(self) -> Encoder:
        return self._encoder


class ContinuousFQFQFunction(FQFQFunction, ContinuousQFunction):
    _action_size: int
    _entropy_coeff: float
    _encoder: EncoderWithAction
    _fc: nn.Linear

    def __init__(
        self,
        encoder: EncoderWithAction,
        n_quantiles: int,
        embed_size: int,
        entropy_coeff: float = 0.0,
    ):
        super().__init__(n_quantiles, embed_size, encoder.get_feature_size())
        self._encoder = encoder
        self._action_size = encoder.action_size
        self._fc = nn.Linear(encoder.get_feature_size(), 1)
        self._entropy_coeff = entropy_coeff

    def _compute_quantiles(
        self, h: torch.Tensor, taus: torch.Tensor
    ) -> torch.Tensor:
        # element-wise product on feature and phi (batch, quantile, feature)
        prod = self._compute_last_feature(h, taus)
        # (batch, quantile, feature) -> (batch, quantile)
        return cast(torch.Tensor, self._fc(prod)).view(h.shape[0], -1)

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        h = self._encoder(x, action)
        taus, taus_minus, taus_prime, _ = self._make_fqf_taus(h)
        quantiles = self._compute_quantiles(h, taus_prime.detach())
        weight = (taus - taus_minus).detach()
        return (weight * quantiles).sum(dim=1, keepdim=True)

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
        taus, _, taus_prime, entropies = self._make_fqf_taus(h)
        quantiles_t = self._compute_quantiles(h, taus_prime.detach())

        quantile_loss = self._compute_quantile_loss(
            quantiles_t=quantiles_t,
            rew_tp1=rew_tp1,
            q_tp1=q_tp1,
            ter_tp1=ter_tp1,
            taus=taus_prime.detach(),
            gamma=gamma,
        )

        # compute proposal network loss
        # original paper explicitly separates the optimization process
        # but, it's combined here
        proposal_loss = self._compute_proposal_loss(h, taus, taus_prime)
        proposal_params = list(self._proposal.parameters())
        proposal_grads = torch.autograd.grad(
            outputs=proposal_loss.mean(),
            inputs=proposal_params,
            retain_graph=True,
        )
        # directly apply gradients
        for param, grad in zip(list(proposal_params), proposal_grads):
            param.grad = 1e-4 * grad

        loss = quantile_loss - self._entropy_coeff * entropies

        return compute_reduce(loss, reduction)

    def _compute_proposal_loss(
        self, h: torch.Tensor, taus: torch.Tensor, taus_prime: torch.Tensor
    ) -> torch.Tensor:
        # (batch, n_quantiles - 1)
        q_taus = self._compute_quantiles(h.detach(), taus)[:, :-1]
        # (batch, n_quantiles)
        q_taus_prime = self._compute_quantiles(h.detach(), taus_prime)

        # compute gradients
        proposal_grad = 2 * q_taus - q_taus_prime[:, :-1] - q_taus_prime[:, 1:]
        return proposal_grad.sum(dim=1)

    def compute_target(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        h = self._encoder(x, action)
        _, _, taus_prime, _ = self._make_fqf_taus(h)
        return self._compute_quantiles(h, taus_prime.detach())

    @property
    def action_size(self) -> int:
        return self._action_size

    @property
    def encoder(self) -> EncoderWithAction:
        return self._encoder

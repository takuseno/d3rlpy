import dataclasses
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

from ...torch_utility import get_batch_size, get_device
from ...types import TorchObservation
from .encoders import EncoderWithAction
from .policies import (
    CategoricalPolicy,
    DeterministicPolicy,
    NormalPolicy,
    build_gaussian_distribution,
)

__all__ = [
    "VAEEncoder",
    "VAEDecoder",
    "forward_vae_sample",
    "forward_vae_sample_n",
    "compute_vae_error",
    "compute_discrete_imitation_loss",
    "compute_deterministic_imitation_loss",
    "compute_stochastic_imitation_loss",
    "ImitationLoss",
    "DiscreteImitationLoss",
]


class VAEEncoder(nn.Module):  # type: ignore
    _encoder: EncoderWithAction
    _mu: nn.Module
    _logstd: nn.Module
    _min_logstd: float
    _max_logstd: float
    _latent_size: int

    def __init__(
        self,
        encoder: EncoderWithAction,
        hidden_size: int,
        latent_size: int,
        min_logstd: float = -20.0,
        max_logstd: float = 2.0,
    ):
        super().__init__()
        self._encoder = encoder
        self._mu = nn.Linear(hidden_size, latent_size)
        self._logstd = nn.Linear(hidden_size, latent_size)
        self._min_logstd = min_logstd
        self._max_logstd = max_logstd
        self._latent_size = latent_size

    def forward(self, x: TorchObservation, action: torch.Tensor) -> Normal:
        h = self._encoder(x, action)
        mu = self._mu(h)
        logstd = self._logstd(h)
        clipped_logstd = logstd.clamp(self._min_logstd, self._max_logstd)
        return Normal(mu, clipped_logstd.exp())

    def __call__(self, x: TorchObservation, action: torch.Tensor) -> Normal:
        return super().__call__(x, action)

    @property
    def latent_size(self) -> int:
        return self._latent_size


class VAEDecoder(nn.Module):  # type: ignore
    _encoder: EncoderWithAction
    _fc: nn.Linear
    _action_size: int

    def __init__(
        self, encoder: EncoderWithAction, hidden_size: int, action_size: int
    ):
        super().__init__()
        self._encoder = encoder
        self._fc = nn.Linear(hidden_size, action_size)
        self._action_size = action_size

    def forward(
        self, x: TorchObservation, latent: torch.Tensor, with_squash: bool
    ) -> torch.Tensor:
        h = self._encoder(x, latent)
        if with_squash:
            return self._fc(h)
        return torch.tanh(self._fc(h))

    def __call__(
        self,
        x: TorchObservation,
        latent: torch.Tensor,
        with_squash: bool = True,
    ) -> torch.Tensor:
        return super().__call__(x, latent, with_squash)

    @property
    def action_size(self) -> int:
        return self._action_size


def forward_vae_sample(
    vae_decoder: VAEDecoder,
    x: TorchObservation,
    latent_size: int,
    with_squash: bool = True,
) -> torch.Tensor:
    batch_size = get_batch_size(x)
    latent = torch.randn((batch_size, latent_size), device=get_device(x))
    # to prevent extreme numbers
    return vae_decoder(x, latent.clamp(-0.5, 0.5), with_squash=with_squash)


def forward_vae_sample_n(
    vae_decoder: VAEDecoder,
    x: TorchObservation,
    latent_size: int,
    n: int,
    with_squash: bool = True,
) -> torch.Tensor:
    batch_size = get_batch_size(x)
    flat_latent_shape = (n * batch_size, latent_size)
    flat_latent = torch.randn(flat_latent_shape, device=get_device(x))
    # to prevent extreme numbers
    clipped_latent = flat_latent.clamp(-0.5, 0.5)

    if isinstance(x, torch.Tensor):
        # (batch, obs) -> (n, batch, obs)
        repeated_x = x.expand((n, *x.shape))
        # (n, batch, obs) -> (n *  batch, obs)
        flat_x = repeated_x.reshape(-1, *x.shape[1:])
    else:
        # (batch, obs) -> (n, batch, obs)
        repeated_x = [_x.expand((n, *_x.shape)) for _x in x]
        # (n, batch, obs) -> (n *  batch, obs)
        flat_x = [_x.reshape(-1, *_x.shape[2:]) for _x in repeated_x]

    flat_actions = vae_decoder(flat_x, clipped_latent, with_squash=with_squash)

    # (n * batch, action) -> (n, batch, action)
    actions = flat_actions.view(n, batch_size, -1)

    # (n, batch, action) -> (batch, n, action)
    return actions.transpose(0, 1)


def compute_vae_error(
    vae_encoder: VAEEncoder,
    vae_decoder: VAEDecoder,
    x: TorchObservation,
    action: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    dist = vae_encoder(x, action)
    kl_loss = kl_divergence(dist, Normal(0.0, 1.0)).mean()
    y = vae_decoder(x, dist.rsample())
    return F.mse_loss(y, action) + cast(torch.Tensor, beta * kl_loss)


@dataclasses.dataclass(frozen=True)
class ImitationLoss:
    loss: torch.Tensor


@dataclasses.dataclass(frozen=True)
class DiscreteImitationLoss(ImitationLoss):
    imitation_loss: torch.Tensor
    regularization_loss: torch.Tensor


def compute_discrete_imitation_loss(
    policy: CategoricalPolicy,
    x: TorchObservation,
    action: torch.Tensor,
    beta: float,
) -> DiscreteImitationLoss:
    dist = policy(x)
    penalty = (dist.logits**2).mean()
    log_probs = F.log_softmax(dist.logits, dim=1)
    imitation_loss = F.nll_loss(log_probs, action.view(-1))
    regularization_loss = beta * penalty
    return DiscreteImitationLoss(
        loss=imitation_loss + regularization_loss,
        imitation_loss=imitation_loss,
        regularization_loss=regularization_loss,
    )


def compute_deterministic_imitation_loss(
    policy: DeterministicPolicy, x: TorchObservation, action: torch.Tensor
) -> ImitationLoss:
    return ImitationLoss(loss=F.mse_loss(policy(x).squashed_mu, action))


def compute_stochastic_imitation_loss(
    policy: NormalPolicy, x: TorchObservation, action: torch.Tensor
) -> ImitationLoss:
    dist = build_gaussian_distribution(policy(x))
    return ImitationLoss(loss=F.mse_loss(dist.sample(), action))

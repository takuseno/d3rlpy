from abc import ABCMeta, abstractmethod
from typing import Tuple, cast

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

from .encoders import Encoder, EncoderWithAction

__all__ = [
    "VAEEncoder",
    "VAEDecoder",
    "ConditionalVAE",
    "forward_vae_encode",
    "forward_vae_decode",
    "forward_vae_sample",
    "forward_vae_sample_n",
    "compute_vae_error",
    "Imitator",
    "DiscreteImitator",
    "DeterministicRegressor",
    "ProbablisticRegressor",
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

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> Normal:
        h = self._encoder(x, action)
        mu = self._mu(h)
        logstd = self._logstd(h)
        clipped_logstd = logstd.clamp(self._min_logstd, self._max_logstd)
        return Normal(mu, clipped_logstd.exp())

    def __call__(self, x: torch.Tensor, action: torch.Tensor) -> Normal:
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
        self, x: torch.Tensor, latent: torch.Tensor, with_squash: bool
    ) -> torch.Tensor:
        h = self._encoder(x, latent)
        if with_squash:
            return self._fc(h)
        return torch.tanh(self._fc(h))

    def __call__(
        self, x: torch.Tensor, latent: torch.Tensor, with_squash: bool = True
    ) -> torch.Tensor:
        return super().__call__(x, latent, with_squash)

    @property
    def action_size(self) -> int:
        return self._action_size


class ConditionalVAE(nn.Module):  # type: ignore
    _encoder: VAEEncoder
    _decoder: VAEDecoder
    _beta: float

    def __init__(self, encoder: VAEEncoder, decoder: VAEDecoder):
        super().__init__()
        self._encoder = encoder
        self._decoder = decoder

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        dist = self._encoder(x, action)
        return self._decoder(x, dist.rsample())

    def __call__(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, super().__call__(x, action))

    @property
    def encoder(self) -> VAEEncoder:
        return self._encoder

    @property
    def decoder(self) -> VAEDecoder:
        return self._decoder


def forward_vae_encode(
    vae: ConditionalVAE, x: torch.Tensor, action: torch.Tensor
) -> Normal:
    return vae.encoder(x, action)


def forward_vae_decode(
    vae: ConditionalVAE, x: torch.Tensor, latent: torch.Tensor
) -> torch.Tensor:
    return vae.decoder(x, latent)


def forward_vae_sample(
    vae: ConditionalVAE, x: torch.Tensor, with_squash: bool = True
) -> torch.Tensor:
    latent = torch.randn((x.shape[0], vae.encoder.latent_size), device=x.device)
    # to prevent extreme numbers
    return vae.decoder(x, latent.clamp(-0.5, 0.5), with_squash=with_squash)


def forward_vae_sample_n(
    vae: ConditionalVAE, x: torch.Tensor, n: int, with_squash: bool = True
) -> torch.Tensor:
    flat_latent_shape = (n * x.shape[0], vae.encoder.latent_size)
    flat_latent = torch.randn(flat_latent_shape, device=x.device)
    # to prevent extreme numbers
    clipped_latent = flat_latent.clamp(-0.5, 0.5)

    # (batch, obs) -> (n, batch, obs)
    repeated_x = x.expand((n, *x.shape))
    # (n, batch, obs) -> (n *  batch, obs)
    flat_x = repeated_x.reshape(-1, *x.shape[1:])

    flat_actions = vae.decoder(flat_x, clipped_latent, with_squash=with_squash)

    # (n * batch, action) -> (n, batch, action)
    actions = flat_actions.view(n, x.shape[0], -1)

    # (n, batch, action) -> (batch, n, action)
    return actions.transpose(0, 1)


def compute_vae_error(
    vae: ConditionalVAE, x: torch.Tensor, action: torch.Tensor, beta: float
) -> torch.Tensor:
    dist = vae.encoder(x, action)
    kl_loss = kl_divergence(dist, Normal(0.0, 1.0)).mean()
    y = vae.decoder(x, dist.rsample())
    return F.mse_loss(y, action) + cast(torch.Tensor, beta * kl_loss)


class Imitator(nn.Module, metaclass=ABCMeta):  # type: ignore
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, super().__call__(x))

    @abstractmethod
    def compute_error(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def encoder(self) -> Encoder:
        pass


class DiscreteImitator(Imitator):
    _encoder: Encoder
    _beta: float
    _fc: nn.Linear

    def __init__(
        self, encoder: Encoder, hidden_size: int, action_size: int, beta: float
    ):
        super().__init__()
        self._encoder = encoder
        self._beta = beta
        self._fc = nn.Linear(hidden_size, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.compute_log_probs_with_logits(x)[0]

    def compute_log_probs_with_logits(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self._encoder(x)
        logits = self._fc(h)
        log_probs = F.log_softmax(logits, dim=1)
        return log_probs, logits

    def compute_error(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        log_probs, logits = self.compute_log_probs_with_logits(x)
        penalty = (logits**2).mean()
        return F.nll_loss(log_probs, action.view(-1)) + self._beta * penalty

    @property
    def encoder(self) -> Encoder:
        return self._encoder


class DeterministicRegressor(Imitator):
    _encoder: Encoder
    _fc: nn.Linear

    def __init__(self, encoder: Encoder, hidden_size: int, action_size: int):
        super().__init__()
        self._encoder = encoder
        self._fc = nn.Linear(hidden_size, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._encoder(x)
        h = self._fc(h)
        return torch.tanh(h)

    def compute_error(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        return F.mse_loss(self.forward(x), action)

    @property
    def encoder(self) -> Encoder:
        return self._encoder


class ProbablisticRegressor(Imitator):
    _min_logstd: float
    _max_logstd: float
    _encoder: Encoder
    _mu: nn.Linear
    _logstd: nn.Linear

    def __init__(
        self,
        encoder: Encoder,
        hidden_size: int,
        action_size: int,
        min_logstd: float,
        max_logstd: float,
    ):
        super().__init__()
        self._min_logstd = min_logstd
        self._max_logstd = max_logstd
        self._encoder = encoder
        self._mu = nn.Linear(hidden_size, action_size)
        self._logstd = nn.Linear(hidden_size, action_size)

    def dist(self, x: torch.Tensor) -> Normal:
        h = self._encoder(x)
        mu = self._mu(h)
        logstd = self._logstd(h)
        clipped_logstd = logstd.clamp(self._min_logstd, self._max_logstd)
        return Normal(mu, clipped_logstd.exp())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._encoder(x)
        mu = self._mu(h)
        return torch.tanh(mu)

    def sample_n(self, x: torch.Tensor, n: int) -> torch.Tensor:
        dist = self.dist(x)
        actions = cast(torch.Tensor, dist.rsample((n,)))
        # (n, batch, action) -> (batch, n, action)
        return actions.transpose(0, 1)

    def compute_error(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        dist = self.dist(x)
        return F.mse_loss(torch.tanh(dist.rsample()), action)

    @property
    def encoder(self) -> Encoder:
        return self._encoder

import torch

from ...models.torch import (
    Policy,
    VAEDecoder,
    VAEEncoder,
    build_gaussian_distribution,
    build_squashed_gaussian_distribution,
    compute_vae_error,
)
from ...torch_utility import TorchMiniBatch
from ...types import TorchObservation
from .functional import ActionSampler

__all__ = [
    "DeterministicContinuousActionSampler",
    "GaussianContinuousActionSampler",
    "SquashedGaussianContinuousActionSampler",
    "VAELossFn",
]


class DeterministicContinuousActionSampler(ActionSampler):
    def __init__(self, policy: Policy):
        self._policy = policy

    def __call__(self, x: TorchObservation) -> torch.Tensor:
        action = self._policy(x)
        return action.squashed_mu


class GaussianContinuousActionSampler(ActionSampler):
    def __init__(self, policy: Policy):
        self._policy = policy

    def __call__(self, x: TorchObservation) -> torch.Tensor:
        dist = build_gaussian_distribution(self._policy(x))
        return dist.sample()


class SquashedGaussianContinuousActionSampler(ActionSampler):
    def __init__(self, policy: Policy):
        self._policy = policy

    def __call__(self, x: TorchObservation) -> torch.Tensor:
        dist = build_squashed_gaussian_distribution(self._policy(x))
        return dist.sample()


class VAELossFn:
    def __init__(
        self, vae_encoder: VAEEncoder, vae_decoder: VAEDecoder, kl_weight: float
    ):
        self._vae_encoder = vae_encoder
        self._vae_decoder = vae_decoder
        self._kl_weight = kl_weight

    def __call__(self, batch: TorchMiniBatch) -> torch.Tensor:
        return compute_vae_error(
            vae_encoder=self._vae_encoder,
            vae_decoder=self._vae_decoder,
            x=batch.observations,
            action=batch.actions,
            beta=self._kl_weight,
        )

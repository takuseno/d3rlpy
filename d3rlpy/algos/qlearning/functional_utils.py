import torch

from ...models.torch import Policy, build_squashed_gaussian_distribution, build_gaussian_distribution
from ...types import TorchObservation
from .functional import ActionSampler

__all__ = ["DeterministicContinuousActionSampler", "GaussianContinuousActionSampler", "SquashedGaussianContinuousActionSampler"]


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

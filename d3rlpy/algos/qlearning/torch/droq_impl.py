import torch

from . import SACImpl
from ....models.torch import  build_squashed_gaussian_distribution
from ....torch_utility import TorchMiniBatch

__all__ = ["DroQImpl"]


class DroQImpl(SACImpl):
    def compute_actor_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        dist = build_squashed_gaussian_distribution(
            self._modules.policy(batch.observations)
        )
        action, log_prob = dist.sample_with_log_prob()
        entropy = self._modules.log_temp().exp() * log_prob
        q_t = self._q_func_forwarder.compute_expected_q(
            # Use "mean" (line 10 of Algorithm 2 in the paper)
            batch.observations, action, "mean"
        )
        return (entropy - q_t).mean()


# (TODO IF VALID) class DiscreteDroQImpl

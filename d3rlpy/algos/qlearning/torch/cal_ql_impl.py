import torch

from ....types import TorchObservation
from .cql_impl import CQLCriticLossFn

__all__ = ["CalQLCriticLossFn"]


class CalQLCriticLossFn(CQLCriticLossFn):
    def _compute_policy_is_values(
        self,
        policy_obs: TorchObservation,
        value_obs: TorchObservation,
        returns_to_go: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values, log_probs = super()._compute_policy_is_values(
            policy_obs=policy_obs,
            value_obs=value_obs,
            returns_to_go=returns_to_go,
        )
        return torch.maximum(values, returns_to_go.view(1, -1, 1)), log_probs

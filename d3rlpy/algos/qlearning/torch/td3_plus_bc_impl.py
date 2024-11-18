# pylint: disable=too-many-ancestors
import dataclasses

import torch

from ....models.torch import ContinuousEnsembleQFunctionForwarder, Policy
from ....torch_utility import TorchMiniBatch
from .ddpg_impl import DDPGBaseActorLoss, DDPGBaseActorLossFn

__all__ = ["TD3PlusBCActorLoss", "TD3PlusBCActorLossFn"]


@dataclasses.dataclass(frozen=True)
class TD3PlusBCActorLoss(DDPGBaseActorLoss):
    bc_loss: torch.Tensor


class TD3PlusBCActorLossFn(DDPGBaseActorLossFn):
    def __init__(self, q_func_forwarder: ContinuousEnsembleQFunctionForwarder, policy: Policy, alpha: float):
        self._q_func_forwarder = q_func_forwarder
        self._policy = policy
        self._alpha = alpha

    def __call__(self, batch: TorchMiniBatch) -> TD3PlusBCActorLoss:
        action = self._policy(batch.observations)
        q_t = self._q_func_forwarder.compute_expected_q(
            batch.observations, action.squashed_mu, "none"
        )[0]
        lam = self._alpha / (q_t.abs().mean()).detach()
        bc_loss = ((batch.actions - action.squashed_mu) ** 2).mean()
        return TD3PlusBCActorLoss(
            actor_loss=lam * -q_t.mean() + bc_loss, bc_loss=bc_loss
        )

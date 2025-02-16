import dataclasses

import torch
from sklearn.neighbors import NearestNeighbors

from ....models.torch import (
    ContinuousEnsembleQFunctionForwarder,
    Policy,
)
from ....torch_utility import TorchMiniBatch
from .ddpg_impl import DDPGBaseActorLoss, DDPGBaseActorLossFn

__all__ = ["PRDCActorLossFn", "PRDCActorLoss"]


@dataclasses.dataclass(frozen=True)
class PRDCActorLoss(DDPGBaseActorLoss):
    dc_loss: torch.Tensor


class PRDCActorLossFn(DDPGBaseActorLossFn):
    def __init__(
        self,
        policy: Policy,
        q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        nbsr: NearestNeighbors,
        alpha: float,
        beta: float,
        action_size: int,
    ):
        self._policy = policy
        self._q_func_forwarder = q_func_forwarder
        self._nbsr = nbsr
        self._alpha = alpha
        self._beta = beta
        self._action_size = action_size

    def __call__(self, batch: TorchMiniBatch) -> PRDCActorLoss:
        assert isinstance(
            batch.observations, torch.Tensor
        ), "PRDC only supports non-tuple observations."
        action = self._policy(batch.observations)
        q_t = self._q_func_forwarder.compute_expected_q(
            batch.observations, action.squashed_mu, "none"
        )[0]
        lam = self._alpha / (q_t.abs().mean()).detach()
        key = (
            torch.cat(
                [torch.mul(batch.observations, self._beta), action.squashed_mu],
                dim=-1,
            )
            .detach()
            .cpu()
            .numpy()
        )
        idx = self._nbsr.kneighbors(key, n_neighbors=1, return_distance=False)
        nearest_neighbor = torch.tensor(
            self._nbsr._fit_X[idx][:, :, -self._action_size :],
            device=action.squashed_mu.device,
            dtype=action.squashed_mu.dtype,
        ).squeeze(dim=1)
        dc_loss = torch.nn.functional.mse_loss(
            action.squashed_mu, nearest_neighbor
        )
        return PRDCActorLoss(
            actor_loss=lam * -q_t.mean() + dc_loss, dc_loss=dc_loss
        )

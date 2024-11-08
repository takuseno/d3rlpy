# pylint: disable=too-many-ancestors
import dataclasses

import torch
from sklearn.neighbors import NearestNeighbors

from ....models.torch import ActionOutput, ContinuousEnsembleQFunctionForwarder
from ....torch_utility import TorchMiniBatch
from ....types import Shape
from .ddpg_impl import DDPGBaseActorLoss, DDPGModules
from .td3_impl import TD3Impl

__all__ = ["PRDCImpl"]


@dataclasses.dataclass(frozen=True)
class PRDCActorLoss(DDPGBaseActorLoss):
    dc_loss: torch.Tensor


class PRDCImpl(TD3Impl):
    _alpha: float
    _beta: float
    _nbsr: NearestNeighbors

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: DDPGModules,
        q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        targ_q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        gamma: float,
        tau: float,
        target_smoothing_sigma: float,
        target_smoothing_clip: float,
        alpha: float,
        beta: float,
        update_actor_interval: int,
        compiled: bool,
        nbsr: NearestNeighbors,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            gamma=gamma,
            tau=tau,
            target_smoothing_sigma=target_smoothing_sigma,
            target_smoothing_clip=target_smoothing_clip,
            update_actor_interval=update_actor_interval,
            compiled=compiled,
            device=device,
        )
        self._alpha = alpha
        self._beta = beta
        self._nbsr = nbsr

    def compute_actor_loss(
        self, batch: TorchMiniBatch, action: ActionOutput
    ) -> PRDCActorLoss:
        q_t = self._q_func_forwarder.compute_expected_q(
            batch.observations, action.squashed_mu, "none"
        )[0]
        lam = self._alpha / (q_t.abs().mean()).detach()
        key = (
            torch.cat(
                [torch.mul(batch.observations, self._beta), action.squashed_mu], dim=-1
            )
            .detach()
            .cpu()
            .numpy()
        )
        idx = self._nbsr.kneighbors(key, n_neighbors=1, return_distance=False)
        nearest_neightbour = torch.tensor(
            self._nbsr._fit_X[idx][:, :, -self.action_size :],
            device=self.device,
            dtype=action.squashed_mu.dtype,
        ).squeeze(dim=1)
        dc_loss = torch.nn.functional.mse_loss(action.squashed_mu, nearest_neightbour)
        return PRDCActorLoss(actor_loss=lam * -q_t.mean() + dc_loss, dc_loss=dc_loss)

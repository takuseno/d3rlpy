# pylint: disable=too-many-ancestors
import dataclasses

import torch

from ....models.torch import ActionOutput, ContinuousEnsembleQFunctionForwarder
from ....torch_utility import TorchMiniBatch
from ....types import Shape
from .ddpg_impl import DDPGBaseActorLoss, DDPGModules
from .td3_impl import TD3Impl

__all__ = ["TD3PlusBCImpl"]


@dataclasses.dataclass(frozen=True)
class TD3PlusBCActorLoss(DDPGBaseActorLoss):
    bc_loss: torch.Tensor


class TD3PlusBCImpl(TD3Impl):
    _alpha: float

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
        update_actor_interval: int,
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
            device=device,
        )
        self._alpha = alpha

    def compute_actor_loss(
        self, batch: TorchMiniBatch, action: ActionOutput, grad_step: int
    ) -> TD3PlusBCActorLoss:
        q_t = self._q_func_forwarder.compute_expected_q(
            batch.observations, action.squashed_mu, "none"
        )[0]
        lam = self._alpha / (q_t.abs().mean()).detach()
        bc_loss = ((batch.actions - action.squashed_mu) ** 2).mean()
        return TD3PlusBCActorLoss(
            actor_loss=lam * -q_t.mean() + bc_loss, bc_loss=bc_loss
        )

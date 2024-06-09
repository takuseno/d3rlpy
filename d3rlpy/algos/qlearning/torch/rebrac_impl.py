# pylint: disable=too-many-ancestors
import torch

from ....models.torch import ActionOutput, ContinuousEnsembleQFunctionForwarder
from ....torch_utility import TorchMiniBatch
from ....types import Shape
from .ddpg_impl import DDPGModules
from .td3_impl import TD3Impl
from .td3_plus_bc_impl import TD3PlusBCActorLoss

__all__ = ["ReBRACImpl"]


class ReBRACImpl(TD3Impl):
    _actor_beta: float
    _critic_beta: float

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
        actor_beta: float,
        critic_beta: float,
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
        self._actor_beta = actor_beta
        self._critic_beta = critic_beta

    def compute_actor_loss(
        self, batch: TorchMiniBatch, action: ActionOutput
    ) -> TD3PlusBCActorLoss:
        q_t = self._q_func_forwarder.compute_expected_q(
            batch.observations,
            action.squashed_mu,
            reduction="min",
        )
        lam = 1 / (q_t.abs().mean()).detach()
        bc_loss = ((batch.actions - action.squashed_mu) ** 2).mean()
        return TD3PlusBCActorLoss(
            actor_loss=lam * -q_t.mean() + self._actor_beta * bc_loss,
            bc_loss=bc_loss,
        )

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            action = self._modules.targ_policy(batch.next_observations)
            # smoothing target
            noise = torch.randn(action.mu.shape, device=batch.device)
            scaled_noise = self._target_smoothing_sigma * noise
            clipped_noise = scaled_noise.clamp(
                -self._target_smoothing_clip, self._target_smoothing_clip
            )
            smoothed_action = action.squashed_mu + clipped_noise
            clipped_action = smoothed_action.clamp(-1.0, 1.0)
            next_q = self._targ_q_func_forwarder.compute_target(
                batch.next_observations,
                clipped_action,
                reduction="min",
            )

            # BRAC reguralization
            bc_penalty = ((clipped_action - batch.next_actions) ** 2).sum(
                dim=1, keepdim=True
            )

            return next_q - self._critic_beta * bc_penalty

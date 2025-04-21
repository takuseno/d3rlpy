import torch

from ....models.torch import (
    ContinuousEnsembleQFunctionForwarder,
    Policy,
)
from ....torch_utility import TorchMiniBatch
from .ddpg_impl import DDPGBaseActorLossFn
from .td3_impl import TD3CriticLossFn
from .td3_plus_bc_impl import TD3PlusBCActorLoss

__all__ = ["ReBRACCriticLossFn", "ReBRACActorLossFn"]


class ReBRACCriticLossFn(TD3CriticLossFn):
    def __init__(
        self,
        q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        targ_q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        targ_policy: Policy,
        gamma: float,
        target_smoothing_sigma: float,
        target_smoothing_clip: float,
        critic_beta: float,
    ):
        super().__init__(
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            targ_policy=targ_policy,
            gamma=gamma,
            target_smoothing_sigma=target_smoothing_sigma,
            target_smoothing_clip=target_smoothing_clip,
        )
        self._critic_beta = critic_beta

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            action = self._targ_policy(batch.next_observations)
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


class ReBRACActorLossFn(DDPGBaseActorLossFn):
    def __init__(
        self,
        policy: Policy,
        q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        actor_beta: float,
    ):
        self._policy = policy
        self._q_func_forwarder = q_func_forwarder
        self._actor_beta = actor_beta

    def __call__(self, batch: TorchMiniBatch) -> TD3PlusBCActorLoss:
        action = self._policy(batch.observations)
        q_t = self._q_func_forwarder.compute_expected_q(
            batch.observations,
            action.squashed_mu,
            reduction="min",
        )
        lam = 1 / (q_t.abs().mean()).detach()
        bc_loss = ((batch.actions - action.squashed_mu) ** 2).sum(
            dim=1, keepdim=True
        )
        return TD3PlusBCActorLoss(
            actor_loss=(lam * -q_t + self._actor_beta * bc_loss).mean(),
            bc_loss=bc_loss.mean(),
        )

import torch
from torch import nn

from ....models.torch import ContinuousEnsembleQFunctionForwarder, Policy
from ....optimizers import OptimizerWrapper
from ....torch_utility import TorchMiniBatch
from ....dataclass_utils import asdict_as_float
from .ddpg_impl import DDPGCriticLossFn, DDPGUpdater, DDPGBaseCriticLossFn, DDPGBaseActorLossFn

__all__ = ["TD3CriticLossFn", "TD3Updater"]


class TD3CriticLossFn(DDPGCriticLossFn):
    def __init__(
        self,
        q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        targ_q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        targ_policy: Policy,
        gamma: float,
        target_smoothing_sigma: float,
        target_smoothing_clip: float,
    ):
        super().__init__(
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            targ_policy=targ_policy,
            gamma=gamma,
        )
        self._target_smoothing_sigma = target_smoothing_sigma
        self._target_smoothing_clip = target_smoothing_clip

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
            return self._targ_q_func_forwarder.compute_target(
                batch.next_observations,
                clipped_action,
                reduction="min",
            )


class TD3Updater(DDPGUpdater):
    def __init__(
        self,
        q_funcs: nn.ModuleList,
        targ_q_funcs: nn.ModuleList,
        policy: Policy,
        targ_policy: Policy,
        critic_optim: OptimizerWrapper,
        actor_optim: OptimizerWrapper,
        critic_loss_fn: DDPGBaseCriticLossFn,
        actor_loss_fn: DDPGBaseActorLossFn,
        tau: float,
        update_actor_interval: int,
        compiled: bool,
    ):
        super().__init__(
            q_funcs=q_funcs,
            targ_q_funcs=targ_q_funcs,
            policy=policy,
            targ_policy=targ_policy,
            critic_optim=critic_optim,
            actor_optim=actor_optim,
            critic_loss_fn=critic_loss_fn,
            actor_loss_fn=actor_loss_fn,
            tau=tau,
            compiled=compiled,
        )
        self._update_actor_interval = update_actor_interval

    def __call__(self, batch: TorchMiniBatch, grad_step: int) -> dict[str, float]:
        metrics = {}

        # update critic
        critic_loss = self._compute_critic_grad(batch)
        self._critic_optim.step()
        metrics.update(asdict_as_float(critic_loss))

        # delayed policy update
        if grad_step % self._update_actor_interval == 0:
            # update actor
            actor_loss = self._compute_actor_grad(batch)
            self._actor_optim.step()
            metrics.update(asdict_as_float(actor_loss))

            # update target networks
            self.update_target()

        return metrics

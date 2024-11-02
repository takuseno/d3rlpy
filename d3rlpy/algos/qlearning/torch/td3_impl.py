from typing import Dict

import torch

from ....models.torch import ContinuousEnsembleQFunctionForwarder
from ....torch_utility import TorchMiniBatch
from ....types import Shape
from .ddpg_impl import DDPGImpl, DDPGModules

__all__ = ["TD3Impl"]


class TD3Impl(DDPGImpl):
    _target_smoothing_sigma: float
    _target_smoothing_clip: float
    _update_actor_interval: int

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
        update_actor_interval: int,
        compile_graph: bool,
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
            compile_graph=compile_graph,
            device=device,
        )
        self._target_smoothing_sigma = target_smoothing_sigma
        self._target_smoothing_clip = target_smoothing_clip
        self._update_actor_interval = update_actor_interval

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
            return self._targ_q_func_forwarder.compute_target(
                batch.next_observations,
                clipped_action,
                reduction="min",
            )

    def inner_update(
        self, batch: TorchMiniBatch, grad_step: int
    ) -> Dict[str, float]:
        metrics = {}

        metrics.update(self.update_critic(batch))

        # delayed policy update
        if grad_step % self._update_actor_interval == 0:
            metrics.update(self.update_actor(batch))
            self.update_critic_target()
            self.update_actor_target()

        return metrics

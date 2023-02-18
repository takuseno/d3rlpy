import torch
from torch.optim import Optimizer

from ....dataset import Shape
from ....models.torch import DeterministicPolicy, EnsembleContinuousQFunction
from ....torch_utility import TorchMiniBatch
from .ddpg_impl import DDPGImpl

__all__ = ["TD3Impl"]


class TD3Impl(DDPGImpl):

    _target_smoothing_sigma: float
    _target_smoothing_clip: float

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        policy: DeterministicPolicy,
        q_func: EnsembleContinuousQFunction,
        actor_optim: Optimizer,
        critic_optim: Optimizer,
        gamma: float,
        tau: float,
        target_smoothing_sigma: float,
        target_smoothing_clip: float,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            policy=policy,
            q_func=q_func,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
            gamma=gamma,
            tau=tau,
            device=device,
        )
        self._target_smoothing_sigma = target_smoothing_sigma
        self._target_smoothing_clip = target_smoothing_clip

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            action = self._targ_policy(batch.next_observations)
            # smoothing target
            noise = torch.randn(action.shape, device=batch.device)
            scaled_noise = self._target_smoothing_sigma * noise
            clipped_noise = scaled_noise.clamp(
                -self._target_smoothing_clip, self._target_smoothing_clip
            )
            smoothed_action = action + clipped_noise
            clipped_action = smoothed_action.clamp(-1.0, 1.0)
            return self._targ_q_func.compute_target(
                batch.next_observations,
                clipped_action,
                reduction="min",
            )

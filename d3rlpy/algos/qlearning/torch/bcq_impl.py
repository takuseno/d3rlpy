import math
from typing import cast

import torch
from torch.optim import Optimizer

from ....dataset import Shape
from ....models.torch import (
    ConditionalVAE,
    DeterministicResidualPolicy,
    DiscreteImitator,
    EnsembleContinuousQFunction,
    EnsembleDiscreteQFunction,
    compute_max_with_n_actions,
    compute_vae_error,
    forward_vae_decode,
)
from ....torch_utility import TorchMiniBatch, train_api
from .ddpg_impl import DDPGBaseImpl
from .dqn_impl import DoubleDQNImpl

__all__ = ["BCQImpl", "DiscreteBCQImpl"]


class BCQImpl(DDPGBaseImpl):
    _lam: float
    _n_action_samples: int
    _action_flexibility: float
    _beta: float
    _policy: DeterministicResidualPolicy
    _targ_policy: DeterministicResidualPolicy
    _imitator: ConditionalVAE
    _imitator_optim: Optimizer

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        policy: DeterministicResidualPolicy,
        q_func: EnsembleContinuousQFunction,
        imitator: ConditionalVAE,
        actor_optim: Optimizer,
        critic_optim: Optimizer,
        imitator_optim: Optimizer,
        gamma: float,
        tau: float,
        lam: float,
        n_action_samples: int,
        action_flexibility: float,
        beta: float,
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
        self._lam = lam
        self._n_action_samples = n_action_samples
        self._action_flexibility = action_flexibility
        self._beta = beta
        self._imitator = imitator
        self._imitator_optim = imitator_optim

    def compute_actor_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        latent = torch.randn(
            batch.observations.shape[0],
            2 * self._action_size,
            device=self._device,
        )
        clipped_latent = latent.clamp(-0.5, 0.5)
        sampled_action = forward_vae_decode(
            vae=self._imitator,
            x=batch.observations,
            latent=clipped_latent,
        )
        action = self._policy(batch.observations, sampled_action)
        return -self._q_func(batch.observations, action.squashed_mu, "none")[
            0
        ].mean()

    @train_api
    def update_imitator(self, batch: TorchMiniBatch) -> float:
        self._imitator_optim.zero_grad()

        loss = compute_vae_error(
            vae=self._imitator,
            x=batch.observations,
            action=batch.actions,
            beta=self._beta,
        )

        loss.backward()
        self._imitator_optim.step()

        return float(loss.cpu().detach().numpy())

    def _repeat_observation(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, *obs_shape) -> (batch_size, n, *obs_shape)
        repeat_shape = (x.shape[0], self._n_action_samples, *x.shape[1:])
        repeated_x = x.view(x.shape[0], 1, *x.shape[1:]).expand(repeat_shape)
        return repeated_x

    def _sample_repeated_action(
        self, repeated_x: torch.Tensor, target: bool = False
    ) -> torch.Tensor:
        # TODO: this seems to be slow with image observation
        flattened_x = repeated_x.reshape(-1, *self.observation_shape)
        # sample latent variable
        latent = torch.randn(
            flattened_x.shape[0], 2 * self._action_size, device=self._device
        )
        clipped_latent = latent.clamp(-0.5, 0.5)
        # sample action
        sampled_action = forward_vae_decode(
            vae=self._imitator,
            x=flattened_x,
            latent=clipped_latent,
        )
        # add residual action
        policy = self._targ_policy if target else self._policy
        action = policy(flattened_x, sampled_action)
        return action.squashed_mu.view(
            -1, self._n_action_samples, self._action_size
        )

    def _predict_value(
        self,
        repeated_x: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        # TODO: this seems to be slow with image observation
        # (batch_size, n, *obs_shape) -> (batch_size * n, *obs_shape)
        flattened_x = repeated_x.reshape(-1, *self.observation_shape)
        # (batch_size, n, action_size) -> (batch_size * n, action_size)
        flattend_action = action.view(-1, self.action_size)
        # estimate values
        return self._q_func(flattened_x, flattend_action, "none")

    def inner_predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: this seems to be slow with image observation
        repeated_x = self._repeat_observation(x)
        action = self._sample_repeated_action(repeated_x)
        values = self._predict_value(repeated_x, action)[0]
        # pick the best (batch_size * n) -> (batch_size,)
        index = values.view(-1, self._n_action_samples).argmax(dim=1)
        return action[torch.arange(action.shape[0]), index]

    def inner_sample_action(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner_predict_best_action(x)

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        # TODO: this seems to be slow with image observation
        with torch.no_grad():
            repeated_x = self._repeat_observation(batch.next_observations)
            actions = self._sample_repeated_action(repeated_x, True)

            values = compute_max_with_n_actions(
                batch.next_observations, actions, self._targ_q_func, self._lam
            )

            return values


class DiscreteBCQImpl(DoubleDQNImpl):
    _action_flexibility: float
    _beta: float
    _imitator: DiscreteImitator

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        q_func: EnsembleDiscreteQFunction,
        imitator: DiscreteImitator,
        optim: Optimizer,
        gamma: float,
        action_flexibility: float,
        beta: float,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            q_func=q_func,
            optim=optim,
            gamma=gamma,
            device=device,
        )
        self._action_flexibility = action_flexibility
        self._beta = beta
        self._imitator = imitator

    def compute_loss(
        self, batch: TorchMiniBatch, q_tpn: torch.Tensor
    ) -> torch.Tensor:
        loss = super().compute_loss(batch, q_tpn)
        imitator_loss = self._imitator.compute_error(
            batch.observations, batch.actions.long()
        )
        return loss + imitator_loss

    def inner_predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        log_probs = self._imitator(x)
        ratio = log_probs - log_probs.max(dim=1, keepdim=True).values
        mask = (ratio > math.log(self._action_flexibility)).float()
        value = self._q_func(x)
        # add a small constant value to deal with the case where the all
        # actions except the min value are masked
        normalized_value = value - value.min(dim=1, keepdim=True).values + 1e-5
        action = (normalized_value * cast(torch.Tensor, mask)).argmax(dim=1)
        return action

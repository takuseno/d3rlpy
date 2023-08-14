import dataclasses
import math
from typing import Dict, cast

import torch
import torch.nn.functional as F
from torch.optim import Optimizer

from ....dataset import Shape
from ....models.torch import (
    CategoricalPolicy,
    ConditionalVAE,
    ContinuousEnsembleQFunctionForwarder,
    DeterministicResidualPolicy,
    DiscreteEnsembleQFunctionForwarder,
    compute_discrete_imitation_loss,
    compute_max_with_n_actions,
    compute_vae_error,
    forward_vae_decode,
)
from ....torch_utility import TorchMiniBatch, soft_sync, train_api
from .ddpg_impl import DDPGBaseImpl, DDPGBaseModules
from .dqn_impl import DoubleDQNImpl, DQNModules

__all__ = ["BCQImpl", "DiscreteBCQImpl", "BCQModules", "DiscreteBCQModules"]


@dataclasses.dataclass(frozen=True)
class BCQModules(DDPGBaseModules):
    policy: DeterministicResidualPolicy
    targ_policy: DeterministicResidualPolicy
    imitator: ConditionalVAE
    imitator_optim: Optimizer


class BCQImpl(DDPGBaseImpl):
    _modules: BCQModules
    _lam: float
    _n_action_samples: int
    _action_flexibility: float
    _beta: float

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: BCQModules,
        q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        targ_q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
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
            modules=modules,
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            gamma=gamma,
            tau=tau,
            device=device,
        )
        self._lam = lam
        self._n_action_samples = n_action_samples
        self._action_flexibility = action_flexibility
        self._beta = beta

    def compute_actor_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        latent = torch.randn(
            batch.observations.shape[0],
            2 * self._action_size,
            device=self._device,
        )
        clipped_latent = latent.clamp(-0.5, 0.5)
        sampled_action = forward_vae_decode(
            vae=self._modules.imitator,
            x=batch.observations,
            latent=clipped_latent,
        )
        action = self._modules.policy(batch.observations, sampled_action)
        value = self._q_func_forwarder.compute_expected_q(
            batch.observations, action.squashed_mu, "none"
        )
        return -value[0].mean()

    @train_api
    def update_imitator(self, batch: TorchMiniBatch) -> Dict[str, float]:
        self._modules.imitator_optim.zero_grad()

        loss = compute_vae_error(
            vae=self._modules.imitator,
            x=batch.observations,
            action=batch.actions,
            beta=self._beta,
        )

        loss.backward()
        self._modules.imitator_optim.step()

        return {"imitator_loss": float(loss.cpu().detach().numpy())}

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
            vae=self._modules.imitator,
            x=flattened_x,
            latent=clipped_latent,
        )
        # add residual action
        policy = self._modules.targ_policy if target else self._modules.policy
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
        return self._q_func_forwarder.compute_expected_q(
            flattened_x, flattend_action, "none"
        )

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
                batch.next_observations,
                actions,
                self._targ_q_func_forwarder,
                self._lam,
            )
            return values

    def update_actor_target(self) -> None:
        soft_sync(self._modules.targ_policy, self._modules.policy, self._tau)


@dataclasses.dataclass(frozen=True)
class DiscreteBCQModules(DQNModules):
    imitator: CategoricalPolicy


class DiscreteBCQImpl(DoubleDQNImpl):
    _modules: DiscreteBCQModules
    _action_flexibility: float
    _beta: float

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: DiscreteBCQModules,
        q_func_forwarder: DiscreteEnsembleQFunctionForwarder,
        targ_q_func_forwarder: DiscreteEnsembleQFunctionForwarder,
        gamma: float,
        action_flexibility: float,
        beta: float,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            gamma=gamma,
            device=device,
        )
        self._action_flexibility = action_flexibility
        self._beta = beta

    def compute_loss(
        self, batch: TorchMiniBatch, q_tpn: torch.Tensor
    ) -> torch.Tensor:
        loss = super().compute_loss(batch, q_tpn)
        imitator_loss = compute_discrete_imitation_loss(
            policy=self._modules.imitator,
            x=batch.observations,
            action=batch.actions.long(),
            beta=self._beta,
        )
        return loss + imitator_loss

    def inner_predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        dist = self._modules.imitator(x)
        log_probs = F.log_softmax(dist.logits, dim=1)
        ratio = log_probs - log_probs.max(dim=1, keepdim=True).values
        mask = (ratio > math.log(self._action_flexibility)).float()
        value = self._q_func_forwarder.compute_expected_q(x)
        # add a small constant value to deal with the case where the all
        # actions except the min value are masked
        normalized_value = value - value.min(dim=1, keepdim=True).values + 1e-5
        action = (normalized_value * cast(torch.Tensor, mask)).argmax(dim=1)
        return action

import dataclasses
import math
from typing import Optional

import torch
import torch.nn.functional as F

from ....models.torch import (
    ContinuousEnsembleQFunctionForwarder,
    DiscreteEnsembleQFunctionForwarder,
    Parameter,
    NormalPolicy,
    get_parameter,
)
from ....optimizers import OptimizerWrapper
from ....torch_utility import (
    TorchMiniBatch,
    expand_and_repeat_recursively,
    flatten_left_recursively,
)
from ....types import Shape, TorchObservation
from .ddpg_impl import DDPGBaseCriticLoss
from .dqn_impl import DoubleDQNLossFn, DQNLoss
from .sac_impl import SACModules, SACCriticLossFn
from .utility import sample_q_values_with_policy

__all__ = ["CQLImpl", "DiscreteCQLLossFn", "CQLModules", "DiscreteCQLLoss"]


@dataclasses.dataclass(frozen=True)
class CQLModules(SACModules):
    log_alpha: Parameter
    alpha_optim: Optional[OptimizerWrapper]


@dataclasses.dataclass(frozen=True)
class CQLCriticLoss(DDPGBaseCriticLoss):
    conservative_loss: torch.Tensor
    alpha: torch.Tensor


class CQLCriticLossFn(SACCriticLossFn):
    _policy: NormalPolicy

    def __init__(
        self,
        q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        targ_q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        policy: NormalPolicy,
        log_temp: Parameter,
        log_alpha: Parameter,
        alpha_optim: Optional[OptimizerWrapper],
        gamma: float,
        n_action_samples: int,
        conservative_weight: float,
        alpha_threshold: float,
        soft_q_backup: bool,
        max_q_backup: bool,
        action_size: int,
    ):
        super().__init__(
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            gamma=gamma,
            log_temp=log_temp,
            policy=policy,
        )
        self._log_alpha = log_alpha
        self._alpha_optim = alpha_optim
        self._n_action_samples = n_action_samples
        self._conservative_weight = conservative_weight
        self._alpha_threshold = alpha_threshold
        self._soft_q_backup = soft_q_backup
        self._max_q_backup = max_q_backup
        self._action_size = action_size

    def __call__(self, batch: TorchMiniBatch) -> DDPGBaseCriticLoss:
        loss = super().__call__(batch)
        conservative_loss = self._compute_conservative_loss(
            obs_t=batch.observations,
            act_t=batch.actions,
            obs_tp1=batch.next_observations,
            returns_to_go=batch.returns_to_go,
        )

        if self._alpha_optim:
            self.update_alpha(conservative_loss.detach())

        # clip for stability
        log_alpha = get_parameter(self._log_alpha)
        clipped_alpha = log_alpha.exp().clamp(0, 1e6)[0][0]
        scaled_conservative_loss = clipped_alpha * conservative_loss

        return CQLCriticLoss(
            critic_loss=loss.critic_loss + scaled_conservative_loss.sum(),
            conservative_loss=scaled_conservative_loss.sum(),
            alpha=clipped_alpha,
        )

    def update_alpha(self, conservative_loss: torch.Tensor) -> None:
        assert self._alpha_optim
        self._alpha_optim.zero_grad()
        log_alpha = get_parameter(self._log_alpha)
        clipped_alpha = log_alpha.exp().clamp(0, 1e6)
        loss = -(clipped_alpha * conservative_loss).mean()
        loss.backward()
        self._alpha_optim.step()

    def _compute_policy_is_values(
        self,
        policy_obs: TorchObservation,
        value_obs: TorchObservation,
        returns_to_go: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return sample_q_values_with_policy(
            policy=self._policy,
            q_func_forwarder=self._q_func_forwarder,
            policy_observations=policy_obs,
            value_observations=value_obs,
            n_action_samples=self._n_action_samples,
            detach_policy_output=True,
        )

    def _compute_random_is_values(
        self, obs: TorchObservation
    ) -> tuple[torch.Tensor, float]:
        # (batch, observation) -> (batch, n, observation)
        repeated_obs = expand_and_repeat_recursively(
            obs, self._n_action_samples
        )
        # (batch, n, observation) -> (batch * n, observation)
        flat_obs = flatten_left_recursively(repeated_obs, dim=1)

        # estimate action-values for actions from uniform distribution
        # uniform distribution between [-1.0, 1.0]
        batch_size = (
            obs.shape[0] if isinstance(obs, torch.Tensor) else obs[0].shape[0]
        )
        flat_shape = (batch_size * self._n_action_samples, self._action_size)
        zero_tensor = torch.zeros(flat_shape, device=self._device)
        random_actions = zero_tensor.uniform_(-1.0, 1.0)
        random_values = self._q_func_forwarder.compute_expected_q(
            flat_obs, random_actions, "none"
        )
        random_values = random_values.view(
            -1, batch_size, self._n_action_samples
        )
        random_log_probs = math.log(0.5**self._action_size)

        # importance sampling
        return random_values, random_log_probs

    def _compute_conservative_loss(
        self,
        obs_t: TorchObservation,
        act_t: torch.Tensor,
        obs_tp1: TorchObservation,
        returns_to_go: torch.Tensor,
    ) -> torch.Tensor:
        policy_values_t, log_probs_t = self._compute_policy_is_values(
            policy_obs=obs_t,
            value_obs=obs_t,
            returns_to_go=returns_to_go,
        )
        policy_values_tp1, log_probs_tp1 = self._compute_policy_is_values(
            policy_obs=obs_tp1,
            value_obs=obs_t,
            returns_to_go=returns_to_go,
        )
        random_values, random_log_probs = self._compute_random_is_values(obs_t)

        # compute logsumexp
        # (n critics, batch, 3 * n samples) -> (n critics, batch, 1)
        target_values = torch.cat(
            [
                policy_values_t - log_probs_t,
                policy_values_tp1 - log_probs_tp1,
                random_values - random_log_probs,
            ],
            dim=2,
        )
        logsumexp = torch.logsumexp(target_values, dim=2, keepdim=True)

        # estimate action-values for data actions
        data_values = self._q_func_forwarder.compute_expected_q(
            obs_t, act_t, "none"
        )

        loss = (logsumexp - data_values).mean(dim=[1, 2])

        return self._conservative_weight * (loss - self._alpha_threshold)

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        if self._soft_q_backup:
            target_value = super().compute_target(batch)
        else:
            with torch.no_grad():
                target_value = self._compute_deterministic_target(batch)
        return target_value

    def _compute_deterministic_target(
        self, batch: TorchMiniBatch
    ) -> torch.Tensor:
        if self._max_q_backup:
            q_values, _ = sample_q_values_with_policy(
                policy=self._policy,
                q_func_forwarder=self._targ_q_func_forwarder,
                policy_observations=batch.next_observations,
                value_observations=batch.next_observations,
                n_action_samples=self._n_action_samples,
                detach_policy_output=True,
            )
            return q_values.min(dim=0).values.max(dim=1, keepdims=True).values
        else:
            action = self._policy(batch.next_observations).squashed_mu
            return self._targ_q_func_forwarder.compute_target(
                batch.next_observations,
                action,
                reduction="min",
            )


class CQLImpl(SACImpl):
    _modules: CQLModules
    _alpha_threshold: float
    _conservative_weight: float
    _n_action_samples: int
    _soft_q_backup: bool
    _max_q_backup: bool

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: CQLModules,
        q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        targ_q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        gamma: float,
        tau: float,
        alpha_threshold: float,
        conservative_weight: float,
        n_action_samples: int,
        soft_q_backup: bool,
        max_q_backup: bool,
        compiled: bool,
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
            compiled=compiled,
            device=device,
        )
        self._alpha_threshold = alpha_threshold
        self._conservative_weight = conservative_weight
        self._n_action_samples = n_action_samples
        self._soft_q_backup = soft_q_backup
        self._max_q_backup = max_q_backup

    def compute_critic_loss(
        self, batch: TorchMiniBatch, q_tpn: torch.Tensor
    ) -> CQLCriticLoss:
        loss = super().compute_critic_loss(batch, q_tpn)
        conservative_loss = self._compute_conservative_loss(
            obs_t=batch.observations,
            act_t=batch.actions,
            obs_tp1=batch.next_observations,
            returns_to_go=batch.returns_to_go,
        )

        if self._modules.alpha_optim:
            self.update_alpha(conservative_loss.detach())

        # clip for stability
        log_alpha = get_parameter(self._modules.log_alpha)
        clipped_alpha = log_alpha.exp().clamp(0, 1e6)[0][0]
        scaled_conservative_loss = clipped_alpha * conservative_loss

        return CQLCriticLoss(
            critic_loss=loss.critic_loss + scaled_conservative_loss.sum(),
            conservative_loss=scaled_conservative_loss.sum(),
            alpha=clipped_alpha,
        )

    def update_alpha(self, conservative_loss: torch.Tensor) -> None:
        assert self._modules.alpha_optim
        self._modules.alpha_optim.zero_grad()
        log_alpha = get_parameter(self._modules.log_alpha)
        clipped_alpha = log_alpha.exp().clamp(0, 1e6)
        loss = -(clipped_alpha * conservative_loss).mean()
        loss.backward()
        self._modules.alpha_optim.step()

    def _compute_policy_is_values(
        self,
        policy_obs: TorchObservation,
        value_obs: TorchObservation,
        returns_to_go: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return sample_q_values_with_policy(
            policy=self._modules.policy,
            q_func_forwarder=self._q_func_forwarder,
            policy_observations=policy_obs,
            value_observations=value_obs,
            n_action_samples=self._n_action_samples,
            detach_policy_output=True,
        )

    def _compute_random_is_values(
        self, obs: TorchObservation
    ) -> tuple[torch.Tensor, float]:
        # (batch, observation) -> (batch, n, observation)
        repeated_obs = expand_and_repeat_recursively(
            obs, self._n_action_samples
        )
        # (batch, n, observation) -> (batch * n, observation)
        flat_obs = flatten_left_recursively(repeated_obs, dim=1)

        # estimate action-values for actions from uniform distribution
        # uniform distribution between [-1.0, 1.0]
        batch_size = (
            obs.shape[0] if isinstance(obs, torch.Tensor) else obs[0].shape[0]
        )
        flat_shape = (batch_size * self._n_action_samples, self._action_size)
        zero_tensor = torch.zeros(flat_shape, device=self._device)
        random_actions = zero_tensor.uniform_(-1.0, 1.0)
        random_values = self._q_func_forwarder.compute_expected_q(
            flat_obs, random_actions, "none"
        )
        random_values = random_values.view(
            -1, batch_size, self._n_action_samples
        )
        random_log_probs = math.log(0.5**self._action_size)

        # importance sampling
        return random_values, random_log_probs

    def _compute_conservative_loss(
        self,
        obs_t: TorchObservation,
        act_t: torch.Tensor,
        obs_tp1: TorchObservation,
        returns_to_go: torch.Tensor,
    ) -> torch.Tensor:
        policy_values_t, log_probs_t = self._compute_policy_is_values(
            policy_obs=obs_t,
            value_obs=obs_t,
            returns_to_go=returns_to_go,
        )
        policy_values_tp1, log_probs_tp1 = self._compute_policy_is_values(
            policy_obs=obs_tp1,
            value_obs=obs_t,
            returns_to_go=returns_to_go,
        )
        random_values, random_log_probs = self._compute_random_is_values(obs_t)

        # compute logsumexp
        # (n critics, batch, 3 * n samples) -> (n critics, batch, 1)
        target_values = torch.cat(
            [
                policy_values_t - log_probs_t,
                policy_values_tp1 - log_probs_tp1,
                random_values - random_log_probs,
            ],
            dim=2,
        )
        logsumexp = torch.logsumexp(target_values, dim=2, keepdim=True)

        # estimate action-values for data actions
        data_values = self._q_func_forwarder.compute_expected_q(
            obs_t, act_t, "none"
        )

        loss = (logsumexp - data_values).mean(dim=[1, 2])

        return self._conservative_weight * (loss - self._alpha_threshold)

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        if self._soft_q_backup:
            target_value = super().compute_target(batch)
        else:
            with torch.no_grad():
                target_value = self._compute_deterministic_target(batch)
        return target_value

    def _compute_deterministic_target(
        self, batch: TorchMiniBatch
    ) -> torch.Tensor:
        if self._max_q_backup:
            q_values, _ = sample_q_values_with_policy(
                policy=self._modules.policy,
                q_func_forwarder=self._targ_q_func_forwarder,
                policy_observations=batch.next_observations,
                value_observations=batch.next_observations,
                n_action_samples=self._n_action_samples,
                detach_policy_output=True,
            )
            return q_values.min(dim=0).values.max(dim=1, keepdims=True).values
        else:
            action = self._modules.policy(batch.next_observations).squashed_mu
            return self._targ_q_func_forwarder.compute_target(
                batch.next_observations,
                action,
                reduction="min",
            )


@dataclasses.dataclass(frozen=True)
class DiscreteCQLLoss(DQNLoss):
    td_loss: torch.Tensor
    conservative_loss: torch.Tensor


class DiscreteCQLLossFn(DoubleDQNLossFn):
    def __init__(
        self,
        action_size: int,
        q_func_forwarder: DiscreteEnsembleQFunctionForwarder,
        targ_q_func_forwarder: DiscreteEnsembleQFunctionForwarder,
        gamma: float,
        alpha: float,
    ):
        super().__init__(q_func_forwarder, targ_q_func_forwarder, gamma)
        self._action_size = action_size
        self._alpha = alpha

    def _compute_conservative_loss(
        self, obs_t: TorchObservation, act_t: torch.Tensor
    ) -> torch.Tensor:
        # compute logsumexp
        values = self._q_func_forwarder.compute_expected_q(obs_t)
        logsumexp = torch.logsumexp(values, dim=1, keepdim=True)

        # estimate action-values under data distribution
        one_hot = F.one_hot(act_t.view(-1), num_classes=self._action_size)
        data_values = (values * one_hot).sum(dim=1, keepdim=True)

        return (logsumexp - data_values).mean()

    def __call__(
        self,
        batch: TorchMiniBatch,
    ) -> DiscreteCQLLoss:
        td_loss = super().__call__(batch).loss
        conservative_loss = self._compute_conservative_loss(
            batch.observations, batch.actions.long()
        )
        loss = td_loss + self._alpha * conservative_loss
        return DiscreteCQLLoss(
            loss=loss, td_loss=td_loss, conservative_loss=conservative_loss
        )

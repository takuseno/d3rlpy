import dataclasses
import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.optim import Optimizer

from ....dataclass_utils import asdict_as_float
from ....dataset import Shape
from ....models.torch import (
    ContinuousEnsembleQFunctionForwarder,
    DiscreteEnsembleQFunctionForwarder,
    Parameter,
    build_squashed_gaussian_distribution,
)
from ....torch_utility import TorchMiniBatch
from .dqn_impl import DoubleDQNImpl, DQNLoss, DQNModules
from .sac_impl import SACImpl, SACModules
from .utility import CriticLoss

__all__ = [
    "CQLImpl",
    "DiscreteCQLImpl",
    "CQLModules",
    "CQLLoss",
]


@dataclasses.dataclass(frozen=True)
class CQLModules(SACModules):
    log_alpha: Parameter
    alpha_optim: Optional[Optimizer]


@dataclasses.dataclass(frozen=True)
class CQLLoss(CriticLoss):
    conservative_loss: torch.Tensor

    def get_loss(self)->torch.Tensor:
        return super().get_loss() + self.conservative_loss


class CQLImpl(SACImpl):
    _modules: CQLModules
    _alpha_threshold: float
    _conservative_weight: float
    _n_action_samples: int
    _soft_q_backup: bool

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
        self._alpha_threshold = alpha_threshold
        self._conservative_weight = conservative_weight
        self._n_action_samples = n_action_samples
        self._soft_q_backup = soft_q_backup

    def compute_critic_loss(
        self, batch: TorchMiniBatch, q_tpn: torch.Tensor
    ) -> CQLLoss:
        td_loss = super().compute_critic_loss(batch, q_tpn).loss
        conservative_loss = self._compute_conservative_loss(
            batch.observations, batch.actions, batch.next_observations
        )
        return CQLLoss(
            td_loss=td_loss,
            conservative_loss=conservative_loss,
        )

    def update_critic(self, batch: TorchMiniBatch) -> Dict[str, float]:
        self._modules.critic_optim.zero_grad()

        q_tpn = self.compute_target(batch)

        loss = self.compute_critic_loss(batch, q_tpn)

        loss.loss.backward()
        self._modules.critic_optim.step()

        return asdict_as_float(loss)

    def update_alpha(self, batch: TorchMiniBatch) -> Dict[str, float]:
        assert self._modules.alpha_optim

        # Q function should be inference mode for stability
        self._modules.q_funcs.eval()

        self._modules.alpha_optim.zero_grad()

        # the original implementation does scale the loss value
        loss = -self._compute_conservative_loss(
            batch.observations, batch.actions, batch.next_observations
        )

        loss.backward()
        self._modules.alpha_optim.step()

        cur_alpha = self._modules.log_alpha().exp().cpu().detach().numpy()[0][0]

        return {
            "alpha_loss": float(loss.cpu().detach().numpy()),
            "alpha": float(cur_alpha),
        }

    def _compute_policy_is_values(
        self, policy_obs: torch.Tensor, value_obs: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            dist = build_squashed_gaussian_distribution(
                self._modules.policy(policy_obs)
            )
            policy_actions, n_log_probs = dist.sample_n_with_log_prob(
                self._n_action_samples
            )

        obs_shape = value_obs.shape

        repeated_obs = value_obs.expand(self._n_action_samples, *obs_shape)
        # (n, batch, observation) -> (batch, n, observation)
        transposed_obs = repeated_obs.transpose(0, 1)
        # (batch, n, observation) -> (batch * n, observation)
        flat_obs = transposed_obs.reshape(-1, *obs_shape[1:])
        # (batch, n, action) -> (batch * n, action)
        flat_policy_acts = policy_actions.reshape(-1, self.action_size)

        # estimate action-values for policy actions
        policy_values = self._q_func_forwarder.compute_expected_q(
            flat_obs, flat_policy_acts, "none"
        )
        policy_values = policy_values.view(
            -1, obs_shape[0], self._n_action_samples
        )
        log_probs = n_log_probs.view(1, -1, self._n_action_samples)

        # importance sampling
        return policy_values - log_probs

    def _compute_random_is_values(self, obs: torch.Tensor) -> torch.Tensor:
        repeated_obs = obs.expand(self._n_action_samples, *obs.shape)
        # (n, batch, observation) -> (batch, n, observation)
        transposed_obs = repeated_obs.transpose(0, 1)
        # (batch, n, observation) -> (batch * n, observation)
        flat_obs = transposed_obs.reshape(-1, *obs.shape[1:])

        # estimate action-values for actions from uniform distribution
        # uniform distribution between [-1.0, 1.0]
        flat_shape = (obs.shape[0] * self._n_action_samples, self._action_size)
        zero_tensor = torch.zeros(flat_shape, device=self._device)
        random_actions = zero_tensor.uniform_(-1.0, 1.0)
        random_values = self._q_func_forwarder.compute_expected_q(
            flat_obs, random_actions, "none"
        )
        random_values = random_values.view(
            -1, obs.shape[0], self._n_action_samples
        )
        random_log_probs = math.log(0.5**self._action_size)

        # importance sampling
        return random_values - random_log_probs

    def _compute_conservative_loss(
        self, obs_t: torch.Tensor, act_t: torch.Tensor, obs_tp1: torch.Tensor
    ) -> torch.Tensor:
        policy_values_t = self._compute_policy_is_values(obs_t, obs_t)
        policy_values_tp1 = self._compute_policy_is_values(obs_tp1, obs_t)
        random_values = self._compute_random_is_values(obs_t)

        # compute logsumexp
        # (n critics, batch, 3 * n samples) -> (n critics, batch, 1)
        target_values = torch.cat(
            [policy_values_t, policy_values_tp1, random_values], dim=2
        )
        logsumexp = torch.logsumexp(target_values, dim=2, keepdim=True)

        # estimate action-values for data actions
        data_values = self._q_func_forwarder.compute_expected_q(
            obs_t, act_t, "none"
        )

        loss = logsumexp.mean(dim=0).mean() - data_values.mean(dim=0).mean()
        scaled_loss = self._conservative_weight * loss

        # clip for stability
        clipped_alpha = self._modules.log_alpha().exp().clamp(0, 1e6)[0][0]

        return clipped_alpha * (scaled_loss - self._alpha_threshold)

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        if self._soft_q_backup:
            target_value = super().compute_target(batch)
        else:
            target_value = self._compute_deterministic_target(batch)
        return target_value

    def _compute_deterministic_target(
        self, batch: TorchMiniBatch
    ) -> torch.Tensor:
        with torch.no_grad():
            action = self._modules.policy(batch.next_observations).squashed_mu
            return self._targ_q_func_forwarder.compute_target(
                batch.next_observations,
                action,
                reduction="min",
            )

    def inner_update(
        self, batch: TorchMiniBatch, grad_step: int
    ) -> Dict[str, float]:
        metrics = {}

        # lagrangian parameter update for SAC temperature
        if self._modules.temp_optim:
            metrics.update(self.update_temp(batch))

        # lagrangian parameter update for conservative loss weight
        if self._modules.alpha_optim:
            metrics.update(self.update_alpha(batch))

        metrics.update(self.update_critic(batch))
        metrics.update(self.update_actor(batch))

        self.update_critic_target()

        return metrics


class DiscreteCQLImpl(DoubleDQNImpl):
    _alpha: float

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: DQNModules,
        q_func_forwarder: DiscreteEnsembleQFunctionForwarder,
        targ_q_func_forwarder: DiscreteEnsembleQFunctionForwarder,
        target_update_interval: int,
        gamma: float,
        alpha: float,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            target_update_interval=target_update_interval,
            gamma=gamma,
            device=device,
        )
        self._alpha = alpha

    def _compute_conservative_loss(
        self, obs_t: torch.Tensor, act_t: torch.Tensor
    ) -> torch.Tensor:
        # compute logsumexp
        values = self._q_func_forwarder.compute_expected_q(obs_t)
        logsumexp = torch.logsumexp(values, dim=1, keepdim=True)

        # estimate action-values under data distribution
        one_hot = F.one_hot(act_t.view(-1), num_classes=self.action_size)
        data_values = (values * one_hot).sum(dim=1, keepdim=True)

        return (logsumexp - data_values).mean()

    def compute_loss(
        self,
        batch: TorchMiniBatch,
        q_tpn: torch.Tensor,
    ) -> CQLLoss:
        td_loss = super().compute_loss(batch, q_tpn).loss
        conservative_loss = self._compute_conservative_loss(
            batch.observations, batch.actions.long()
        )
        return CQLLoss(
            td_loss=td_loss,
            conservative_loss=self._alpha * conservative_loss,
        )

import dataclasses

import torch
import torch.nn.functional as F
from torch import nn

from ....dataclass_utils import asdict_as_float
from ....models.torch import (
    ContinuousEnsembleQFunctionForwarder,
    NormalPolicy,
    Policy,
    build_gaussian_distribution,
)
from ....optimizers.optimizers import OptimizerWrapper
from ....torch_utility import (
    TorchMiniBatch,
    expand_and_repeat_recursively,
    flatten_left_recursively,
    hard_sync,
    soft_sync,
)
from ....types import TorchObservation
from ..functional import ActionSampler
from .ddpg_impl import (
    DDPGBaseActorLoss,
    DDPGBaseActorLossFn,
    DDPGBaseCriticLossFn,
    DDPGBaseModules,
    DDPGBaseUpdater,
)

__all__ = [
    "CRRActorLossFn",
    "CRRCriticLossFn",
    "CRRUpdater",
    "CRRActionSampler",
    "CRRModules",
]


@dataclasses.dataclass(frozen=True)
class CRRModules(DDPGBaseModules):
    policy: NormalPolicy
    targ_policy: NormalPolicy


class CRRCriticLossFn(DDPGBaseCriticLossFn):
    def __init__(
        self,
        q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        targ_q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        targ_policy: Policy,
        gamma: float,
    ):
        super().__init__(
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            gamma=gamma,
        )
        self._targ_policy = targ_policy

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            action = build_gaussian_distribution(
                self._targ_policy(batch.next_observations)
            ).sample()
            return self._targ_q_func_forwarder.compute_target(
                batch.next_observations,
                action.clamp(-1.0, 1.0),
                reduction="min",
            )


class CRRActorLossFn(DDPGBaseActorLossFn):
    def __init__(
        self,
        policy: Policy,
        q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        n_action_samples: int,
        advantage_type: str,
        weight_type: str,
        beta: float,
        max_weight: float,
        action_size: int,
    ):
        self._policy = policy
        self._q_func_forwarder = q_func_forwarder
        self._n_action_samples = n_action_samples
        self._advantage_type = advantage_type
        self._weight_type = weight_type
        self._beta = beta
        self._max_weight = max_weight
        self._action_size = action_size

    def _compute_weight(
        self, obs_t: TorchObservation, act_t: torch.Tensor
    ) -> torch.Tensor:
        advantages = self._compute_advantage(obs_t, act_t)
        if self._weight_type == "binary":
            return (advantages > 0.0).float()
        elif self._weight_type == "exp":
            return (advantages / self._beta).exp().clamp(0.0, self._max_weight)
        raise ValueError(f"invalid weight type: {self._weight_type}.")

    def _compute_advantage(
        self, obs_t: TorchObservation, act_t: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            # (batch_size, N, action)
            dist = build_gaussian_distribution(self._policy(obs_t))
            policy_actions = dist.sample_n(self._n_action_samples)
            flat_actions = policy_actions.reshape(-1, self._action_size)

            # repeat observation
            # (batch_size, obs_size) -> (batch_size, N, obs_size)
            repeated_obs_t = expand_and_repeat_recursively(
                obs_t, self._n_action_samples
            )
            # (batch_size, N, obs_size) -> (batch_size * N, obs_size)
            flat_obs_t = flatten_left_recursively(repeated_obs_t, dim=1)

            flat_values = self._q_func_forwarder.compute_expected_q(
                flat_obs_t, flat_actions
            )
            reshaped_values = flat_values.view(-1, self._n_action_samples, 1)

            if self._advantage_type == "mean":
                values = reshaped_values.mean(dim=1)
            elif self._advantage_type == "max":
                values = reshaped_values.max(dim=1).values
            else:
                raise ValueError(
                    f"invalid advantage type: {self._advantage_type}."
                )

            return (
                self._q_func_forwarder.compute_expected_q(obs_t, act_t) - values
            )

    def __call__(self, batch: TorchMiniBatch) -> DDPGBaseActorLoss:
        # compute log probability
        action = self._policy(batch.observations)
        dist = build_gaussian_distribution(action)
        log_probs = dist.log_prob(batch.actions)
        weight = self._compute_weight(batch.observations, batch.actions)
        return DDPGBaseActorLoss(-(log_probs * weight).mean())


class CRRActionSampler(ActionSampler):
    def __init__(
        self,
        policy: Policy,
        q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        n_action_samples: int,
        action_size: int,
    ):
        self._policy = policy
        self._q_func_forwarder = q_func_forwarder
        self._n_action_samples = n_action_samples
        self._action_size = action_size

    def __call__(self, x: TorchObservation) -> torch.Tensor:
        # compute CWP

        dist = build_gaussian_distribution(self._policy(x))
        actions = dist.onnx_safe_sample_n(self._n_action_samples)
        # (batch_size, N, action_size) -> (batch_size * N, action_size)
        flat_actions = actions.reshape(-1, self._action_size)

        # repeat observation
        # (batch_size, obs_size) -> (batch_size, N, obs_size)
        repeated_obs_t = expand_and_repeat_recursively(
            x, self._n_action_samples
        )
        # (batch_size, N, obs_size) -> (batch_size * N, obs_size)
        flat_obs_t = flatten_left_recursively(repeated_obs_t, dim=1)

        # (batch_size * N, 1)
        flat_values = self._q_func_forwarder.compute_expected_q(
            flat_obs_t, flat_actions
        )
        # (batch_size * N, 1) -> (batch_size, N)
        reshaped_values = flat_values.view(-1, self._n_action_samples)

        # re-sampling
        probs = F.softmax(reshaped_values, dim=1)
        indices = torch.multinomial(probs, 1, replacement=True)

        return actions[torch.arange(probs.shape[0]), indices.view(-1)]


class CRRUpdater(DDPGBaseUpdater):
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
        target_update_type: str,
        target_update_interval: int,
        tau: float,
        compiled: bool,
    ):
        super().__init__(
            critic_optim=critic_optim,
            actor_optim=actor_optim,
            critic_loss_fn=critic_loss_fn,
            actor_loss_fn=actor_loss_fn,
            compiled=compiled,
        )
        self._q_funcs = q_funcs
        self._targ_q_funcs = targ_q_funcs
        self._policy = policy
        self._targ_policy = targ_policy
        self._target_update_type = target_update_type
        self._target_update_interval = target_update_interval
        self._tau = tau

    def __call__(
        self, batch: TorchMiniBatch, grad_step: int
    ) -> dict[str, float]:
        metrics = {}

        # update critic
        critic_loss = self._compute_critic_grad(batch)
        self._critic_optim.step()
        metrics.update(asdict_as_float(critic_loss))

        # update actor
        actor_loss = self._compute_actor_grad(batch)
        self._actor_optim.step()
        metrics.update(asdict_as_float(actor_loss))

        # update target networks
        if self._target_update_type == "hard":
            if grad_step % self._target_update_interval == 0:
                hard_sync(self._targ_q_funcs, self._q_funcs)
                hard_sync(self._targ_policy, self._policy)
        elif self._target_update_type == "soft":
            self.update_target()
        else:
            raise ValueError(
                f"invalid target_update_type: {self._target_update_type}"
            )

        return metrics

    def update_target(self) -> None:
        soft_sync(self._targ_q_funcs, self._q_funcs, self._tau)
        soft_sync(self._targ_policy, self._policy, self._tau)

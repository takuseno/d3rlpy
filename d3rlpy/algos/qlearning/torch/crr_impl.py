import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer

from ....dataset import Shape
from ....models.torch import (
    ContinuousEnsembleQFunctionForwarder,
    NormalPolicy,
    build_gaussian_distribution,
)
from ....torch_utility import Checkpointer, TorchMiniBatch, hard_sync
from .ddpg_impl import DDPGBaseImpl

__all__ = ["CRRImpl"]


class CRRImpl(DDPGBaseImpl):
    _beta: float
    _n_action_samples: int
    _advantage_type: str
    _weight_type: str
    _max_weight: float
    _policy: NormalPolicy
    _targ_policy: NormalPolicy

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        policy: NormalPolicy,
        q_funcs: nn.ModuleList,
        q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        targ_q_funcs: nn.ModuleList,
        targ_q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        actor_optim: Optimizer,
        critic_optim: Optimizer,
        gamma: float,
        beta: float,
        n_action_samples: int,
        advantage_type: str,
        weight_type: str,
        max_weight: float,
        tau: float,
        checkpointer: Checkpointer,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            policy=policy,
            q_funcs=q_funcs,
            q_func_forwarder=q_func_forwarder,
            targ_q_funcs=targ_q_funcs,
            targ_q_func_forwarder=targ_q_func_forwarder,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
            gamma=gamma,
            tau=tau,
            checkpointer=checkpointer,
            device=device,
        )
        self._beta = beta
        self._n_action_samples = n_action_samples
        self._advantage_type = advantage_type
        self._weight_type = weight_type
        self._max_weight = max_weight

    def compute_actor_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        # compute log probability
        dist = build_gaussian_distribution(self._policy(batch.observations))
        log_probs = dist.log_prob(batch.actions)

        weight = self._compute_weight(batch.observations, batch.actions)

        return -(log_probs * weight).mean()

    def _compute_weight(
        self, obs_t: torch.Tensor, act_t: torch.Tensor
    ) -> torch.Tensor:
        advantages = self._compute_advantage(obs_t, act_t)
        if self._weight_type == "binary":
            return (advantages > 0.0).float()
        elif self._weight_type == "exp":
            return (advantages / self._beta).exp().clamp(0.0, self._max_weight)
        raise ValueError(f"invalid weight type: {self._weight_type}.")

    def _compute_advantage(
        self, obs_t: torch.Tensor, act_t: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            batch_size = obs_t.shape[0]

            # (batch_size, N, action)
            dist = build_gaussian_distribution(self._policy(obs_t))
            policy_actions = dist.sample_n(self._n_action_samples)
            flat_actions = policy_actions.reshape(-1, self._action_size)

            # repeat observation
            # (batch_size, obs_size) -> (batch_size, 1, obs_size)
            reshaped_obs_t = obs_t.view(batch_size, 1, *obs_t.shape[1:])
            # (batch_sie, 1, obs_size) -> (batch_size, N, obs_size)
            repeated_obs_t = reshaped_obs_t.expand(
                batch_size, self._n_action_samples, *obs_t.shape[1:]
            )
            # (batch_size, N, obs_size) -> (batch_size * N, obs_size)
            flat_obs_t = repeated_obs_t.reshape(-1, *obs_t.shape[1:])

            flat_values = self._q_func_forwarder.compute_expected_q(
                flat_obs_t, flat_actions
            )
            reshaped_values = flat_values.view(obs_t.shape[0], -1, 1)

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

    def inner_predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        # compute CWP

        dist = build_gaussian_distribution(self._policy(x))
        actions = dist.onnx_safe_sample_n(self._n_action_samples)
        # (batch_size, N, action_size) -> (batch_size * N, action_size)
        flat_actions = actions.reshape(-1, self._action_size)

        # repeat observation
        # (batch_size, obs_size) -> (batch_size, 1, obs_size)
        reshaped_obs_t = x.view(x.shape[0], 1, *x.shape[1:])
        # (batch_size, 1, obs_size) -> (batch_size, N, obs_size)
        repeated_obs_t = reshaped_obs_t.expand(
            x.shape[0], self._n_action_samples, *x.shape[1:]
        )
        # (batch_size, N, obs_size) -> (batch_size * N, obs_size)
        flat_obs_t = repeated_obs_t.reshape(-1, *x.shape[1:])

        # (batch_size * N, 1)
        flat_values = self._q_func_forwarder.compute_expected_q(
            flat_obs_t, flat_actions
        )
        # (batch_size * N, 1) -> (batch_size, N)
        reshaped_values = flat_values.view(x.shape[0], -1)

        # re-sampling
        probs = F.softmax(reshaped_values, dim=1)
        indices = torch.multinomial(probs, 1, replacement=True)

        return actions[torch.arange(x.shape[0]), indices.view(-1)]

    def inner_sample_action(self, x: torch.Tensor) -> torch.Tensor:
        dist = build_gaussian_distribution(self._policy(x))
        return dist.sample()

    def sync_critic_target(self) -> None:
        hard_sync(self._targ_q_funcs, self._q_funcs)

    def sync_actor_target(self) -> None:
        hard_sync(self._targ_policy, self._policy)

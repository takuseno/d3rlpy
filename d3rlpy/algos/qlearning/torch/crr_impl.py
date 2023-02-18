import torch
import torch.nn.functional as F
from torch.optim import Optimizer

from ....dataset import Shape
from ....models.torch import (
    EnsembleContinuousQFunction,
    NonSquashedNormalPolicy,
)
from ....torch_utility import TorchMiniBatch, hard_sync
from .ddpg_impl import DDPGBaseImpl

__all__ = ["CRRImpl"]


class CRRImpl(DDPGBaseImpl):

    _beta: float
    _n_action_samples: int
    _advantage_type: str
    _weight_type: str
    _max_weight: float
    _policy: NonSquashedNormalPolicy
    _targ_policy: NonSquashedNormalPolicy

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        policy: NonSquashedNormalPolicy,
        q_func: EnsembleContinuousQFunction,
        actor_optim: Optimizer,
        critic_optim: Optimizer,
        gamma: float,
        beta: float,
        n_action_samples: int,
        advantage_type: str,
        weight_type: str,
        max_weight: float,
        tau: float,
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
        self._beta = beta
        self._n_action_samples = n_action_samples
        self._advantage_type = advantage_type
        self._weight_type = weight_type
        self._max_weight = max_weight

    def compute_actor_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        # compute log probability
        dist = self._policy.dist(batch.observations)
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
            policy_actions = self._policy.sample_n(
                obs_t, self._n_action_samples
            )
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

            flat_values = self._q_func(flat_obs_t, flat_actions)
            reshaped_values = flat_values.view(obs_t.shape[0], -1, 1)

            if self._advantage_type == "mean":
                values = reshaped_values.mean(dim=1)
            elif self._advantage_type == "max":
                values = reshaped_values.max(dim=1).values
            else:
                raise ValueError(
                    f"invalid advantage type: {self._advantage_type}."
                )

            return self._q_func(obs_t, act_t) - values

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            action = self._targ_policy.sample(batch.next_observations)
            return self._targ_q_func.compute_target(
                batch.next_observations,
                action.clamp(-1.0, 1.0),
                reduction="min",
            )

    def inner_predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        # compute CWP

        actions = self._policy.onnx_safe_sample_n(x, self._n_action_samples)
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
        flat_values = self._q_func(flat_obs_t, flat_actions)
        # (batch_size * N, 1) -> (batch_size, N)
        reshaped_values = flat_values.view(x.shape[0], -1)

        # re-sampling
        probs = F.softmax(reshaped_values, dim=1)
        indices = torch.multinomial(probs, 1, replacement=True)

        return actions[torch.arange(x.shape[0]), indices.view(-1)]

    def sync_critic_target(self) -> None:
        hard_sync(self._targ_q_func, self._q_func)

    def sync_actor_target(self) -> None:
        hard_sync(self._targ_policy, self._policy)

import torch
import torch.nn.functional as F
from torch.optim import Adam, Optimizer

from ....dataset import Shape
from ....models.torch import (
    EnsembleContinuousQFunction,
    NonSquashedNormalPolicy,
    Parameter,
    Policy,
)
from ....torch_utility import TorchMiniBatch
from .sac_impl import SACImpl

__all__ = ["AWACImpl"]


class AWACImpl(SACImpl):
    _policy: NonSquashedNormalPolicy
    _lam: float
    _n_action_samples: int

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        q_func: EnsembleContinuousQFunction,
        policy: Policy,
        actor_optim: Optimizer,
        critic_optim: Optimizer,
        gamma: float,
        tau: float,
        lam: float,
        n_action_samples: int,
        device: str,
    ):
        assert isinstance(policy, NonSquashedNormalPolicy)
        dummy_log_temp = Parameter(torch.zeros(1))
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            q_func=q_func,
            policy=policy,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
            log_temp=dummy_log_temp,
            temp_optim=Adam(dummy_log_temp.parameters(), lr=0.0),
            gamma=gamma,
            tau=tau,
            device=device,
        )
        self._lam = lam
        self._n_action_samples = n_action_samples

    def compute_actor_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        # compute log probability
        dist = self._policy.dist(batch.observations)
        log_probs = dist.log_prob(batch.actions)

        # compute exponential weight
        weights = self._compute_weights(batch.observations, batch.actions)

        return -(log_probs * weights).sum()

    def _compute_weights(
        self, obs_t: torch.Tensor, act_t: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            batch_size = obs_t.shape[0]

            # compute action-value
            q_values = self._q_func(obs_t, act_t, "min")

            # sample actions
            # (batch_size * N, action_size)
            policy_actions = self._policy.sample_n(
                obs_t, self._n_action_samples
            )
            flat_actions = policy_actions.reshape(-1, self.action_size)

            # repeat observation
            # (batch_size, obs_size) -> (batch_size, 1, obs_size)
            reshaped_obs_t = obs_t.view(batch_size, 1, *obs_t.shape[1:])
            # (batch_sie, 1, obs_size) -> (batch_size, N, obs_size)
            repeated_obs_t = reshaped_obs_t.expand(
                batch_size, self._n_action_samples, *obs_t.shape[1:]
            )
            # (batch_size, N, obs_size) -> (batch_size * N, obs_size)
            flat_obs_t = repeated_obs_t.reshape(-1, *obs_t.shape[1:])

            # compute state-value
            flat_v_values = self._q_func(flat_obs_t, flat_actions, "min")
            reshaped_v_values = flat_v_values.view(obs_t.shape[0], -1, 1)
            v_values = reshaped_v_values.mean(dim=1)

            # compute normalized weight
            adv_values = (q_values - v_values).view(-1)
            weights = F.softmax(adv_values / self._lam, dim=0).view(-1, 1)

        return weights * adv_values.numel()

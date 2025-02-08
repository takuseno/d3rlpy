import torch
import torch.nn.functional as F

from ....models.torch import (
    ContinuousEnsembleQFunctionForwarder,
    Policy,
    build_gaussian_distribution,
)
from ....torch_utility import (
    TorchMiniBatch,
    expand_and_repeat_recursively,
    flatten_left_recursively,
    get_batch_size,
)
from ....types import TorchObservation
from .ddpg_impl import DDPGBaseActorLoss, DDPGBaseActorLossFn

__all__ = ["AWACActorLossFn"]


class AWACActorLossFn(DDPGBaseActorLossFn):
    def __init__(
        self,
        q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        policy: Policy,
        n_action_samples: int,
        lam: float,
        action_size: int,
    ):
        self._q_func_forwarder = q_func_forwarder
        self._policy = policy
        self._n_action_samples = n_action_samples
        self._lam = lam
        self._action_size = action_size

    def __call__(self, batch: TorchMiniBatch) -> DDPGBaseActorLoss:
        # compute log probability
        action = self._policy(batch.observations)
        dist = build_gaussian_distribution(action)
        log_probs = dist.log_prob(batch.actions)
        # compute exponential weight
        weights = self._compute_weights(batch.observations, batch.actions)
        loss = -(log_probs * weights).sum()
        return DDPGBaseActorLoss(actor_loss=loss)

    def _compute_weights(
        self, obs_t: TorchObservation, act_t: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            batch_size = get_batch_size(obs_t)

            # compute action-value
            q_values = self._q_func_forwarder.compute_expected_q(
                obs_t, act_t, "min"
            )

            # sample actions
            # (batch_size * N, action_size)
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

            # compute state-value
            flat_v_values = self._q_func_forwarder.compute_expected_q(
                flat_obs_t, flat_actions, "min"
            )
            reshaped_v_values = flat_v_values.view(batch_size, -1, 1)
            v_values = reshaped_v_values.mean(dim=1)

            # compute normalized weight
            adv_values = (q_values - v_values).view(-1)
            weights = F.softmax(adv_values / self._lam, dim=0).view(-1, 1)

        return weights * adv_values.numel()

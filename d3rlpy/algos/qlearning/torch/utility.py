
import torch
from typing_extensions import Protocol

from ....models.torch import (
    ContinuousEnsembleQFunctionForwarder,
    DiscreteEnsembleQFunctionForwarder,
    NormalPolicy,
    build_squashed_gaussian_distribution,
)
from ....torch_utility import (
    expand_and_repeat_recursively,
    flatten_left_recursively,
)
from ....types import TorchObservation

__all__ = [
    "DiscreteQFunctionMixin",
    "ContinuousQFunctionMixin",
    "sample_q_values_with_policy",
]


class _DiscreteQFunctionProtocol(Protocol):
    _q_func_forwarder: DiscreteEnsembleQFunctionForwarder


class _ContinuousQFunctionProtocol(Protocol):
    _q_func_forwarder: ContinuousEnsembleQFunctionForwarder


class DiscreteQFunctionMixin:
    def inner_predict_value(
        self: _DiscreteQFunctionProtocol,
        x: TorchObservation,
        action: torch.Tensor,
    ) -> torch.Tensor:
        values = self._q_func_forwarder.compute_expected_q(x, reduction="mean")
        flat_action = action.reshape(-1)
        return values[torch.arange(0, values.size(0)), flat_action].reshape(-1)


class ContinuousQFunctionMixin:
    def inner_predict_value(
        self: _ContinuousQFunctionProtocol,
        x: TorchObservation,
        action: torch.Tensor,
    ) -> torch.Tensor:
        return self._q_func_forwarder.compute_expected_q(
            x, action, reduction="mean"
        ).reshape(-1)


def sample_q_values_with_policy(
    policy: NormalPolicy,
    q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
    policy_observations: TorchObservation,
    value_observations: TorchObservation,
    n_action_samples: int,
    detach_policy_output: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    dist = build_squashed_gaussian_distribution(policy(policy_observations))
    # (batch, n, action), (batch, n)
    policy_actions, n_log_probs = dist.sample_n_with_log_prob(n_action_samples)

    if detach_policy_output:
        policy_actions = policy_actions.detach()
        n_log_probs = n_log_probs.detach()

    # (batch, observation) -> (batch, n, observation)
    repeated_obs = expand_and_repeat_recursively(
        x=value_observations,
        n=n_action_samples,
    )
    # (batch, n, observation) -> (batch * n, observation)
    flat_obs = flatten_left_recursively(repeated_obs, dim=1)
    # (batch, n, action) -> (batch * n, action)
    flat_policy_acts = policy_actions.reshape(-1, policy_actions.shape[-1])

    # estimate action-values for policy actions
    # (M, batch * n, 1)
    policy_values = q_func_forwarder.compute_expected_q(
        flat_obs, flat_policy_acts, "none"
    )
    batch_size = (
        policy_observations.shape[0]
        if isinstance(policy_observations, torch.Tensor)
        else policy_observations[0].shape[0]
    )
    policy_values = policy_values.view(-1, batch_size, n_action_samples)
    log_probs = n_log_probs.view(1, -1, n_action_samples)

    # (M, batch, n), (1, batch, n)
    return policy_values, log_probs

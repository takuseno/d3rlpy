import torch
import torch.nn.functional as F

from ....models.torch import (
    ActionOutput,
    ContinuousEnsembleQFunctionForwarder,
    build_gaussian_distribution,
)
from ....torch_utility import (
    TorchMiniBatch,
    expand_and_repeat_recursively,
    flatten_left_recursively,
    get_batch_size,
)
from ....types import Shape, TorchObservation
from .sac_impl import SACActorLoss, SACImpl, SACModules

__all__ = ["AWACImpl"]


class AWACImpl(SACImpl):
    _lam: float
    _n_action_samples: int

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: SACModules,
        q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        targ_q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        gamma: float,
        tau: float,
        lam: float,
        n_action_samples: int,
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

    def compute_actor_loss(
        self, batch: TorchMiniBatch, action: ActionOutput, grad_step: int
    ) -> SACActorLoss:
        # compute log probability
        dist = build_gaussian_distribution(action)
        log_probs = dist.log_prob(batch.actions)
        # compute exponential weight
        weights = self._compute_weights(batch.observations, batch.actions)
        loss = -(log_probs * weights).sum()
        return SACActorLoss(
            actor_loss=loss,
            temp_loss=torch.tensor(
                0.0, dtype=torch.float32, device=loss.device
            ),
            temp=torch.tensor(0.0, dtype=torch.float32, device=loss.device),
        )

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
            dist = build_gaussian_distribution(self._modules.policy(obs_t))
            policy_actions = dist.sample_n(self._n_action_samples)
            flat_actions = policy_actions.reshape(-1, self.action_size)

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

    def inner_sample_action(self, x: TorchObservation) -> torch.Tensor:
        dist = build_gaussian_distribution(self._modules.policy(x))
        return dist.sample()

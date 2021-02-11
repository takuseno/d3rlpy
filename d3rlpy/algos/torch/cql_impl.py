import math
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Optimizer

from ...models.torch import Parameter
from ...models.builders import create_parameter
from ...models.optimizers import OptimizerFactory
from ...models.encoders import EncoderFactory
from ...models.q_functions import QFunctionFactory
from ...gpu import Device
from ...augmentation import AugmentationPipeline
from ...preprocessing import Scaler, ActionScaler
from ...torch_utility import torch_api, train_api, augmentation_api
from .sac_impl import SACImpl
from .dqn_impl import DoubleDQNImpl


class CQLImpl(SACImpl):

    _alpha_learning_rate: float
    _alpha_optim_factory: OptimizerFactory
    _initial_alpha: float
    _alpha_threshold: float
    _n_action_samples: int
    _soft_q_backup: bool
    _log_alpha: Optional[Parameter]
    _alpha_optim: Optional[Optimizer]

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        actor_learning_rate: float,
        critic_learning_rate: float,
        temp_learning_rate: float,
        alpha_learning_rate: float,
        actor_optim_factory: OptimizerFactory,
        critic_optim_factory: OptimizerFactory,
        temp_optim_factory: OptimizerFactory,
        alpha_optim_factory: OptimizerFactory,
        actor_encoder_factory: EncoderFactory,
        critic_encoder_factory: EncoderFactory,
        q_func_factory: QFunctionFactory,
        gamma: float,
        tau: float,
        n_critics: int,
        bootstrap: bool,
        share_encoder: bool,
        target_reduction_type: str,
        initial_temperature: float,
        initial_alpha: float,
        alpha_threshold: float,
        n_action_samples: int,
        soft_q_backup: bool,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        action_scaler: Optional[ActionScaler],
        augmentation: AugmentationPipeline,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            temp_learning_rate=temp_learning_rate,
            actor_optim_factory=actor_optim_factory,
            critic_optim_factory=critic_optim_factory,
            temp_optim_factory=temp_optim_factory,
            actor_encoder_factory=actor_encoder_factory,
            critic_encoder_factory=critic_encoder_factory,
            q_func_factory=q_func_factory,
            gamma=gamma,
            tau=tau,
            n_critics=n_critics,
            bootstrap=bootstrap,
            share_encoder=share_encoder,
            target_reduction_type=target_reduction_type,
            initial_temperature=initial_temperature,
            use_gpu=use_gpu,
            scaler=scaler,
            action_scaler=action_scaler,
            augmentation=augmentation,
        )
        self._alpha_learning_rate = alpha_learning_rate
        self._alpha_optim_factory = alpha_optim_factory
        self._initial_alpha = initial_alpha
        self._alpha_threshold = alpha_threshold
        self._n_action_samples = n_action_samples
        self._soft_q_backup = soft_q_backup

        # initialized in build
        self._log_alpha = None
        self._alpha_optim = None

    def build(self) -> None:
        self._build_alpha()
        super().build()
        self._build_alpha_optim()

    def _build_alpha(self) -> None:
        initial_val = math.log(self._initial_alpha)
        self._log_alpha = create_parameter((1, 1), initial_val)

    def _build_alpha_optim(self) -> None:
        assert self._log_alpha is not None
        self._alpha_optim = self._alpha_optim_factory.create(
            self._log_alpha.parameters(), lr=self._alpha_learning_rate
        )

    def _compute_critic_loss(
        self,
        obs_t: torch.Tensor,
        act_t: torch.Tensor,
        rew_tpn: torch.Tensor,
        q_tpn: torch.Tensor,
        ter_tpn: torch.Tensor,
        n_steps: torch.Tensor,
    ) -> torch.Tensor:
        loss = super()._compute_critic_loss(
            obs_t, act_t, rew_tpn, q_tpn, ter_tpn, n_steps
        )
        conservative_loss = self._compute_conservative_loss(obs_t, act_t)
        return loss + conservative_loss

    @train_api
    @torch_api(scaler_targets=["obs_t"], action_scaler_targets=["act_t"])
    def update_alpha(
        self, obs_t: torch.Tensor, act_t: torch.Tensor
    ) -> np.ndarray:
        assert self._alpha_optim is not None
        assert self._q_func is not None
        assert self._log_alpha is not None

        # Q function should be inference mode for stability
        self._q_func.eval()

        self._alpha_optim.zero_grad()

        loss = -self._compute_conservative_loss(obs_t, act_t)

        loss.backward()
        self._alpha_optim.step()

        cur_alpha = self._log_alpha().exp().cpu().detach().numpy()[0][0]

        return loss.cpu().detach().numpy(), cur_alpha

    def _compute_conservative_loss(
        self, obs_t: torch.Tensor, act_t: torch.Tensor
    ) -> torch.Tensor:
        assert self._policy is not None
        assert self._q_func is not None
        assert self._log_alpha is not None
        with torch.no_grad():
            policy_actions, n_log_probs = self._policy.sample_n_with_log_prob(
                obs_t, self._n_action_samples
            )

        # policy action for t
        repeated_obs = obs_t.expand(self._n_action_samples, *obs_t.shape)
        # (n, batch, observation) -> (batch, n, observation)
        transposed_obs = repeated_obs.transpose(0, 1)
        # (batch, n, observation) -> (batch * n, observation)
        flat_obs = transposed_obs.reshape(-1, *obs_t.shape[1:])
        # (batch, n, action) -> (batch * n, action)
        flat_policy_acts = policy_actions.reshape(-1, self.action_size)

        # estimate action-values for policy actions
        policy_values = self._q_func(flat_obs, flat_policy_acts, "none")
        policy_values = policy_values.view(
            self._n_critics, obs_t.shape[0], self._n_action_samples
        )
        log_probs = n_log_probs.view(1, -1, self._n_action_samples)

        # estimate action-values for actions from uniform distribution
        # uniform distribution between [-1.0, 1.0]
        random_actions = torch.zeros_like(flat_policy_acts).uniform_(-1.0, 1.0)
        random_values = self._q_func(flat_obs, random_actions, "none")
        random_values = random_values.view(
            self._n_critics, obs_t.shape[0], self._n_action_samples
        )
        random_log_probs = math.log(0.5 ** self._action_size)

        # compute logsumexp
        # (n critics, batch, 2 * n samples) -> (n critics, batch, 1)
        target_values = torch.cat(
            [policy_values - log_probs, random_values - random_log_probs], dim=2
        )
        logsumexp = torch.logsumexp(target_values, dim=2, keepdim=True)

        # estimate action-values for data actions
        data_values = self._q_func(obs_t, act_t, "none")

        element_wise_loss = logsumexp - data_values - self._alpha_threshold

        # clip for stability
        clipped_alpha = self._log_alpha().exp().clamp(0, 1e6)

        return (clipped_alpha[0][0] * element_wise_loss).sum(dim=0).mean()

    @augmentation_api(targets=["x"])
    def compute_target(self, x: torch.Tensor) -> torch.Tensor:
        assert self._policy is not None
        assert self._log_temp is not None
        assert self._targ_q_func is not None
        with torch.no_grad():
            action, log_prob = self._policy.sample_with_log_prob(x)
            target_value = self._targ_q_func.compute_target(
                x, action, reduction=self._target_reduction_type
            )
            if self._soft_q_backup:
                entropy = self._log_temp().exp() * log_prob
                if self._target_reduction_type == "none":
                    target_value -= entropy.view(1, -1, 1)
                else:
                    target_value -= entropy
            return target_value


class DiscreteCQLImpl(DoubleDQNImpl):
    def _compute_loss(
        self,
        obs_t: torch.Tensor,
        act_t: torch.Tensor,
        rew_tpn: torch.Tensor,
        q_tpn: torch.Tensor,
        ter_tpn: torch.Tensor,
        n_steps: torch.Tensor,
    ) -> torch.Tensor:
        loss = super()._compute_loss(
            obs_t, act_t, rew_tpn, q_tpn, ter_tpn, n_steps
        )
        conservative_loss = self._compute_conservative_loss(obs_t, act_t)
        return loss + conservative_loss

    def _compute_conservative_loss(
        self, obs_t: torch.Tensor, act_t: torch.Tensor
    ) -> torch.Tensor:
        assert self._q_func is not None
        # compute logsumexp
        policy_values = self._q_func(obs_t)
        logsumexp = torch.logsumexp(policy_values, dim=1, keepdim=True)

        # estimate action-values under data distribution
        one_hot = F.one_hot(act_t.view(-1), num_classes=self.action_size)
        data_values = (self._q_func(obs_t) * one_hot).sum(dim=1, keepdim=True)

        return (logsumexp - data_values).mean()

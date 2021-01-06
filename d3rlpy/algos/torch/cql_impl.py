import math
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer

from ...optimizers import OptimizerFactory
from ...encoders import EncoderFactory
from ...q_functions import QFunctionFactory
from ...gpu import Device
from ...augmentation import AugmentationPipeline
from ...preprocessing import Scaler
from ...torch_utility import torch_api, train_api
from .sac_impl import SACImpl
from .dqn_impl import DoubleDQNImpl


class CQLImpl(SACImpl):

    _alpha_learning_rate: float
    _alpha_optim_factory: OptimizerFactory
    _initial_alpha: float
    _alpha_threshold: float
    _n_action_samples: int
    _log_alpha: Optional[nn.Parameter]
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
        initial_temperature: float,
        initial_alpha: float,
        alpha_threshold: float,
        n_action_samples: int,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
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
            initial_temperature=initial_temperature,
            use_gpu=use_gpu,
            scaler=scaler,
            augmentation=augmentation,
        )
        self._alpha_learning_rate = alpha_learning_rate
        self._alpha_optim_factory = alpha_optim_factory
        self._initial_alpha = initial_alpha
        self._alpha_threshold = alpha_threshold
        self._n_action_samples = n_action_samples

        # initialized in build
        self._log_alpha = None
        self._alpha_optim = None

    def build(self) -> None:
        super().build()
        self._build_alpha()
        self._build_alpha_optim()

    def _build_alpha(self) -> None:
        initial_val = math.log(self._initial_alpha)
        data = torch.full((1, 1), initial_val, device=self._device)
        self._log_alpha = nn.Parameter(data)

    def _build_alpha_optim(self) -> None:
        assert self._log_alpha is not None
        self._alpha_optim = self._alpha_optim_factory.create(
            [self._log_alpha], lr=self._alpha_learning_rate
        )

    def _compute_critic_loss(
        self,
        obs_t: torch.Tensor,
        act_t: torch.Tensor,
        rew_tpn: torch.Tensor,
        q_tpn: torch.Tensor,
        n_steps: torch.Tensor,
    ) -> torch.Tensor:
        loss = super()._compute_critic_loss(
            obs_t, act_t, rew_tpn, q_tpn, n_steps
        )
        conservative_loss = self._compute_conservative_loss(obs_t, act_t)
        return loss + conservative_loss

    @train_api
    @torch_api(scaler_targets=["obs_t"])
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

        cur_alpha = self._log_alpha.exp().cpu().detach().numpy()[0][0]

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

        repeated_obs_t = obs_t.expand(self._n_action_samples, *obs_t.shape)
        # (n, batch, observation) -> (batch, n, observation)
        transposed_obs_t = repeated_obs_t.transpose(0, 1)
        # (batch, n, observation) -> (batch * n, observation)
        flat_obs_t = transposed_obs_t.reshape(-1, *obs_t.shape[1:])
        # (batch, n, action) -> (batch * n, action)
        flat_policy_acts = policy_actions.reshape(-1, self.action_size)

        # estimate action-values for policy actions
        policy_values = self._q_func(flat_obs_t, flat_policy_acts, "none")
        policy_values = policy_values.view(
            self._n_critics, obs_t.shape[0], self._n_action_samples, 1
        )
        log_probs = n_log_probs.view(1, -1, self._n_action_samples, 1)

        # estimate action-values for actions from uniform distribution
        # uniform distribution between [-1.0, 1.0]
        random_actions = torch.zeros_like(flat_policy_acts).uniform_(-1.0, 1.0)
        random_values = self._q_func(flat_obs_t, random_actions, "none")
        random_values = random_values.view(
            self._n_critics, obs_t.shape[0], self._n_action_samples, 1
        )

        # get maximum value to avoid overflow
        base = torch.max(policy_values.max(), random_values.max()).detach()

        # compute logsumexp
        policy_meanexp = (policy_values - base - log_probs).exp().mean(dim=2)
        random_meanexp = (random_values - base).exp().mean(dim=2) / 0.5
        # small constant value seems to be necessary to avoid nan
        logsumexp = (0.5 * random_meanexp + 0.5 * policy_meanexp + 1e-10).log()
        logsumexp += base

        # estimate action-values for data actions
        data_values = self._q_func(obs_t, act_t, "none")

        element_wise_loss = logsumexp - data_values - self._alpha_threshold

        # this clipping seems to stabilize training
        clipped_alpha = self._log_alpha.clamp(-10.0, 2.0).exp()

        return (clipped_alpha * element_wise_loss).sum(dim=0).mean()


class DiscreteCQLImpl(DoubleDQNImpl):
    def _compute_loss(
        self,
        obs_t: torch.Tensor,
        act_t: torch.Tensor,
        rew_tpn: torch.Tensor,
        q_tpn: torch.Tensor,
        n_steps: torch.Tensor,
    ) -> torch.Tensor:
        loss = super()._compute_loss(obs_t, act_t, rew_tpn, q_tpn, n_steps)
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

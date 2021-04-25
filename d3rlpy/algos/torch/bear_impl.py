import math
from typing import Optional, Sequence

import numpy as np
import torch
from torch.optim import Optimizer

from ...gpu import Device
from ...models.builders import create_parameter, create_probablistic_regressor
from ...models.encoders import EncoderFactory
from ...models.optimizers import OptimizerFactory
from ...models.q_functions import QFunctionFactory
from ...models.torch import (
    Parameter,
    ProbablisticRegressor,
    compute_max_with_n_actions_and_indices,
)
from ...preprocessing import ActionScaler, Scaler
from ...torch_utility import torch_api, train_api
from .sac_impl import SACImpl


def _gaussian_kernel(
    x: torch.Tensor, y: torch.Tensor, sigma: float
) -> torch.Tensor:
    # x: (batch, n, 1, action), y: (batch, 1, n, action) -> (batch, n, n)
    return (-((x - y) ** 2).sum(dim=3) / (2 * sigma)).exp()


def _laplacian_kernel(
    x: torch.Tensor, y: torch.Tensor, sigma: float
) -> torch.Tensor:
    # x: (batch, n, 1, action), y: (batch, 1, n, action) -> (batch, n, n)
    return (-(x - y).abs().sum(dim=3) / (2 * sigma)).exp()


class BEARImpl(SACImpl):

    _imitator_learning_rate: float
    _alpha_learning_rate: float
    _imitator_optim_factory: OptimizerFactory
    _alpha_optim_factory: OptimizerFactory
    _imitator_encoder_factory: EncoderFactory
    _initial_alpha: float
    _alpha_threshold: float
    _lam: float
    _n_action_samples: int
    _mmd_kernel: str
    _mmd_sigma: float
    _imitator: Optional[ProbablisticRegressor]
    _imitator_optim: Optional[Optimizer]
    _log_alpha: Optional[Parameter]
    _alpha_optim: Optional[Optimizer]

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        actor_learning_rate: float,
        critic_learning_rate: float,
        imitator_learning_rate: float,
        temp_learning_rate: float,
        alpha_learning_rate: float,
        actor_optim_factory: OptimizerFactory,
        critic_optim_factory: OptimizerFactory,
        imitator_optim_factory: OptimizerFactory,
        temp_optim_factory: OptimizerFactory,
        alpha_optim_factory: OptimizerFactory,
        actor_encoder_factory: EncoderFactory,
        critic_encoder_factory: EncoderFactory,
        imitator_encoder_factory: EncoderFactory,
        q_func_factory: QFunctionFactory,
        gamma: float,
        tau: float,
        n_critics: int,
        initial_temperature: float,
        initial_alpha: float,
        alpha_threshold: float,
        lam: float,
        n_action_samples: int,
        mmd_kernel: str,
        mmd_sigma: float,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        action_scaler: Optional[ActionScaler],
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
            target_reduction_type="mix",
            initial_temperature=initial_temperature,
            use_gpu=use_gpu,
            scaler=scaler,
            action_scaler=action_scaler,
        )
        self._imitator_learning_rate = imitator_learning_rate
        self._alpha_learning_rate = alpha_learning_rate
        self._imitator_optim_factory = imitator_optim_factory
        self._alpha_optim_factory = alpha_optim_factory
        self._imitator_encoder_factory = imitator_encoder_factory
        self._initial_alpha = initial_alpha
        self._alpha_threshold = alpha_threshold
        self._lam = lam
        self._n_action_samples = n_action_samples
        self._mmd_kernel = mmd_kernel
        self._mmd_sigma = mmd_sigma

        # initialized in build
        self._imitator = None
        self._imitator_optim = None
        self._log_alpha = None
        self._alpha_optim = None

    def build(self) -> None:
        self._build_imitator()
        self._build_alpha()
        super().build()
        self._build_imitator_optim()
        self._build_alpha_optim()

    def _build_imitator(self) -> None:
        self._imitator = create_probablistic_regressor(
            self._observation_shape,
            self._action_size,
            self._imitator_encoder_factory,
        )

    def _build_imitator_optim(self) -> None:
        assert self._imitator is not None
        self._imitator_optim = self._imitator_optim_factory.create(
            self._imitator.parameters(), lr=self._imitator_learning_rate
        )

    def _build_alpha(self) -> None:
        initial_val = math.log(self._initial_alpha)
        self._log_alpha = create_parameter((1, 1), initial_val)

    def _build_alpha_optim(self) -> None:
        assert self._log_alpha is not None
        self._alpha_optim = self._alpha_optim_factory.create(
            self._log_alpha.parameters(), lr=self._alpha_learning_rate
        )

    def _compute_actor_loss(self, obs_t: torch.Tensor) -> torch.Tensor:
        loss = super()._compute_actor_loss(obs_t)
        mmd_loss = self._compute_mmd_loss(obs_t)
        return loss + mmd_loss

    @train_api
    @torch_api(scaler_targets=["obs_t"])
    def warmup_actor(self, obs_t: torch.Tensor) -> np.ndarray:
        assert self._actor_optim is not None

        self._actor_optim.zero_grad()

        loss = self.compute_mmd_loss(obs_t)

        loss.backward()
        self._actor_optim.step()

        return loss.cpu().detach().numpy()

    def compute_mmd_loss(self, obs_t: torch.Tensor) -> torch.Tensor:
        return self._compute_mmd_loss(obs_t)

    def _compute_mmd_loss(self, obs_t: torch.Tensor) -> torch.Tensor:
        assert self._log_alpha
        mmd = self._compute_mmd(obs_t)
        alpha = self._log_alpha().exp()
        return (alpha * (mmd - self._alpha_threshold)).sum(dim=1).mean()

    @train_api
    @torch_api(scaler_targets=["obs_t"], action_scaler_targets=["act_t"])
    def update_imitator(
        self, obs_t: torch.Tensor, act_t: torch.Tensor
    ) -> np.ndarray:
        assert self._imitator_optim is not None

        self._imitator_optim.zero_grad()

        loss = self.compute_imitator_loss(obs_t, act_t)

        loss.backward()

        self._imitator_optim.step()

        return loss.cpu().detach().numpy()

    def compute_imitator_loss(
        self, obs_t: torch.Tensor, act_t: torch.Tensor
    ) -> torch.Tensor:
        assert self._imitator is not None
        return self._imitator.compute_error(obs_t, act_t)

    @train_api
    @torch_api(scaler_targets=["obs_t"])
    def update_alpha(self, obs_t: torch.Tensor) -> np.ndarray:
        assert self._alpha_optim is not None
        assert self._log_alpha is not None

        loss = -self._compute_mmd_loss(obs_t)

        self._alpha_optim.zero_grad()
        loss.backward()
        self._alpha_optim.step()

        # clip for stability
        self._log_alpha.data.clamp_(-5.0, 10.0)

        cur_alpha = self._log_alpha().exp().cpu().detach().numpy()[0][0]

        return loss.cpu().detach().numpy(), cur_alpha

    def _compute_mmd(self, x: torch.Tensor) -> torch.Tensor:
        assert self._imitator is not None
        assert self._policy is not None
        with torch.no_grad():
            behavior_actions = self._imitator.sample_n(
                x, self._n_action_samples
            )
        policy_actions = self._policy.sample_n(x, self._n_action_samples)

        if self._mmd_kernel == "gaussian":
            kernel = _gaussian_kernel
        elif self._mmd_kernel == "laplacian":
            kernel = _laplacian_kernel
        else:
            raise ValueError(f"Invalid kernel type: {self._mmd_kernel}")

        # (batch, n, action) -> (batch, n, 1, action)
        behavior_actions = behavior_actions.reshape(
            x.shape[0], -1, 1, self.action_size
        )
        policy_actions = policy_actions.reshape(
            x.shape[0], -1, 1, self.action_size
        )
        # (batch, n, action) -> (batch, 1, n, action)
        behavior_actions_T = behavior_actions.reshape(
            x.shape[0], 1, -1, self.action_size
        )
        policy_actions_T = policy_actions.reshape(
            x.shape[0], 1, -1, self.action_size
        )

        # 1 / N^2 \sum k(a_\pi, a_\pi)
        inter_policy = kernel(policy_actions, policy_actions_T, self._mmd_sigma)
        mmd = inter_policy.sum(dim=1).sum(dim=1) / self._n_action_samples ** 2

        # 1 / N^2 \sum k(a_\beta, a_\beta)
        inter_data = kernel(
            behavior_actions, behavior_actions_T, self._mmd_sigma
        )
        mmd += inter_data.sum(dim=1).sum(dim=1) / self._n_action_samples ** 2

        # 2 / N^2 \sum k(a_\pi, a_\beta)
        distance = kernel(policy_actions, behavior_actions_T, self._mmd_sigma)
        mmd -= 2 * distance.sum(dim=1).sum(dim=1) / self._n_action_samples ** 2

        return (mmd + 1e-6).sqrt().view(-1, 1)

    def compute_target(self, x: torch.Tensor) -> torch.Tensor:
        assert self._policy is not None
        assert self._targ_q_func is not None
        assert self._log_temp is not None
        with torch.no_grad():
            # BCQ-like target computation
            actions, log_probs = self._policy.sample_n_with_log_prob(x, 100)
            values, indices = compute_max_with_n_actions_and_indices(
                x, actions, self._targ_q_func, self._lam
            )

            # (batch, n, 1) -> (batch, 1)
            max_log_prob = log_probs[torch.arange(x.shape[0]), indices]

            return values - self._log_temp().exp() * max_log_prob

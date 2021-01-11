# pylint: disable=arguments-differ

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ...models.torch import squash_action
from ...models.builders import create_normal_policy
from ...models.optimizers import OptimizerFactory, AdamFactory
from ...models.encoders import EncoderFactory
from ...models.q_functions import QFunctionFactory
from ...gpu import Device
from ...preprocessing import Scaler, ActionScaler
from ...augmentation import AugmentationPipeline
from ...torch_utility import torch_api, train_api, augmentation_api
from .sac_impl import SACImpl


class AWACImpl(SACImpl):

    _lam: float
    _n_action_samples: int
    _max_weight: float

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        actor_learning_rate: float,
        critic_learning_rate: float,
        actor_optim_factory: OptimizerFactory,
        critic_optim_factory: OptimizerFactory,
        actor_encoder_factory: EncoderFactory,
        critic_encoder_factory: EncoderFactory,
        q_func_factory: QFunctionFactory,
        gamma: float,
        tau: float,
        lam: float,
        n_action_samples: int,
        max_weight: float,
        n_critics: int,
        bootstrap: bool,
        share_encoder: bool,
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
            temp_learning_rate=0.0,
            actor_optim_factory=actor_optim_factory,
            critic_optim_factory=critic_optim_factory,
            temp_optim_factory=AdamFactory(),
            actor_encoder_factory=actor_encoder_factory,
            critic_encoder_factory=critic_encoder_factory,
            q_func_factory=q_func_factory,
            gamma=gamma,
            tau=tau,
            n_critics=n_critics,
            bootstrap=bootstrap,
            share_encoder=share_encoder,
            initial_temperature=1e-20,
            use_gpu=use_gpu,
            scaler=scaler,
            action_scaler=action_scaler,
            augmentation=augmentation,
        )
        self._lam = lam
        self._n_action_samples = n_action_samples
        self._max_weight = max_weight

    def _build_actor(self) -> None:
        self._policy = create_normal_policy(
            self._observation_shape,
            self._action_size,
            self._actor_encoder_factory,
            min_logstd=-6.0,
            max_logstd=0.0,
            use_std_parameter=True,
        )

    @train_api
    @torch_api(scaler_targets=["obs_t"], action_scaler_targets=["act_t"])
    def update_actor(
        self, obs_t: torch.Tensor, act_t: torch.Tensor
    ) -> np.ndarray:
        assert self._q_func is not None
        assert self._policy is not None
        assert self._actor_optim is not None

        # Q function should be inference mode for stability
        self._q_func.eval()

        self._actor_optim.zero_grad()

        loss = self.compute_actor_loss(obs_t, act_t)

        loss.backward()
        self._actor_optim.step()

        # get current standard deviation for policy function for debug
        mean_std = self._policy.get_logstd_parameter().exp().mean()

        return loss.cpu().detach().numpy(), mean_std.cpu().detach().numpy()

    @augmentation_api(targets=["obs_t"])
    def compute_actor_loss(
        self, obs_t: torch.Tensor, act_t: torch.Tensor
    ) -> torch.Tensor:
        return self._compute_actor_loss(obs_t, act_t)

    def _compute_actor_loss(  # type: ignore
        self, obs_t: torch.Tensor, act_t: torch.Tensor
    ) -> torch.Tensor:
        assert self._policy is not None

        dist = self._policy.dist(obs_t)

        # unnormalize action via inverse tanh function
        unnormalized_act_t = torch.atanh(act_t.clamp(-0.999999, 0.999999))

        # compute log probability
        _, log_probs = squash_action(dist, unnormalized_act_t)

        # compute exponential weight
        weights = self._compute_weights(obs_t, act_t)

        return -(log_probs * weights).sum()

    def _compute_weights(
        self, obs_t: torch.Tensor, act_t: torch.Tensor
    ) -> torch.Tensor:
        assert self._q_func is not None
        assert self._policy is not None
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

            # clip like AWR
            clipped_weights = weights.clamp(0.0, self._max_weight)

        return clipped_weights

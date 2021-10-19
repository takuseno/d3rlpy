# pylint: disable=too-many-ancestors

from typing import Optional, Sequence

import torch
import numpy as np
import copy

from .td3_impl import TD3Impl

from ...models.builders import (
    create_squashed_normal_policy,
    create_value_function,
)

from ...gpu import Device
from ...models.builders import (
    create_continuous_q_function,
)
from ...models.encoders import EncoderFactory
from ...models.optimizers import OptimizerFactory
from ...models.q_functions import QFunctionFactory
from ...models.torch import (
    SquashedNormalPolicy,
    squash_action,
)

from ...preprocessing import ActionScaler, RewardScaler, Scaler
from ...torch_utility import TorchMiniBatch, torch_api, train_api


class IQLImpl(TD3Impl):

    _alpha: float

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
        n_critics: int,
        target_reduction_type: str,
        expectile: float,
        beta: float,
        max_weight: float,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        action_scaler: Optional[ActionScaler],
        reward_scaler: Optional[RewardScaler],
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            actor_optim_factory=actor_optim_factory,
            critic_optim_factory=critic_optim_factory,
            actor_encoder_factory=actor_encoder_factory,
            critic_encoder_factory=critic_encoder_factory,
            q_func_factory=q_func_factory,
            gamma=gamma,
            tau=tau,
            n_critics=n_critics,
            target_reduction_type=target_reduction_type,
            target_smoothing_clip=0.5,
            target_smoothing_sigma=0.2,
            use_gpu=use_gpu,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
        )
        self._expectile = expectile
        self._beta = beta
        self._max_weight = max_weight

    def compute_v_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._targ_q_func is not None
        with torch.no_grad():
            return self._targ_q_func.compute_target(batch.observations,
                                                    batch.actions,
                                                    reduction='min')

    @train_api
    @torch_api()
    def update_v_func(self, batch: TorchMiniBatch) -> np.ndarray:
        assert self._v_func_optim is not None
        self._v_func_optim.zero_grad()
        q_t = self.compute_v_target(batch)
        loss, v_t = self.compute_v_func_loss(batch, q_t)
        loss.backward()
        self._v_func_optim.step()
        return loss.cpu().detach().numpy(), q_t.detach(), v_t.detach()

    def expectile_loss(self, diff, expectile=0.8):
        weight = torch.where(diff > 0, expectile, (1 - expectile))
        return weight * (diff ** 2)

    def compute_v_func_loss(self, batch: TorchMiniBatch, q_tpn: torch.Tensor) -> torch.Tensor:
        assert self._q_func is not None
        v_t = self._v_func.forward(batch.observations)
        loss = self.expectile_loss(v_t - q_tpn, expectile=self._expectile).mean()
        return loss, v_t

    def compute_q_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._v_func is not None
        with torch.no_grad():
            return self._v_func(batch.next_observations)

    @train_api
    @torch_api()
    def update_critic(self, batch: TorchMiniBatch) -> np.ndarray:
        assert self._critic_optim is not None
        self._critic_optim.zero_grad()
        q_tpn = self.compute_q_target(batch)
        loss = self.compute_critic_loss(batch, q_tpn)
        loss.backward()
        self._critic_optim.step()

        return loss.cpu().detach().numpy()

    def compute_critic_loss(self, batch: TorchMiniBatch, v_tpn: torch.Tensor) -> torch.Tensor:
        assert self._q_func is not None
        error = self._q_func.compute_error(
            obs_t=batch.observations,
            act_t=batch.actions,
            rew_tp1=batch.next_rewards,
            q_tp1=v_tpn,
            ter_tp1=batch.terminals,
            gamma=self._gamma ** batch.n_steps,
            use_independent_target=self._target_reduction_type == "none",
            masks=batch.masks,
        )
        return error

    @train_api
    @torch_api()
    def update_actor(self, batch: TorchMiniBatch, q_t, v_t) -> np.ndarray:
        assert self._q_func is not None
        assert self._actor_optim is not None
        self._q_func.eval()
        self._actor_optim.zero_grad()
        loss = self.compute_actor_loss(batch, q_t, v_t)
        loss.backward()
        self._actor_optim.step()
        return loss.cpu().detach().numpy()

    def compute_actor_loss(self, batch: TorchMiniBatch, q_t, v_t) -> torch.Tensor:
        assert self._policy is not None
        advantage = q_t - v_t
        weights = torch.exp(advantage * self._beta)
        weights = torch.clip(weights, max=self._max_weight)
        dist = self._policy.dist(batch.observations)
        unnormalized_action = torch.atanh(batch.actions.clamp(-0.999999, 0.999999))
        _, log_probs = squash_action(dist, unnormalized_action)
        return -(weights * log_probs).mean()

    def build(self) -> None:
        # setup torch models
        self._build_critic()
        self._build_actor()
        self._build_v_func()

        # setup target networks
        self._targ_q_func = copy.deepcopy(self._q_func)

        if self._use_gpu:
            self.to_gpu(self._use_gpu)
        else:
            self.to_cpu()

        # setup optimizer after the parameters move to GPU
        self._build_critic_optim()
        self._build_actor_optim()
        self._build_v_func_optim()

    def _build_critic(self) -> None:
        self._q_func = create_continuous_q_function(
            self._observation_shape,
            self._action_size,
            self._critic_encoder_factory,
            self._q_func_factory,
            n_ensembles=self._n_critics,
        )

    def _build_critic_optim(self) -> None:
        assert self._q_func is not None
        self._critic_optim = self._critic_optim_factory.create(
            self._q_func.parameters(), lr=self._critic_learning_rate
        )

    def _build_v_func(self) -> None:
        self._v_func = create_value_function(
            self._observation_shape, self._critic_encoder_factory
        )

    def _build_v_func_optim(self) -> None:
        assert self._v_func is not None
        self._v_func_optim = self._critic_optim_factory.create(
            self._v_func.parameters(), lr=self._critic_learning_rate
        )

    def _build_actor(self) -> None:
        self._policy = create_squashed_normal_policy(
            self._observation_shape,
            self._action_size,
            self._actor_encoder_factory,
            use_std_parameter=True,
        )

    def _build_actor_optim(self) -> None:
        assert self._policy is not None
        self._actor_optim = self._actor_optim_factory.create(
            self._policy.parameters(), lr=self._actor_learning_rate
        )


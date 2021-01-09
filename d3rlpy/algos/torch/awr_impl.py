from typing import Any, Optional, Sequence
from abc import ABCMeta, abstractmethod

import numpy as np
import torch
from torch.optim import Optimizer

from ...models.torch import create_value_function, create_normal_policy
from ...models.torch import create_categorical_policy
from ...models.torch import squash_action
from ...models.torch import Policy, NormalPolicy, CategoricalPolicy
from ...models.torch import ValueFunction
from ...optimizers import OptimizerFactory
from ...encoders import EncoderFactory
from ...gpu import Device
from ...preprocessing import Scaler
from ...augmentation import AugmentationPipeline
from ...torch_utility import torch_api, train_api, eval_api, augmentation_api
from .base import TorchImplBase


class AWRBaseImpl(TorchImplBase, metaclass=ABCMeta):

    _actor_learning_rate: float
    _critic_learning_rate: float
    _actor_optim_factory: OptimizerFactory
    _critic_optim_factory: OptimizerFactory
    _actor_encoder_factory: EncoderFactory
    _critic_encoder_factory: EncoderFactory
    _use_gpu: Optional[Device]
    _v_func: Optional[ValueFunction]
    _policy: Optional[Policy]
    _critic_optim: Optional[Optimizer]
    _actor_optim: Optional[Optimizer]

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
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        augmentation: AugmentationPipeline,
    ):
        super().__init__(observation_shape, action_size, scaler, augmentation)
        self._actor_learning_rate = actor_learning_rate
        self._critic_learning_rate = critic_learning_rate
        self._actor_optim_factory = actor_optim_factory
        self._critic_optim_factory = critic_optim_factory
        self._actor_encoder_factory = actor_encoder_factory
        self._critic_encoder_factory = critic_encoder_factory
        self._use_gpu = use_gpu

        # initialized in build
        self._v_func = None
        self._policy = None
        self._critic_optim = None
        self._actor_optim = None

    def build(self) -> None:
        # setup torch models
        self._build_critic()
        self._build_actor()

        if self._use_gpu:
            self.to_gpu(self._use_gpu)
        else:
            self.to_cpu()

        # setup optimizer after the parameters move to GPU
        self._build_critic_optim()
        self._build_actor_optim()

    def _build_critic(self) -> None:
        self._v_func = create_value_function(
            self._observation_shape, self._critic_encoder_factory
        )

    def _build_critic_optim(self) -> None:
        assert self._v_func is not None
        self._critic_optim = self._critic_optim_factory.create(
            self._v_func.parameters(), lr=self._critic_learning_rate
        )

    @abstractmethod
    def _build_actor(self) -> None:
        pass

    def _build_actor_optim(self) -> None:
        assert self._policy is not None
        self._actor_optim = self._actor_optim_factory.create(
            self._policy.parameters(), lr=self._actor_learning_rate
        )

    @train_api
    @torch_api(scaler_targets=["observation"])
    def update_critic(
        self, observation: torch.Tensor, value: torch.Tensor
    ) -> np.ndarray:
        assert self._critic_optim is not None

        self._critic_optim.zero_grad()

        loss = self.compute_critic_loss(observation, value)

        loss.backward()
        self._critic_optim.step()

        return loss.cpu().detach().numpy()

    @augmentation_api(targets=["observation"])
    def compute_critic_loss(
        self, observation: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        assert self._v_func is not None
        return self._v_func.compute_error(observation, value)

    @train_api
    @torch_api(scaler_targets=["observation"])
    def update_actor(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        weight: torch.Tensor,
    ) -> np.ndarray:
        assert self._actor_optim is not None

        self._actor_optim.zero_grad()

        loss = self.compute_actor_loss(observation, action, weight)

        loss.backward()
        self._actor_optim.step()

        return loss.cpu().detach().numpy()

    @augmentation_api(targets=["observation"])
    def compute_actor_loss(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        return self._compute_actor_loss(observation, action, weight)

    @abstractmethod
    def _compute_actor_loss(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        pass

    def _predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        assert self._policy is not None
        return self._policy.best_action(x)

    @eval_api
    @torch_api(scaler_targets=["x"])
    def predict_value(
        self, x: torch.Tensor, *args: Any, **kwargs: Any
    ) -> np.ndarray:
        assert self._v_func is not None
        with torch.no_grad():
            return self._v_func(x).view(-1).cpu().detach().numpy()

    @eval_api
    @torch_api(scaler_targets=["x"])
    def sample_action(self, x: torch.Tensor) -> np.ndarray:
        assert self._policy is not None
        with torch.no_grad():
            return self._policy.sample(x).cpu().detach().numpy()


class AWRImpl(AWRBaseImpl):

    _policy: Optional[NormalPolicy]

    def _build_actor(self) -> None:
        self._policy = create_normal_policy(
            self._observation_shape,
            self._action_size,
            self._actor_encoder_factory,
        )

    def _compute_actor_loss(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        assert self._policy is not None

        dist = self._policy.dist(observation)

        # unnormalize action via inverse tanh function
        unnormalized_action = torch.atanh(action.clamp(-0.999999, 0.999999))

        # compute log probability
        _, log_probs = squash_action(dist, unnormalized_action)

        return -(weight * log_probs).mean()


class DiscreteAWRImpl(AWRBaseImpl):

    _policy: Optional[CategoricalPolicy]

    def _build_actor(self) -> None:
        self._policy = create_categorical_policy(
            self._observation_shape,
            self._action_size,
            self._actor_encoder_factory,
        )

    def _compute_actor_loss(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        assert self._policy is not None
        dist = self._policy.dist(observation)
        log_probs = dist.log_prob(action).view(observation.shape[0], -1)
        return -(weight * log_probs.sum(dim=1, keepdim=True)).mean()

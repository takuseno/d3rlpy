from abc import ABCMeta, abstractmethod
from typing import Optional, Union

import torch
from torch.optim import Optimizer

from ...dataset import Shape
from ...models.builders import (
    create_deterministic_policy,
    create_deterministic_regressor,
    create_discrete_imitator,
    create_probablistic_regressor,
    create_squashed_normal_policy,
)
from ...models.encoders import EncoderFactory
from ...models.optimizers import OptimizerFactory
from ...models.torch import (
    DeterministicRegressor,
    DiscreteImitator,
    Imitator,
    Policy,
    ProbablisticRegressor,
)
from ...torch_utility import TorchMiniBatch, hard_sync, to_device, train_api
from ..base import AlgoImplBase

__all__ = ["BCImpl", "DiscreteBCImpl"]


class BCBaseImpl(AlgoImplBase, metaclass=ABCMeta):

    _learning_rate: float
    _optim_factory: OptimizerFactory
    _encoder_factory: EncoderFactory
    _imitator: Optional[Imitator]
    _optim: Optional[Optimizer]

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        learning_rate: float,
        optim_factory: OptimizerFactory,
        encoder_factory: EncoderFactory,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            device=device,
        )
        self._learning_rate = learning_rate
        self._optim_factory = optim_factory
        self._encoder_factory = encoder_factory

        # initialized in build
        self._imitator = None
        self._optim = None

    def build(self) -> None:
        self._build_network()

        to_device(self, self._device)

        self._build_optim()

    @abstractmethod
    def _build_network(self) -> None:
        pass

    def _build_optim(self) -> None:
        assert self._imitator is not None
        self._optim = self._optim_factory.create(
            self._imitator.parameters(), lr=self._learning_rate
        )

    @train_api
    def update_imitator(self, batch: TorchMiniBatch) -> float:
        assert self._optim is not None

        self._optim.zero_grad()

        loss = self.compute_loss(batch.observations, batch.actions)

        loss.backward()
        self._optim.step()

        return float(loss.cpu().detach().numpy())

    def compute_loss(
        self, obs_t: torch.Tensor, act_t: torch.Tensor
    ) -> torch.Tensor:
        assert self._imitator is not None
        return self._imitator.compute_error(obs_t, act_t)

    def inner_predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        assert self._imitator is not None
        return self._imitator(x)

    def inner_sample_action(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner_predict_best_action(x)

    def inner_predict_value(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError("BC does not support value estimation")


class BCImpl(BCBaseImpl):

    _policy_type: str
    _imitator: Optional[Union[DeterministicRegressor, ProbablisticRegressor]]

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        learning_rate: float,
        optim_factory: OptimizerFactory,
        encoder_factory: EncoderFactory,
        policy_type: str,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            learning_rate=learning_rate,
            optim_factory=optim_factory,
            encoder_factory=encoder_factory,
            device=device,
        )
        self._policy_type = policy_type

    def _build_network(self) -> None:
        if self._policy_type == "deterministic":
            self._imitator = create_deterministic_regressor(
                self._observation_shape,
                self._action_size,
                self._encoder_factory,
            )
        elif self._policy_type == "stochastic":
            self._imitator = create_probablistic_regressor(
                self._observation_shape,
                self._action_size,
                self._encoder_factory,
                min_logstd=-4.0,
                max_logstd=15.0,
            )
        else:
            raise ValueError("invalid policy_type: {self._policy_type}")

    @property
    def policy(self) -> Policy:
        assert self._imitator

        policy: Policy
        if self._policy_type == "deterministic":
            policy = create_deterministic_policy(
                self._observation_shape,
                self._action_size,
                self._encoder_factory,
            )
        elif self._policy_type == "stochastic":
            policy = create_squashed_normal_policy(
                self._observation_shape,
                self._action_size,
                self._encoder_factory,
                min_logstd=-4.0,
                max_logstd=15.0,
            )
        else:
            raise ValueError(f"invalid policy_type: {self._policy_type}")

        # copy parameters
        hard_sync(policy, self._imitator)

        return policy

    @property
    def policy_optim(self) -> Optimizer:
        assert self._optim
        return self._optim


class DiscreteBCImpl(BCBaseImpl):

    _beta: float
    _imitator: Optional[DiscreteImitator]

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        learning_rate: float,
        optim_factory: OptimizerFactory,
        encoder_factory: EncoderFactory,
        beta: float,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            learning_rate=learning_rate,
            optim_factory=optim_factory,
            encoder_factory=encoder_factory,
            device=device,
        )
        self._beta = beta

    def _build_network(self) -> None:
        self._imitator = create_discrete_imitator(
            self._observation_shape,
            self._action_size,
            self._beta,
            self._encoder_factory,
        )

    def inner_predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        assert self._imitator is not None
        return self._imitator(x).argmax(dim=1)

    def compute_loss(
        self, obs_t: torch.Tensor, act_t: torch.Tensor
    ) -> torch.Tensor:
        assert self._imitator is not None
        return self._imitator.compute_error(obs_t, act_t.long())

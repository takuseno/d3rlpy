from abc import ABCMeta
from typing import Union

import torch
from torch.optim import Optimizer

from ....dataset import Shape
from ....models.torch import (
    DeterministicPolicy,
    DeterministicRegressor,
    DiscreteImitator,
    Imitator,
    Policy,
    ProbablisticRegressor,
    SquashedNormalPolicy,
    compute_output_size,
)
from ....torch_utility import TorchMiniBatch, hard_sync, train_api
from ..base import QLearningAlgoImplBase

__all__ = ["BCImpl", "DiscreteBCImpl"]


class BCBaseImpl(QLearningAlgoImplBase, metaclass=ABCMeta):
    _learning_rate: float
    _imitator: Imitator
    _optim: Optimizer

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        imitator: Imitator,
        optim: Optimizer,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            device=device,
        )
        self._imitator = imitator
        self._optim = optim

    @train_api
    def update_imitator(self, batch: TorchMiniBatch) -> float:
        self._optim.zero_grad()

        loss = self.compute_loss(batch.observations, batch.actions)

        loss.backward()
        self._optim.step()

        return float(loss.cpu().detach().numpy())

    def compute_loss(
        self, obs_t: torch.Tensor, act_t: torch.Tensor
    ) -> torch.Tensor:
        return self._imitator.compute_error(obs_t, act_t)

    def inner_predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        return self._imitator(x)

    def inner_sample_action(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner_predict_best_action(x)

    def inner_predict_value(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError("BC does not support value estimation")


class BCImpl(BCBaseImpl):
    _policy_type: str
    _imitator: Union[DeterministicRegressor, ProbablisticRegressor]

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        imitator: Union[DeterministicRegressor, ProbablisticRegressor],
        optim: Optimizer,
        policy_type: str,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            imitator=imitator,
            optim=optim,
            device=device,
        )
        self._policy_type = policy_type

    @property
    def policy(self) -> Policy:
        policy: Policy
        hidden_size = compute_output_size(
            [self._observation_shape, (self._action_size,)],
            self._imitator.encoder,
            device=self._device,
        )
        if self._policy_type == "deterministic":
            hidden_size = compute_output_size(
                [self._observation_shape, (self._action_size,)],
                self._imitator.encoder,
                device=self._device,
            )
            policy = DeterministicPolicy(
                encoder=self._imitator.encoder,
                hidden_size=hidden_size,
                action_size=self._action_size,
            )
        elif self._policy_type == "stochastic":
            return SquashedNormalPolicy(
                encoder=self._imitator.encoder,
                hidden_size=hidden_size,
                action_size=self._action_size,
                min_logstd=-4.0,
                max_logstd=15.0,
                use_std_parameter=False,
            )
        else:
            raise ValueError(f"invalid policy_type: {self._policy_type}")
        policy.to(self._device)

        # copy parameters
        hard_sync(policy, self._imitator)

        return policy

    @property
    def policy_optim(self) -> Optimizer:
        return self._optim


class DiscreteBCImpl(BCBaseImpl):
    _beta: float
    _imitator: DiscreteImitator

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        imitator: DiscreteImitator,
        optim: Optimizer,
        beta: float,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            imitator=imitator,
            optim=optim,
            device=device,
        )
        self._beta = beta

    def inner_predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        return self._imitator(x).argmax(dim=1)

    def compute_loss(
        self, obs_t: torch.Tensor, act_t: torch.Tensor
    ) -> torch.Tensor:
        return self._imitator.compute_error(obs_t, act_t.long())

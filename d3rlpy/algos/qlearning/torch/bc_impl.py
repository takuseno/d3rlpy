import dataclasses
from abc import ABCMeta, abstractmethod
from typing import Callable, Union

import torch
from torch.optim import Optimizer

from ....dataclass_utils import asdict_as_float
from ....models.torch import (
    CategoricalPolicy,
    DeterministicPolicy,
    DiscreteImitationLoss,
    ImitationLoss,
    NormalPolicy,
    Policy,
    compute_deterministic_imitation_loss,
    compute_discrete_imitation_loss,
    compute_stochastic_imitation_loss,
)
from ....optimizers import OptimizerWrapper
from ....torch_utility import CudaGraphWrapper, Modules, TorchMiniBatch
from ....types import Shape, TorchObservation
from ..base import QLearningAlgoImplBase

__all__ = ["BCImpl", "DiscreteBCImpl", "BCModules", "DiscreteBCModules"]


@dataclasses.dataclass(frozen=True)
class BCBaseModules(Modules):
    optim: OptimizerWrapper


class BCBaseImpl(QLearningAlgoImplBase, metaclass=ABCMeta):
    _modules: BCBaseModules
    _compute_imitator_grad: Callable[[TorchMiniBatch], ImitationLoss]

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: BCBaseModules,
        compiled: bool,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            device=device,
        )
        self._compute_imitator_grad = (
            CudaGraphWrapper(self.compute_imitator_grad)
            if compiled
            else self.compute_imitator_grad
        )

    def compute_imitator_grad(self, batch: TorchMiniBatch) -> ImitationLoss:
        self._modules.optim.zero_grad()
        loss = self.compute_loss(batch.observations, batch.actions)
        loss.loss.backward()
        return loss

    def update_imitator(self, batch: TorchMiniBatch) -> dict[str, float]:
        loss = self._compute_imitator_grad(batch)
        self._modules.optim.step()
        return asdict_as_float(loss)

    @abstractmethod
    def compute_loss(
        self, obs_t: TorchObservation, act_t: torch.Tensor
    ) -> ImitationLoss:
        pass

    def inner_sample_action(self, x: TorchObservation) -> torch.Tensor:
        return self.inner_predict_best_action(x)

    def inner_predict_value(
        self, x: TorchObservation, action: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError("BC does not support value estimation")

    def inner_update(
        self, batch: TorchMiniBatch, grad_step: int
    ) -> dict[str, float]:
        return self.update_imitator(batch)


@dataclasses.dataclass(frozen=True)
class BCModules(BCBaseModules):
    imitator: Union[DeterministicPolicy, NormalPolicy]


class BCImpl(BCBaseImpl):
    _modules: BCModules
    _policy_type: str

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: BCModules,
        policy_type: str,
        compiled: bool,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            compiled=compiled,
            device=device,
        )
        self._policy_type = policy_type

    def inner_predict_best_action(self, x: TorchObservation) -> torch.Tensor:
        return self._modules.imitator(x).squashed_mu

    def compute_loss(
        self, obs_t: TorchObservation, act_t: torch.Tensor
    ) -> ImitationLoss:
        if self._policy_type == "deterministic":
            assert isinstance(self._modules.imitator, DeterministicPolicy)
            return compute_deterministic_imitation_loss(
                self._modules.imitator, obs_t, act_t
            )
        elif self._policy_type == "stochastic":
            assert isinstance(self._modules.imitator, NormalPolicy)
            return compute_stochastic_imitation_loss(
                self._modules.imitator, obs_t, act_t
            )
        else:
            raise ValueError(f"invalid policy_type: {self._policy_type}")

    @property
    def policy(self) -> Policy:
        return self._modules.imitator

    @property
    def policy_optim(self) -> Optimizer:
        return self._modules.optim.optim


@dataclasses.dataclass(frozen=True)
class DiscreteBCModules(BCBaseModules):
    imitator: CategoricalPolicy


class DiscreteBCImpl(BCBaseImpl):
    _modules: DiscreteBCModules
    _beta: float

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: DiscreteBCModules,
        beta: float,
        compiled: bool,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            compiled=compiled,
            device=device,
        )
        self._beta = beta

    def inner_predict_best_action(self, x: TorchObservation) -> torch.Tensor:
        return self._modules.imitator(x).logits.argmax(dim=1)

    def compute_loss(
        self, obs_t: TorchObservation, act_t: torch.Tensor
    ) -> DiscreteImitationLoss:
        return compute_discrete_imitation_loss(
            policy=self._modules.imitator,
            x=obs_t,
            action=act_t.long(),
            beta=self._beta,
        )

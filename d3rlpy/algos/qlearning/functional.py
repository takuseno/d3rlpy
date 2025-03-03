from typing import Optional, Protocol

import torch
from torch import nn

from ...models.torch.policies import Policy
from ...torch_utility import Modules, TorchMiniBatch
from ...types import Shape, TorchObservation
from .base import QLearningAlgoImplBase

__all__ = [
    "Updater",
    "ActionSampler",
    "ValuePredictor",
    "FunctionalQLearningAlgoImplBase",
]


class Updater(Protocol):
    def __call__(
        self, batch: TorchMiniBatch, grad_step: int
    ) -> dict[str, float]: ...


class ActionSampler(Protocol):
    def __call__(self, x: TorchObservation) -> torch.Tensor: ...


class ValuePredictor(Protocol):
    def __call__(
        self, x: TorchObservation, action: torch.Tensor
    ) -> torch.Tensor: ...


class FunctionalQLearningAlgoImplBase(QLearningAlgoImplBase):
    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: Modules,
        updater: Updater,
        exploit_action_sampler: ActionSampler,
        explore_action_sampler: ActionSampler,
        value_predictor: ValuePredictor,
        q_function: nn.ModuleList,
        q_function_optim: torch.optim.Optimizer,
        policy: Optional[Policy],
        policy_optim: Optional[torch.optim.Optimizer],
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            device=device,
        )
        self._updater = updater
        self._exploit_action_sampler = exploit_action_sampler
        self._explore_action_sampler = explore_action_sampler
        self._value_predictor = value_predictor
        self._q_function = q_function
        self._q_function_optim = q_function_optim
        self._policy = policy
        self._policy_optim = policy_optim

    def inner_update(
        self, batch: TorchMiniBatch, grad_step: int
    ) -> dict[str, float]:
        return self._updater(batch, grad_step)

    def inner_predict_best_action(self, x: TorchObservation) -> torch.Tensor:
        return self._exploit_action_sampler(x)

    def inner_sample_action(self, x: TorchObservation) -> torch.Tensor:
        return self._explore_action_sampler(x)

    def inner_predict_value(
        self, x: TorchObservation, action: torch.Tensor
    ) -> torch.Tensor:
        return self._value_predictor(x, action)

    @property
    def policy(self) -> Policy:
        assert self._policy
        return self._policy

    @property
    def policy_optim(self) -> torch.optim.Optimizer:
        assert self._policy_optim
        return self._policy_optim

    @property
    def q_function(self) -> nn.ModuleList:
        return self._q_function

    @property
    def q_function_optim(self) -> torch.optim.Optimizer:
        return self._q_function_optim

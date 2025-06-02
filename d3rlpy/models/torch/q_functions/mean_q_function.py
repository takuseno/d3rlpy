from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from ....types import TorchObservation
from ..encoders import Encoder, EncoderWithAction
from .base import (
    ContinuousQFunction,
    ContinuousQFunctionForwarder,
    DiscreteQFunction,
    DiscreteQFunctionForwarder,
    QFunctionOutput,
    TargetOutput,
)
from .utility import compute_huber_loss, compute_reduce, pick_value_by_action

__all__ = [
    "DiscreteMeanQFunction",
    "ContinuousMeanQFunction",
    "DiscreteMeanQFunctionForwarder",
    "ContinuousMeanQFunctionForwarder",
]


class DiscreteMeanQFunction(DiscreteQFunction):
    _encoder: Encoder
    _fc: nn.Linear

    def __init__(self, encoder: Encoder, hidden_size: int, action_size: int):
        super().__init__()
        self._encoder = encoder
        self._fc = nn.Linear(hidden_size, action_size)

    def forward(self, x: TorchObservation) -> QFunctionOutput:
        return QFunctionOutput(
            q_value=self._fc(self._encoder(x)),
            quantiles=None,
            taus=None,
        )

    @property
    def encoder(self) -> Encoder:
        return self._encoder


class DiscreteMeanQFunctionForwarder(DiscreteQFunctionForwarder):
    _q_func: DiscreteMeanQFunction
    _action_size: int

    def __init__(self, q_func: DiscreteMeanQFunction, action_size: int):
        self._q_func = q_func
        self._action_size = action_size

    def compute_expected_q(self, x: TorchObservation) -> torch.Tensor:
        return self._q_func(x).q_value

    def compute_error(
        self,
        observations: TorchObservation,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        target: TargetOutput,
        terminals: torch.Tensor,
        gamma: Union[float, torch.Tensor] = 0.99,
        reduction: str = "mean",
    ) -> torch.Tensor:
        one_hot = F.one_hot(actions.view(-1), num_classes=self._action_size)
        value = (self._q_func(observations).q_value * one_hot.float()).sum(
            dim=1, keepdim=True
        )
        y = rewards + gamma * target.q_value * (1 - terminals)
        loss = compute_huber_loss(value, y)
        return compute_reduce(loss, reduction)

    def compute_target(
        self, x: TorchObservation, action: Optional[torch.Tensor] = None
    ) -> TargetOutput:
        if action is None:
            value = self._q_func(x).q_value
        else:
            value = pick_value_by_action(
                self._q_func(x).q_value, action, keepdim=True
            )
        return TargetOutput(q_value=value)

    def set_q_func(self, q_func: DiscreteQFunction) -> None:
        self._q_func = q_func


class ContinuousMeanQFunction(ContinuousQFunction):
    _encoder: EncoderWithAction
    _fc: nn.Linear

    def __init__(self, encoder: EncoderWithAction, hidden_size: int):
        super().__init__()
        self._encoder = encoder
        self._fc = nn.Linear(hidden_size, 1)

    def forward(
        self, x: TorchObservation, action: torch.Tensor
    ) -> QFunctionOutput:
        return QFunctionOutput(
            q_value=self._fc(self._encoder(x, action)),
            quantiles=None,
            taus=None,
        )

    @property
    def encoder(self) -> EncoderWithAction:
        return self._encoder


class ContinuousMeanQFunctionForwarder(ContinuousQFunctionForwarder):
    _q_func: ContinuousMeanQFunction

    def __init__(self, q_func: ContinuousMeanQFunction):
        self._q_func = q_func

    def compute_expected_q(
        self, x: TorchObservation, action: torch.Tensor
    ) -> torch.Tensor:
        return self._q_func(x, action).q_value

    def compute_error(
        self,
        observations: TorchObservation,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        target: TargetOutput,
        terminals: torch.Tensor,
        gamma: Union[float, torch.Tensor] = 0.99,
        reduction: str = "mean",
    ) -> torch.Tensor:
        value = self._q_func(observations, actions).q_value
        y = rewards + gamma * target.q_value * (1 - terminals)
        loss = F.mse_loss(value, y, reduction="none")
        return compute_reduce(loss, reduction)

    def compute_target(
        self, x: TorchObservation, action: torch.Tensor
    ) -> TargetOutput:
        return TargetOutput(q_value=self._q_func(x, action).q_value)

    def set_q_func(self, q_func: ContinuousQFunction) -> None:
        self._q_func = q_func

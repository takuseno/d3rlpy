import dataclasses
from abc import ABCMeta, abstractmethod
from typing import NamedTuple, Optional, Union

import torch
from torch import nn

from ....types import TorchObservation
from ..encoders import Encoder, EncoderWithAction

__all__ = [
    "DiscreteQFunction",
    "ContinuousQFunction",
    "ContinuousQFunctionForwarder",
    "DiscreteQFunctionForwarder",
    "QFunctionOutput",
    "TargetOutput",
]


class QFunctionOutput(NamedTuple):
    q_value: torch.Tensor
    quantiles: Optional[torch.Tensor]
    taus: Optional[torch.Tensor]


@dataclasses.dataclass(frozen=True)
class TargetOutput:
    q_value: torch.Tensor

    def add(self, v: torch.Tensor) -> "TargetOutput":
        assert v.shape == (self.q_value.shape[0], 1)
        return TargetOutput(self.q_value + v)


class ContinuousQFunction(nn.Module, metaclass=ABCMeta):  # type: ignore
    @abstractmethod
    def forward(
        self, x: TorchObservation, action: torch.Tensor
    ) -> QFunctionOutput:
        pass

    def __call__(
        self, x: TorchObservation, action: torch.Tensor
    ) -> QFunctionOutput:
        return super().__call__(x, action)  # type: ignore

    @property
    @abstractmethod
    def encoder(self) -> EncoderWithAction:
        pass


class DiscreteQFunction(nn.Module, metaclass=ABCMeta):  # type: ignore
    @abstractmethod
    def forward(self, x: TorchObservation) -> QFunctionOutput:
        pass

    def __call__(self, x: TorchObservation) -> QFunctionOutput:
        return super().__call__(x)  # type: ignore

    @property
    @abstractmethod
    def encoder(self) -> Encoder:
        pass


class ContinuousQFunctionForwarder(metaclass=ABCMeta):
    @abstractmethod
    def compute_expected_q(
        self, x: TorchObservation, action: torch.Tensor
    ) -> torch.Tensor:
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def compute_target(
        self, x: TorchObservation, action: torch.Tensor
    ) -> TargetOutput:
        pass

    @abstractmethod
    def set_q_func(self, q_func: ContinuousQFunction) -> None:
        pass


class DiscreteQFunctionForwarder(metaclass=ABCMeta):
    @abstractmethod
    def compute_expected_q(self, x: TorchObservation) -> torch.Tensor:
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def compute_target(
        self, x: TorchObservation, action: Optional[torch.Tensor] = None
    ) -> TargetOutput:
        pass

    @abstractmethod
    def set_q_func(self, q_func: DiscreteQFunction) -> None:
        pass

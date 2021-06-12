from abc import ABCMeta, abstractmethod
from typing import Optional

import torch

from ..encoders import Encoder, EncoderWithAction


class QFunction(metaclass=ABCMeta):
    @abstractmethod
    def compute_error(
        self,
        obs_t: torch.Tensor,
        act_t: torch.Tensor,
        rew_tp1: torch.Tensor,
        q_tp1: torch.Tensor,
        ter_tp1: torch.Tensor,
        gamma: float = 0.99,
        reduction: str = "mean",
    ) -> torch.Tensor:
        pass

    @property
    def action_size(self) -> int:
        pass


class DiscreteQFunction(QFunction):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def compute_target(
        self, x: torch.Tensor, action: Optional[torch.Tensor]
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @property
    def encoder(self) -> Encoder:
        pass


class ContinuousQFunction(QFunction):
    @abstractmethod
    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def compute_target(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def __call__(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        pass

    @property
    def encoder(self) -> EncoderWithAction:
        pass

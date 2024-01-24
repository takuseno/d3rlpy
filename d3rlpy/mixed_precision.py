from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
from typing import Any, Generator

import torch
from torch.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim import Optimizer

__all__ = ["PrecisionScaler", "NoCastPrecisionScaler", "MixedPrecisionScaler"]


class PrecisionScaler(metaclass=ABCMeta):
    @abstractmethod
    def scale_and_backward(
        self, optimizer: Optimizer, loss: torch.Tensor
    ) -> None:
        ...

    @abstractmethod
    def step(self, optimizer: Optimizer) -> None:
        ...

    @abstractmethod
    @contextmanager
    def autocast(self) -> Generator[Any, None, None]:
        ...


class NoCastPrecisionScaler(PrecisionScaler):
    def scale_and_backward(
        self, optimizer: Optimizer, loss: torch.Tensor
    ) -> None:
        loss.backward()

    def step(self, optimizer: Optimizer) -> None:
        optimizer.step()

    @contextmanager
    def autocast(self) -> Generator[Any, None, None]:
        yield None


class MixedPrecisionScaler(PrecisionScaler):
    _scaler: GradScaler

    def __init__(self):
        self._scaler = GradScaler()

    def scale_and_backward(
        self, optimizer: Optimizer, loss: torch.Tensor
    ) -> None:
        self._scaler.scale(loss).backward()
        self._scaler.unscale_(optimizer)

    def step(self, optimizer: Optimizer) -> None:
        self._scaler.step(optimizer)
        self._scaler.update()

    @contextmanager
    def autocast(self) -> Generator[Any, None, None]:
        yield autocast("cuda", dtype=torch.float16)

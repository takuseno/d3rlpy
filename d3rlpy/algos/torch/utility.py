from typing import Optional

import torch
from typing_extensions import Protocol

from ...models.torch import (
    EnsembleContinuousQFunction,
    EnsembleDiscreteQFunction,
)

__all__ = ["DiscreteQFunctionMixin", "ContinuousQFunctionMixin"]


class _DiscreteQFunctionProtocol(Protocol):
    _q_func: Optional[EnsembleDiscreteQFunction]


class _ContinuousQFunctionProtocol(Protocol):
    _q_func: Optional[EnsembleContinuousQFunction]


class DiscreteQFunctionMixin:
    def inner_predict_value(
        self: _DiscreteQFunctionProtocol, x: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        assert self._q_func is not None
        values = self._q_func(x, reduction="mean")
        return values[torch.arange(0, x.size(0)), action].reshape(-1)


class ContinuousQFunctionMixin:
    def inner_predict_value(
        self: _ContinuousQFunctionProtocol,
        x: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        assert self._q_func is not None
        return self._q_func(x, action, reduction="mean").reshape(-1)

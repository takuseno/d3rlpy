import dataclasses

import torch
from typing_extensions import Protocol

from ....models.torch import (
    ContinuousEnsembleQFunctionForwarder,
    DiscreteEnsembleQFunctionForwarder,
)

__all__ = ["DiscreteQFunctionMixin", "ContinuousQFunctionMixin", "CriticLoss"]


class _DiscreteQFunctionProtocol(Protocol):
    _q_func_forwarder: DiscreteEnsembleQFunctionForwarder


class _ContinuousQFunctionProtocol(Protocol):
    _q_func_forwarder: ContinuousEnsembleQFunctionForwarder


class DiscreteQFunctionMixin:
    def inner_predict_value(
        self: _DiscreteQFunctionProtocol, x: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        values = self._q_func_forwarder.compute_expected_q(x, reduction="mean")
        flat_action = action.reshape(-1)
        return values[torch.arange(0, x.size(0)), flat_action].reshape(-1)


class ContinuousQFunctionMixin:
    def inner_predict_value(
        self: _ContinuousQFunctionProtocol,
        x: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        return self._q_func_forwarder.compute_expected_q(
            x, action, reduction="mean"
        ).reshape(-1)


@dataclasses.dataclass(frozen=True)
class CriticLoss:
    td_loss: torch.Tensor
    loss: torch.Tensor = dataclasses.field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "loss", self.get_loss())

    def get_loss(self):
        return self.td_loss

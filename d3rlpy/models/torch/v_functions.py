from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

from ...types import TorchObservation
from .encoders import Encoder

__all__ = ["ValueFunction", "compute_v_function_error"]


class ValueFunction(nn.Module):  # type: ignore
    _encoder: Encoder
    _fc: nn.Linear

    def __init__(self, encoder: Encoder, hidden_size: int):
        super().__init__()
        self._encoder = encoder
        self._fc = nn.Linear(hidden_size, 1)

    def forward(self, x: TorchObservation) -> torch.Tensor:
        h = self._encoder(x)
        return cast(torch.Tensor, self._fc(h))

    def __call__(self, x: TorchObservation) -> torch.Tensor:
        return cast(torch.Tensor, super().__call__(x))


def compute_v_function_error(
    v_function: ValueFunction,
    observations: TorchObservation,
    target: torch.Tensor,
) -> torch.Tensor:
    v_t = v_function(observations)
    loss = F.mse_loss(v_t, target)
    return loss

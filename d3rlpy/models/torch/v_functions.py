import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import cast
from .encoders import Encoder


class ValueFunction(nn.Module):
    _encoder: Encoder
    _fc: nn.Linear

    def __init__(self, encoder: Encoder):
        super().__init__()
        self._encoder = encoder
        self._fc = nn.Linear(encoder.get_feature_size(), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._encoder(x)
        return cast(torch.Tensor, self._fc(h))

    def compute_error(
        self, obs_t: torch.Tensor, ret_t: torch.Tensor
    ) -> torch.Tensor:
        v_t = self.forward(obs_t)
        loss = F.mse_loss(v_t, ret_t)
        return loss

import math

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "Scaler",
    "HyperDense",
    "HyperMLP",
    "HyperInputEncoder",
    "HyperLERPBlock",
]


class Scaler(nn.Module):  # type: ignore
    def __init__(self, dim: int, init: float = 1.0, scale: float = 1.0):
        super().__init__()
        self._scaler = nn.Parameter(
            scale * torch.ones(dim, dtype=torch.float32), requires_grad=True
        )
        self._forward_scaler = init / scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._scaler * self._forward_scaler * x


class HyperDense(nn.Module):  # type: ignore
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self._dense = nn.Linear(in_features, out_features, bias=False)
        nn.init.orthogonal_(self._dense.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._dense(x)


class HyperMLP(nn.Module):  # type: ignore
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        scaler_init: float,
        scaler_scale: float,
        eps: float = 1e-8,
    ):
        super().__init__()
        self._w1 = HyperDense(in_features, hidden_dim)
        self._scaler = Scaler(hidden_dim, scaler_init, scaler_scale)
        self._w2 = HyperDense(hidden_dim, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._w1(x)
        x = self._scaler(x)
        x = torch.relu(x) + self._eps
        x = self._w2(x)
        x = F.normalize(x, dim=-1)
        return x


class HyperInputEncoder(nn.Module):  # type: ignore
    def __init__(
        self,
        in_features: int,
        out_features: int,
        scaler_init: float,
        scaler_scale: float,
        c_shift: float,
    ):
        self._w = HyperDense(in_features, out_features)
        self._scaler = Scaler(out_features, scaler_init, scaler_scale)
        self._c_shift = c_shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_dim = self._c_shif * torch.ones(
            x.shape[:-1] + (1,), dtype=x.dtype, device=x.device
        )
        x = torch.cat([x, new_dim], dim=-1)
        x = F.normalize(x, dim=-1)
        x = self._w(x)
        x = self._scaler(x)
        x = F.normalize(x, dim=-1)
        return x


class HyperLERPBlock(nn.Module):  # type: ignore
    def __init__(
        self,
        hidden_dim: int,
        scaler_init: float,
        scaler_scale: float,
        alpha_init: float,
        alpha_scale: float,
        expansion: int = 4,
    ):
        self._mlp = HyperMLP(
            in_features=hidden_dim,
            hidden_dim=hidden_dim * expansion,
            scaler_init=scaler_init / math.sqrt(expansion),
            scaler_scale=scaler_scale / math.sqrt(expansion),
        )
        self._alpha_scaler = Scaler(
            dim=hidden_dim,
            init=alpha_init,
            scale=alpha_scale,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self._mlp(x)
        x = residual + self._alpha_scaler(x - residual)
        x = F.normalize(x, dim=-1)
        return x

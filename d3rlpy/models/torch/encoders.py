from abc import ABCMeta, abstractmethod
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from ...itertools import last_flag
from ...types import Shape, TorchObservation
from .layers import HyperInputEncoder, HyperLERPBlock

__all__ = [
    "Encoder",
    "EncoderWithAction",
    "PixelEncoder",
    "PixelEncoderWithAction",
    "VectorEncoder",
    "VectorEncoderWithAction",
    "SimBaEncoder",
    "SimBaEncoderWithAction",
    "SimbaV2Encoder",
    "SimbaV2EncoderWithAction",
    "compute_output_size",
]


class Encoder(nn.Module, metaclass=ABCMeta):  # type: ignore
    @abstractmethod
    def forward(self, x: TorchObservation) -> torch.Tensor:
        pass

    def __call__(self, x: TorchObservation) -> torch.Tensor:
        return super().__call__(x)


class EncoderWithAction(nn.Module, metaclass=ABCMeta):  # type: ignore
    @abstractmethod
    def forward(
        self, x: TorchObservation, action: torch.Tensor
    ) -> torch.Tensor:
        pass

    def __call__(
        self, x: TorchObservation, action: torch.Tensor
    ) -> torch.Tensor:
        return super().__call__(x, action)


class PixelEncoder(Encoder):
    _cnn_layers: nn.Module
    _last_layers: nn.Module

    def __init__(
        self,
        observation_shape: Sequence[int],
        filters: Optional[list[list[int]]] = None,
        feature_size: int = 512,
        use_batch_norm: bool = False,
        dropout_rate: Optional[float] = False,
        activation: nn.Module = nn.ReLU(),
        exclude_last_activation: bool = False,
        last_activation: Optional[nn.Module] = None,
    ):
        super().__init__()

        # default architecture is based on Nature DQN paper.
        if filters is None:
            filters = [[32, 8, 4], [64, 4, 2], [64, 3, 1]]
        if feature_size is None:
            feature_size = 512

        # convolutional layers
        cnn_layers = []
        in_channels = [observation_shape[0]] + [f[0] for f in filters[:-1]]
        for in_channel, f in zip(in_channels, filters):
            out_channel, kernel_size, stride = f
            conv = nn.Conv2d(
                in_channel, out_channel, kernel_size=kernel_size, stride=stride
            )
            cnn_layers.append(conv)
            cnn_layers.append(activation)

            # use batch normalization layer
            if use_batch_norm:
                cnn_layers.append(nn.BatchNorm2d(out_channel))

            # use dropout layer
            if dropout_rate is not None:
                cnn_layers.append(nn.Dropout2d(dropout_rate))
        self._cnn_layers = nn.Sequential(*cnn_layers)

        # compute output shape of CNN layers
        x = torch.rand((1,) + tuple(observation_shape))
        with torch.no_grad():
            cnn_output_size = self._cnn_layers(x).view(1, -1).shape[1]

        # last dense layer
        layers: list[nn.Module] = []
        layers.append(nn.Linear(cnn_output_size, feature_size))
        if not exclude_last_activation:
            layers.append(last_activation if last_activation else activation)
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(feature_size))
        if dropout_rate is not None:
            layers.append(nn.Dropout(dropout_rate))

        self._last_layers = nn.Sequential(*layers)

    def forward(self, x: TorchObservation) -> torch.Tensor:
        assert isinstance(x, torch.Tensor)
        h = self._cnn_layers(x)
        return self._last_layers(h.reshape(x.shape[0], -1))


class PixelEncoderWithAction(EncoderWithAction):
    _cnn_layers: nn.Module
    _last_layers: nn.Module
    _discrete_action: bool
    _action_size: int

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        filters: Optional[list[list[int]]] = None,
        feature_size: int = 512,
        use_batch_norm: bool = False,
        dropout_rate: Optional[float] = False,
        discrete_action: bool = False,
        activation: nn.Module = nn.ReLU(),
        exclude_last_activation: bool = False,
        last_activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        self._discrete_action = discrete_action
        self._action_size = action_size

        # default architecture is based on Nature DQN paper.
        if filters is None:
            filters = [[32, 8, 4], [64, 4, 2], [64, 3, 1]]
        if feature_size is None:
            feature_size = 512

        # convolutional layers
        cnn_layers = []
        in_channels = [observation_shape[0]] + [f[0] for f in filters[:-1]]
        for in_channel, f in zip(in_channels, filters):
            out_channel, kernel_size, stride = f
            conv = nn.Conv2d(
                in_channel, out_channel, kernel_size=kernel_size, stride=stride
            )
            cnn_layers.append(conv)
            cnn_layers.append(activation)

            # use batch normalization layer
            if use_batch_norm:
                cnn_layers.append(nn.BatchNorm2d(out_channel))

            # use dropout layer
            if dropout_rate is not None:
                cnn_layers.append(nn.Dropout2d(dropout_rate))
        self._cnn_layers = nn.Sequential(*cnn_layers)

        # compute output shape of CNN layers
        x = torch.rand((1,) + tuple(observation_shape))
        with torch.no_grad():
            cnn_output_size = self._cnn_layers(x).view(1, -1).shape[1]

        # last dense layer
        layers: list[nn.Module] = []
        layers.append(nn.Linear(cnn_output_size + action_size, feature_size))
        if not exclude_last_activation:
            layers.append(last_activation if last_activation else activation)
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(feature_size))
        if dropout_rate is not None:
            layers.append(nn.Dropout(dropout_rate))
        self._last_layers = nn.Sequential(*layers)

    def forward(
        self, x: TorchObservation, action: torch.Tensor
    ) -> torch.Tensor:
        assert isinstance(x, torch.Tensor)
        h = self._cnn_layers(x)

        if self._discrete_action:
            action = F.one_hot(
                action.view(-1).long(), num_classes=self._action_size
            ).float()

        # cocat feature and action
        h = torch.cat([h.reshape(h.shape[0], -1), action], dim=1)

        return self._last_layers(h)


class VectorEncoder(Encoder):
    _layers: nn.Module

    def __init__(
        self,
        observation_shape: Sequence[int],
        hidden_units: Optional[Sequence[int]] = None,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        dropout_rate: Optional[float] = None,
        activation: nn.Module = nn.ReLU(),
        exclude_last_activation: bool = False,
        last_activation: Optional[nn.Module] = None,
    ):
        super().__init__()

        if hidden_units is None:
            hidden_units = [256, 256]

        layers = []
        in_units = [observation_shape[0]] + list(hidden_units[:-1])
        for is_last, (in_unit, out_unit) in last_flag(
            zip(in_units, hidden_units)
        ):
            layers.append(nn.Linear(in_unit, out_unit))
            if not is_last or not exclude_last_activation:
                if is_last and last_activation:
                    layers.append(last_activation)
                else:
                    layers.append(activation)
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_unit))
            if use_layer_norm:
                layers.append(nn.LayerNorm(out_unit))
            if dropout_rate is not None:
                layers.append(nn.Dropout(dropout_rate))
        self._layers = nn.Sequential(*layers)

    def forward(self, x: TorchObservation) -> torch.Tensor:
        assert isinstance(x, torch.Tensor)
        return self._layers(x)


class VectorEncoderWithAction(EncoderWithAction):
    _layers: nn.Module
    _action_size: int
    _discrete_action: bool

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        hidden_units: Optional[Sequence[int]] = None,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        dropout_rate: Optional[float] = None,
        discrete_action: bool = False,
        activation: nn.Module = nn.ReLU(),
        exclude_last_activation: bool = False,
        last_activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        self._action_size = action_size
        self._discrete_action = discrete_action

        if hidden_units is None:
            hidden_units = [256, 256]

        layers = []
        in_units = [observation_shape[0] + action_size] + list(
            hidden_units[:-1]
        )
        for is_last, (in_unit, out_unit) in last_flag(
            zip(in_units, hidden_units)
        ):
            layers.append(nn.Linear(in_unit, out_unit))
            if not is_last or not exclude_last_activation:
                if is_last and last_activation:
                    layers.append(last_activation)
                else:
                    layers.append(activation)
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_unit))
            if use_layer_norm:
                layers.append(nn.LayerNorm(out_unit))
            if dropout_rate is not None:
                layers.append(nn.Dropout(dropout_rate))
        self._layers = nn.Sequential(*layers)

    def forward(
        self, x: TorchObservation, action: torch.Tensor
    ) -> torch.Tensor:
        assert isinstance(x, torch.Tensor)
        if self._discrete_action:
            action = F.one_hot(
                action.view(-1).long(), num_classes=self._action_size
            ).float()
        x = torch.cat([x, action], dim=1)
        return self._layers(x)


class SimBaBlock(nn.Module):  # type: ignore
    def __init__(self, input_size: int, hidden_size: int, out_size: int):
        super().__init__()
        layers = [
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size),
        ]
        self._layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self._layers(x)


class SimBaEncoder(Encoder):
    def __init__(
        self,
        observation_shape: Sequence[int],
        hidden_size: int,
        output_size: int,
        n_blocks: int,
    ):
        super().__init__()
        layers = [
            nn.Linear(observation_shape[0], output_size),
            *[
                SimBaBlock(output_size, hidden_size, output_size)
                for _ in range(n_blocks)
            ],
            nn.LayerNorm(output_size),
        ]
        self._layers = nn.Sequential(*layers)

    def forward(self, x: TorchObservation) -> torch.Tensor:
        assert isinstance(x, torch.Tensor)
        return self._layers(x)


class SimBaEncoderWithAction(EncoderWithAction):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        hidden_size: int,
        output_size: int,
        n_blocks: int,
        discrete_action: bool,
    ):
        super().__init__()
        layers = [
            nn.Linear(observation_shape[0] + action_size, output_size),
            *[
                SimBaBlock(output_size, hidden_size, output_size)
                for _ in range(n_blocks)
            ],
            nn.LayerNorm(output_size),
        ]
        self._layers = nn.Sequential(*layers)
        self._action_size = action_size
        self._discrete_action = discrete_action

    def forward(
        self, x: TorchObservation, action: torch.Tensor
    ) -> torch.Tensor:
        assert isinstance(x, torch.Tensor)
        if self._discrete_action:
            action = F.one_hot(
                action.view(-1).long(), num_classes=self._action_size
            ).float()
        h = torch.cat([x, action], dim=1)
        return self._layers(h)


class SimbaV2Encoder(Encoder):
    def __init__(
        self,
        observation_shape: Sequence[int],
        hidden_size: int,
        n_blocks: int,
        scaler_init: float,
        scaler_scale: float,
        alpha_init: float,
        alpha_scale: float,
        c_shift: float,
    ):
        super().__init__()
        assert len(observation_shape) == 1
        layers = [
            HyperInputEncoder(
                in_features=observation_shape[0],
                out_features=hidden_size,
                scaler_init=scaler_init,
                scaler_scale=scaler_scale,
                c_shift=c_shift,
            ),
            *[
                HyperLERPBlock(
                    hidden_dim=hidden_size,
                    scaler_init=scaler_init,
                    scaler_scale=scaler_scale,
                    alpha_init=alpha_init,
                    alpha_scale=alpha_scale,
                )
                for _ in range(n_blocks)
            ],
        ]
        self._layers = nn.Sequential(*layers)

    def forward(self, x: TorchObservation) -> torch.Tensor:
        assert isinstance(x, torch.Tensor)
        return self._layers(x)


class SimbaV2EncoderWithAction(EncoderWithAction):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        hidden_size: int,
        n_blocks: int,
        scaler_init: float,
        scaler_scale: float,
        alpha_init: float,
        alpha_scale: float,
        c_shift: float,
        discrete_action: bool,
    ):
        super().__init__()
        assert len(observation_shape) == 1
        layers = [
            HyperInputEncoder(
                in_features=observation_shape[0] + action_size,
                out_features=hidden_size,
                scaler_init=scaler_init,
                scaler_scale=scaler_scale,
                c_shift=c_shift,
            ),
            *[
                HyperLERPBlock(
                    hidden_dim=hidden_size,
                    scaler_init=scaler_init,
                    scaler_scale=scaler_scale,
                    alpha_init=alpha_init,
                    alpha_scale=alpha_scale,
                )
                for _ in range(n_blocks)
            ],
        ]
        self._layers = nn.Sequential(*layers)
        self._discrete_action = discrete_action

    def forward(
        self, x: TorchObservation, action: torch.Tensor
    ) -> torch.Tensor:
        assert isinstance(x, torch.Tensor)
        if self._discrete_action:
            action = F.one_hot(
                action.view(-1).long(), num_classes=self._action_size
            ).float()
        h = torch.cat([x, action], dim=1)
        return self._layers(h)


def compute_output_size(
    input_shapes: Sequence[Shape], encoder: nn.Module
) -> int:
    device = next(encoder.parameters()).device
    with torch.no_grad():
        inputs = []
        for shape in input_shapes:
            if isinstance(shape[0], (list, tuple)):
                inputs.append([torch.rand(2, *s, device=device) for s in shape])
            else:
                inputs.append(torch.rand(2, *shape, device=device))
        y = encoder(*inputs)
    return int(y.shape[1])

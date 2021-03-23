from abc import ABCMeta, abstractmethod
from typing import Callable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(metaclass=ABCMeta):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_feature_size(self) -> int:
        pass

    @property
    def observation_shape(self) -> Sequence[int]:
        pass

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        pass


class EncoderWithAction(metaclass=ABCMeta):
    @abstractmethod
    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_feature_size(self) -> int:
        pass

    @property
    def action_size(self) -> int:
        pass

    @property
    def observation_shape(self) -> Sequence[int]:
        pass

    @abstractmethod
    def __call__(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        pass


class _PixelEncoder(nn.Module):  # type: ignore

    _observation_shape: Sequence[int]
    _feature_size: int
    _use_batch_norm: bool
    _activation: Callable[[torch.Tensor], torch.Tensor]
    _convs: nn.ModuleList
    _conv_bns: nn.ModuleList
    _fc: nn.Linear
    _fc_bn: nn.BatchNorm1d

    def __init__(
        self,
        observation_shape: Sequence[int],
        filters: Optional[List[Sequence[int]]] = None,
        feature_size: int = 512,
        use_batch_norm: bool = False,
        activation: Callable[[torch.Tensor], torch.Tensor] = torch.relu,
    ):
        super().__init__()

        # default architecture is based on Nature DQN paper.
        if filters is None:
            filters = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
        if feature_size is None:
            feature_size = 512

        self._observation_shape = observation_shape
        self._use_batch_norm = use_batch_norm
        self._activation = activation  # type: ignore
        self._feature_size = feature_size

        # convolutional layers
        in_channels = [observation_shape[0]] + [f[0] for f in filters[:-1]]
        self._convs = nn.ModuleList()
        self._conv_bns = nn.ModuleList()
        for in_channel, f in zip(in_channels, filters):
            out_channel, kernel_size, stride = f
            conv = nn.Conv2d(
                in_channel, out_channel, kernel_size=kernel_size, stride=stride
            )
            self._convs.append(conv)

            if use_batch_norm:
                self._conv_bns.append(nn.BatchNorm2d(out_channel))

        # last dense layer
        self._fc = nn.Linear(self._get_linear_input_size(), feature_size)
        if use_batch_norm:
            self._fc_bn = nn.BatchNorm1d(feature_size)

    def _get_linear_input_size(self) -> int:
        x = torch.rand((1,) + tuple(self._observation_shape))
        with torch.no_grad():
            return self._conv_encode(x).view(1, -1).shape[1]  # type: ignore

    def _conv_encode(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i in range(len(self._convs)):
            h = self._activation(self._convs[i](h))  # type: ignore
            if self._use_batch_norm:
                h = self._conv_bns[i](h)
        return h

    def get_feature_size(self) -> int:
        return self._feature_size

    @property
    def observation_shape(self) -> Sequence[int]:
        return self._observation_shape


class PixelEncoder(_PixelEncoder, Encoder):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._conv_encode(x)

        h = self._activation(self._fc(h.view(h.shape[0], -1)))  # type: ignore
        if self._use_batch_norm:
            h = self._fc_bn(h)

        return h


class PixelEncoderWithAction(_PixelEncoder, EncoderWithAction):

    _action_size: int
    _discrete_action: bool

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        filters: Optional[List[Sequence[int]]] = None,
        feature_size: int = 512,
        use_batch_norm: bool = False,
        discrete_action: bool = False,
        activation: Callable[[torch.Tensor], torch.Tensor] = torch.relu,
    ):
        self._action_size = action_size
        self._discrete_action = discrete_action
        super().__init__(
            observation_shape, filters, feature_size, use_batch_norm, activation
        )

    def _get_linear_input_size(self) -> int:
        size = super()._get_linear_input_size()
        return size + self._action_size

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        h = self._conv_encode(x)

        if self._discrete_action:
            action = F.one_hot(
                action.view(-1).long(), num_classes=self._action_size
            ).float()

        # cocat feature and action
        h = torch.cat([h.view(h.shape[0], -1), action], dim=1)
        h = self._activation(self._fc(h))  # type: ignore
        if self._use_batch_norm:
            h = self._fc_bn(h)

        return h

    @property
    def action_size(self) -> int:
        return self._action_size


class _VectorEncoder(nn.Module):  # type: ignore

    _observation_shape: Sequence[int]
    _use_batch_norm: bool
    _use_dense: bool
    _activation: Callable[[torch.Tensor], torch.Tensor]
    _feature_size: int
    _fcs: nn.ModuleList
    _bns: nn.ModuleList

    def __init__(
        self,
        observation_shape: Sequence[int],
        hidden_units: Optional[Sequence[int]] = None,
        use_batch_norm: bool = False,
        use_dense: bool = False,
        activation: Callable[[torch.Tensor], torch.Tensor] = torch.relu,
    ):
        super().__init__()
        self._observation_shape = observation_shape

        if hidden_units is None:
            hidden_units = [256, 256]

        self._use_batch_norm = use_batch_norm
        self._feature_size = hidden_units[-1]
        self._activation = activation  # type: ignore
        self._use_dense = use_dense

        in_units = [observation_shape[0]] + list(hidden_units[:-1])
        self._fcs = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i, (in_unit, out_unit) in enumerate(zip(in_units, hidden_units)):
            if use_dense and i > 0:
                in_unit += observation_shape[0]
            self._fcs.append(nn.Linear(in_unit, out_unit))
            if use_batch_norm:
                self._bns.append(nn.BatchNorm1d(out_unit))

    def _fc_encode(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i in range(len(self._fcs)):
            if self._use_dense and i > 0:
                h = torch.cat([h, x], dim=1)
            h = self._activation(self._fcs[i](h))  # type: ignore
            if self._use_batch_norm:
                h = self._bns[i](h)
        return h

    def get_feature_size(self) -> int:
        return self._feature_size

    @property
    def observation_shape(self) -> Sequence[int]:
        return self._observation_shape


class VectorEncoder(_VectorEncoder, Encoder):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._fc_encode(x)


class VectorEncoderWithAction(_VectorEncoder, EncoderWithAction):

    _action_size: int
    _discrete_action: bool

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        hidden_units: Optional[Sequence[int]] = None,
        use_batch_norm: bool = False,
        use_dense: bool = False,
        discrete_action: bool = False,
        activation: Callable[[torch.Tensor], torch.Tensor] = torch.relu,
    ):
        self._action_size = action_size
        self._discrete_action = discrete_action
        concat_shape = (observation_shape[0] + action_size,)
        super().__init__(
            concat_shape, hidden_units, use_batch_norm, use_dense, activation
        )
        self._observation_shape = observation_shape

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if self._discrete_action:
            action = F.one_hot(
                action.view(-1).long(), num_classes=self.action_size
            ).float()

        x = torch.cat([x, action], dim=1)
        return self._fc_encode(x)

    @property
    def action_size(self) -> int:
        return self._action_size

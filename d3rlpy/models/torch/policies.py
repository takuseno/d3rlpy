from abc import ABCMeta, abstractmethod
from typing import Any, NamedTuple, Optional, Union, cast

import torch
from torch import nn
from torch.distributions import Categorical

from .distributions import GaussianDistribution, SquashedGaussianDistribution
from .encoders import Encoder, EncoderWithAction

__all__ = [
    "Policy",
    "DeterministicPolicy",
    "DeterministicResidualPolicy",
    "NormalPolicy",
    "CategoricalPolicy",
    "build_gaussian_distribution",
    "build_squashed_gaussian_distribution",
    "build_categorical_distribution",
    "ActionOutput",
]


class ActionOutput(NamedTuple):
    mu: torch.Tensor
    squashed_mu: torch.Tensor
    logstd: Optional[torch.Tensor]


def build_gaussian_distribution(action: ActionOutput) -> GaussianDistribution:
    assert action.logstd is not None
    return GaussianDistribution(
        loc=action.squashed_mu, std=action.logstd.exp(), raw_loc=action.mu
    )


def build_squashed_gaussian_distribution(
    action: ActionOutput,
) -> SquashedGaussianDistribution:
    assert action.logstd is not None
    return SquashedGaussianDistribution(loc=action.mu, std=action.logstd.exp())


class Policy(nn.Module, metaclass=ABCMeta):  # type: ignore
    @abstractmethod
    def forward(self, x: torch.Tensor, *args: Any) -> ActionOutput:
        pass

    def __call__(self, x: torch.Tensor, *args: Any) -> ActionOutput:
        return super().__call__(x, *args)  # type: ignore


class DeterministicPolicy(Policy):
    _encoder: Encoder
    _fc: nn.Linear

    def __init__(self, encoder: Encoder, hidden_size: int, action_size: int):
        super().__init__()
        self._encoder = encoder
        self._fc = nn.Linear(hidden_size, action_size)

    def forward(self, x: torch.Tensor, *args: Any) -> ActionOutput:
        h = self._encoder(x)
        mu = self._fc(h)
        return ActionOutput(mu, torch.tanh(mu), logstd=None)


class DeterministicResidualPolicy(Policy):
    _encoder: EncoderWithAction
    _scale: float
    _fc: nn.Linear

    def __init__(
        self,
        encoder: EncoderWithAction,
        hidden_size: int,
        action_size: int,
        scale: float,
    ):
        super().__init__()
        self._scale = scale
        self._encoder = encoder
        self._fc = nn.Linear(hidden_size, action_size)

    def forward(self, x: torch.Tensor, *args: Any) -> ActionOutput:
        action = args[0]
        h = self._encoder(x, action)
        residual_action = self._scale * torch.tanh(self._fc(h))
        action = (action + cast(torch.Tensor, residual_action)).clamp(-1.0, 1.0)
        return ActionOutput(mu=action, squashed_mu=action, logstd=None)


class NormalPolicy(Policy):
    _encoder: Encoder
    _action_size: int
    _min_logstd: float
    _max_logstd: float
    _use_std_parameter: bool
    _mu: nn.Linear
    _logstd: Union[nn.Linear, nn.Parameter]

    def __init__(
        self,
        encoder: Encoder,
        hidden_size: int,
        action_size: int,
        min_logstd: float,
        max_logstd: float,
        use_std_parameter: bool,
    ):
        super().__init__()
        self._action_size = action_size
        self._encoder = encoder
        self._min_logstd = min_logstd
        self._max_logstd = max_logstd
        self._use_std_parameter = use_std_parameter
        self._mu = nn.Linear(hidden_size, action_size)
        if use_std_parameter:
            initial_logstd = torch.zeros(1, action_size, dtype=torch.float32)
            self._logstd = nn.Parameter(initial_logstd)
        else:
            self._logstd = nn.Linear(hidden_size, action_size)

    def forward(self, x: torch.Tensor, *args: Any) -> ActionOutput:
        h = self._encoder(x)
        mu = self._mu(h)

        if self._use_std_parameter:
            logstd = torch.sigmoid(cast(nn.Parameter, self._logstd))
            base_logstd = self._max_logstd - self._min_logstd
            clipped_logstd = self._min_logstd + logstd * base_logstd
        else:
            logstd = cast(nn.Linear, self._logstd)(h)
            clipped_logstd = logstd.clamp(self._min_logstd, self._max_logstd)

        return ActionOutput(mu, torch.tanh(mu), clipped_logstd)


class CategoricalPolicy(nn.Module):  # type: ignore
    _encoder: Encoder
    _fc: nn.Linear

    def __init__(self, encoder: Encoder, hidden_size: int, action_size: int):
        super().__init__()
        self._encoder = encoder
        self._fc = nn.Linear(hidden_size, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._fc(self._encoder(x))


def build_categorical_distribution(logits: torch.Tensor) -> Categorical:
    return Categorical(probs=torch.softmax(logits, dim=1))

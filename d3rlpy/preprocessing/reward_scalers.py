import dataclasses
from typing import Any, Optional, Sequence

import gym
import numpy as np
import torch

from ..dataset import (
    EpisodeBase,
    TrajectorySlicerProtocol,
    TransitionPickerProtocol,
)
from ..serializable_config import generate_optional_config_generation
from .base import Scaler

__all__ = [
    "RewardScaler",
    "MultiplyRewardScaler",
    "ClipRewardScaler",
    "MinMaxRewardScaler",
    "StandardRewardScaler",
    "ReturnBasedRewardScaler",
    "ConstantShiftRewardScaler",
    "register_reward_scaler",
    "make_reward_scaler_field",
]


class RewardScaler(Scaler):
    def fit_with_env(self, env: gym.Env[Any, Any]) -> None:
        pass


@dataclasses.dataclass()
class MultiplyRewardScaler(RewardScaler):
    r"""Multiplication reward preprocessing.

    This preprocessor multiplies rewards by a constant number.

    .. code-block:: python

        from d3rlpy.preprocessing import MultiplyRewardScaler
        from d3rlpy.algos import CQLConfig

        # multiply rewards by 10
        reward_scaler = MultiplyRewardScaler(10.0)
        cql = CQLConfig(reward_scaler=reward_scaler).create()

    Args:
        multiplier (float): Constant multiplication value.
    """
    multiplier: float = 1.0

    def fit_with_transition_picker(
        self,
        episodes: Sequence[EpisodeBase],
        transition_picker: TransitionPickerProtocol,
    ) -> None:
        pass

    def fit_with_trajectory_slicer(
        self,
        episodes: Sequence[EpisodeBase],
        trajectory_slicer: TrajectorySlicerProtocol,
    ) -> None:
        pass

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.multiplier * x

    def reverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.multiplier

    def transform_numpy(self, x: np.ndarray) -> np.ndarray:
        return self.multiplier * x

    def reverse_transform_numpy(self, x: np.ndarray) -> np.ndarray:
        return x / self.multiplier

    @staticmethod
    def get_type() -> str:
        return "multiply"

    @property
    def built(self) -> bool:
        return True


@dataclasses.dataclass()
class ClipRewardScaler(RewardScaler):
    r"""Reward clipping preprocessing.

    .. code-block:: python

        from d3rlpy.preprocessing import ClipRewardScaler
        from d3rlpy.algos import CQLConfig

        # clip rewards within [-1.0, 1.0]
        reward_scaler = ClipRewardScaler(low=-1.0, high=1.0)
        cql = CQLConfig(reward_scaler=reward_scaler).create()

    Args:
        low (Optional[float]): Minimum value to clip.
        high (Optional[float]): Maximum value to clip.
        multiplier (float): Constant multiplication value.
    """
    low: Optional[float] = None
    high: Optional[float] = None
    multiplier: float = 1.0

    def fit_with_transition_picker(
        self,
        episodes: Sequence[EpisodeBase],
        transition_picker: TransitionPickerProtocol,
    ) -> None:
        pass

    def fit_with_trajectory_slicer(
        self,
        episodes: Sequence[EpisodeBase],
        trajectory_slicer: TrajectorySlicerProtocol,
    ) -> None:
        pass

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.multiplier * x.clamp(self.low, self.high)

    def reverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.multiplier

    def transform_numpy(self, x: np.ndarray) -> np.ndarray:
        return self.multiplier * np.clip(x, self.low, self.high)

    def reverse_transform_numpy(self, x: np.ndarray) -> np.ndarray:
        return x / self.multiplier

    @staticmethod
    def get_type() -> str:
        return "clip"

    @property
    def built(self) -> bool:
        return True


@dataclasses.dataclass()
class MinMaxRewardScaler(RewardScaler):
    r"""Min-Max reward normalization preprocessing.

    Rewards will be normalized in range ``[0.0, 1.0]``.

    .. math::

        r' = (r - \min(r)) / (\max(r) - \min(r))

    .. code-block:: python

        from d3rlpy.preprocessing import MinMaxRewardScaler
        from d3rlpy.algos import CQLConfig

        # normalize based on datasets
        cql = CQLConfig(reward_scaler=MinMaxRewardScaler()).create()

        # initialize manually
        reward_scaler = MinMaxRewardScaler(minimum=0.0, maximum=10.0)
        cql = CQLConfig(reward_scaler=reward_scaler).create()

    Args:
        minimum (float): Minimum value.
        maximum (float): Maximum value.
        multiplier (float): Constant multiplication value.
    """
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    multiplier: float = 1.0

    def fit_with_transition_picker(
        self,
        episodes: Sequence[EpisodeBase],
        transition_picker: TransitionPickerProtocol,
    ) -> None:
        assert not self.built
        rewards = []
        for episode in episodes:
            for i in range(episode.transition_count):
                transition = transition_picker(episode, i)
                rewards.append(transition.reward)
        self.minimum = float(np.min(rewards))
        self.maximum = float(np.max(rewards))

    def fit_with_trajectory_slicer(
        self,
        episodes: Sequence[EpisodeBase],
        trajectory_slicer: TrajectorySlicerProtocol,
    ) -> None:
        assert not self.built
        rewards = [
            trajectory_slicer(
                episode, episode.size() - 1, episode.size()
            ).rewards
            for episode in episodes
        ]
        self.minimum = float(np.min(rewards))
        self.maximum = float(np.max(rewards))

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        assert self.built
        assert self.maximum is not None and self.minimum is not None
        base = self.maximum - self.minimum
        return self.multiplier * (x - self.minimum) / base

    def reverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        assert self.built
        assert self.maximum is not None and self.minimum is not None
        base = self.maximum - self.minimum
        return x * base / self.multiplier + self.minimum

    def transform_numpy(self, x: np.ndarray) -> np.ndarray:
        assert self.built
        assert self.maximum is not None and self.minimum is not None
        base = self.maximum - self.minimum
        return self.multiplier * (x - self.minimum) / base

    def reverse_transform_numpy(self, x: np.ndarray) -> np.ndarray:
        assert self.built
        assert self.maximum is not None and self.minimum is not None
        base = self.maximum - self.minimum
        return x * base / self.multiplier + self.minimum

    @staticmethod
    def get_type() -> str:
        return "min_max"

    @property
    def built(self) -> bool:
        return self.minimum is not None and self.maximum is not None


@dataclasses.dataclass()
class StandardRewardScaler(RewardScaler):
    r"""Reward standardization preprocessing.

    .. math::

        r' = (r - \mu) / \sigma

    .. code-block:: python

        from d3rlpy.preprocessing import StandardRewardScaler
        from d3rlpy.algos import CQLConfig

        # normalize based on datasets
        cql = CQLConfig(reward_scaler=StandardRewardScaler()).create()

        # initialize manually
        reward_scaler = StandardRewardScaler(mean=0.0, std=1.0)
        cql = CQLConfig(reward_scaler=reward_scaler).create()

    Args:
        mean (float): Mean value.
        std (float): Standard deviation value.
        eps (float): Constant value to avoid zero-division.
        multiplier (float): Constant multiplication value
    """
    mean: Optional[float] = None
    std: Optional[float] = None
    eps: float = 1e-3
    multiplier: float = 1.0

    def fit_with_transition_picker(
        self,
        episodes: Sequence[EpisodeBase],
        transition_picker: TransitionPickerProtocol,
    ) -> None:
        assert not self.built
        rewards = []
        for episode in episodes:
            for i in range(episode.transition_count):
                transition = transition_picker(episode, i)
                rewards.append(transition.reward)
        self.mean = float(np.mean(rewards))
        self.std = float(np.std(rewards))

    def fit_with_trajectory_slicer(
        self,
        episodes: Sequence[EpisodeBase],
        trajectory_slicer: TrajectorySlicerProtocol,
    ) -> None:
        assert not self.built
        rewards = [
            trajectory_slicer(
                episode, episode.size() - 1, episode.size()
            ).rewards
            for episode in episodes
        ]
        self.mean = float(np.mean(rewards))
        self.std = float(np.std(rewards))

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        assert self.built
        assert self.mean is not None and self.std is not None
        nonzero_std = self.std + self.eps
        return self.multiplier * (x - self.mean) / nonzero_std

    def reverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        assert self.built
        assert self.mean is not None and self.std is not None
        return x * (self.std + self.eps) / self.multiplier + self.mean

    def transform_numpy(self, x: np.ndarray) -> np.ndarray:
        assert self.built
        assert self.mean is not None and self.std is not None
        nonzero_std = self.std + self.eps
        return self.multiplier * (x - self.mean) / nonzero_std

    def reverse_transform_numpy(self, x: np.ndarray) -> np.ndarray:
        assert self.built
        assert self.mean is not None and self.std is not None
        return x * (self.std + self.eps) / self.multiplier + self.mean

    @staticmethod
    def get_type() -> str:
        return "standard"

    @property
    def built(self) -> bool:
        return self.mean is not None and self.std is not None


@dataclasses.dataclass()
class ReturnBasedRewardScaler(RewardScaler):
    r"""Reward normalization preprocessing based on return scale.

    .. math::

        r' = r / (R_{max} - R_{min})

    .. code-block:: python

        from d3rlpy.preprocessing import ReturnBasedRewardScaler
        from d3rlpy.algos import CQLConfig

        # normalize based on datasets
        cql = CQLConfig(reward_scaler=ReturnBasedRewardScaler()).create()

        # initialize manually
        reward_scaler = ReturnBasedRewardScaler(
            return_max=100.0,
            return_min=1.0,
        )
        cql = CQLConfig(reward_scaler=reward_scaler).create()

    References:
        * `Kostrikov et al., Offline Reinforcement Learning with Implicit
          Q-Learning. <https://arxiv.org/abs/2110.06169>`_

    Args:
        return_max (float): Maximum return value.
        return_min (float): Standard deviation value.
        multiplier (float): Constant multiplication value
    """
    return_max: Optional[float] = None
    return_min: Optional[float] = None
    multiplier: float = 1.0

    def fit_with_transition_picker(
        self,
        episodes: Sequence[EpisodeBase],
        transition_picker: TransitionPickerProtocol,
    ) -> None:
        assert not self.built
        returns = []
        for episode in episodes:
            rewards = []
            for i in range(episode.transition_count):
                transition = transition_picker(episode, i)
                rewards.append(transition.reward)
            returns.append(float(np.sum(rewards)))
        self.return_max = float(np.max(returns))
        self.return_min = float(np.min(returns))

    def fit_with_trajectory_slicer(
        self,
        episodes: Sequence[EpisodeBase],
        trajectory_slicer: TrajectorySlicerProtocol,
    ) -> None:
        assert not self.built
        returns = []
        for episode in episodes:
            traj = trajectory_slicer(
                episode, episode.size() - 1, episode.size()
            )
            returns.append(float(np.sum(traj.rewards)))
        self.return_max = float(np.max(returns))
        self.return_min = float(np.min(returns))

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        assert self.built
        assert self.return_min is not None and self.return_max is not None
        return self.multiplier * x / (self.return_max - self.return_min)

    def reverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        assert self.built
        assert self.return_min is not None and self.return_max is not None
        return x * (self.return_max - self.return_min) / self.multiplier

    def transform_numpy(self, x: np.ndarray) -> np.ndarray:
        assert self.built
        assert self.return_min is not None and self.return_max is not None
        return self.multiplier * x / (self.return_max - self.return_min)

    def reverse_transform_numpy(self, x: np.ndarray) -> np.ndarray:
        assert self.built
        assert self.return_min is not None and self.return_max is not None
        return x * (self.return_max - self.return_min) / self.multiplier

    @staticmethod
    def get_type() -> str:
        return "return"

    @property
    def built(self) -> bool:
        return self.return_max is not None and self.return_min is not None


@dataclasses.dataclass()
class ConstantShiftRewardScaler(RewardScaler):
    r"""Reward shift preprocessing.

    .. math::

        r' = r + c

    You need to initialize manually.

    .. code-block:: python

        from d3rlpy.preprocessing import ConstantShiftRewardScaler
        from d3rlpy.algos import CQLConfig

        reward_scaler = ConstantShiftRewardScaler(shift=-1.0)
        cql = CQLConfig(reward_scaler=reward_scaler).create()

    References:
        * `Kostrikov et al., Offline Reinforcement Learning with Implicit
          Q-Learning. <https://arxiv.org/abs/2110.06169>`_

    Args:
        shift (float): Constant shift value
    """
    shift: float

    def fit_with_transition_picker(
        self,
        episodes: Sequence[EpisodeBase],
        transition_picker: TransitionPickerProtocol,
    ) -> None:
        pass

    def fit_with_trajectory_slicer(
        self,
        episodes: Sequence[EpisodeBase],
        trajectory_slicer: TrajectorySlicerProtocol,
    ) -> None:
        pass

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.shift + x

    def reverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.shift

    def transform_numpy(self, x: np.ndarray) -> np.ndarray:
        return self.shift + x

    def reverse_transform_numpy(self, x: np.ndarray) -> np.ndarray:
        return x - self.shift

    @staticmethod
    def get_type() -> str:
        return "shift"

    @property
    def built(self) -> bool:
        return True


(
    register_reward_scaler,
    make_reward_scaler_field,
) = generate_optional_config_generation(
    RewardScaler  # type: ignore
)


register_reward_scaler(MultiplyRewardScaler)
register_reward_scaler(ClipRewardScaler)
register_reward_scaler(MinMaxRewardScaler)
register_reward_scaler(StandardRewardScaler)
register_reward_scaler(ReturnBasedRewardScaler)
register_reward_scaler(ConstantShiftRewardScaler)

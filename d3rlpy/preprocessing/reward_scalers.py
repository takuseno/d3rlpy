import dataclasses
from typing import Optional, Sequence

import gym
import numpy as np
import torch

from ..dataset import EpisodeBase
from ..serializable_config import (
    DynamicConfig,
    generate_optional_config_generation,
)

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


class RewardScaler(DynamicConfig):
    def fit(self, episodes: Sequence[EpisodeBase]) -> None:
        """Estimates scaling parameters from dataset.

        Args:
            episodes: list of episodes.

        """
        raise NotImplementedError

    def fit_with_env(self, env: gym.Env) -> None:
        """Gets scaling parameters from environment.

        Note:
            ``RewardScaler`` does not support fitting with environment.

        Args:
            env: gym environment.

        """
        raise NotImplementedError("Please initialize with dataset.")

    def transform(self, reward: torch.Tensor) -> torch.Tensor:
        """Returns processed rewards.

        Args:
            reward: reward.

        Returns:
            processed reward.

        """
        raise NotImplementedError

    def reverse_transform(self, reward: torch.Tensor) -> torch.Tensor:
        """Returns reversely processed rewards.

        Args:
            reward: reward.

        Returns:
            reversely processed reward.

        """
        raise NotImplementedError

    def transform_numpy(self, reward: np.ndarray) -> np.ndarray:
        """Returns transformed rewards in numpy array.

        Args:
            reward: reward.

        Returns:
            transformed reward.

        """
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class MultiplyRewardScaler(RewardScaler):
    r"""Multiplication reward preprocessing.

    This preprocessor multiplies rewards by a constant number.

    .. code-block:: python

        from d3rlpy.preprocessing import MultiplyRewardScaler

        # multiply rewards by 10
        reward_scaler = MultiplyRewardScaler(10.0)

        cql = CQL(reward_scaler=reward_scaler)

    Args:
        multiplier (float): constant multiplication value.

    """
    multiplier: float = 1.0

    def fit(self, episodes: Sequence[EpisodeBase]) -> None:
        pass

    def transform(self, reward: torch.Tensor) -> torch.Tensor:
        return self.multiplier * reward

    def reverse_transform(self, reward: torch.Tensor) -> torch.Tensor:
        return reward / self.multiplier

    def transform_numpy(self, reward: np.ndarray) -> np.ndarray:
        return self.multiplier * reward

    @staticmethod
    def get_type() -> str:
        return "multiply"


@dataclasses.dataclass(frozen=True)
class ClipRewardScaler(RewardScaler):
    r"""Reward clipping preprocessing.

    .. code-block:: python

        from d3rlpy.preprocessing import ClipRewardScaler

        # clip rewards within [-1.0, 1.0]
        reward_scaler = ClipRewardScaler(low=-1.0, high=1.0)

        cql = CQL(reward_scaler=reward_scaler)

    Args:
        low (float): minimum value to clip.
        high (float): maximum value to clip.
        multiplier (float): constant multiplication value.

    """
    low: Optional[float] = None
    high: Optional[float] = None
    multiplier: float = 1.0

    def fit(self, episodes: Sequence[EpisodeBase]) -> None:
        pass

    def transform(self, reward: torch.Tensor) -> torch.Tensor:
        return self.multiplier * reward.clamp(self.low, self.high)

    def reverse_transform(self, reward: torch.Tensor) -> torch.Tensor:
        return reward / self.multiplier

    def transform_numpy(self, reward: np.ndarray) -> np.ndarray:
        return self.multiplier * np.clip(reward, self.low, self.high)

    @staticmethod
    def get_type() -> str:
        return "clip"


@dataclasses.dataclass()
class MinMaxRewardScaler(RewardScaler):
    r"""Min-Max reward normalization preprocessing.

    .. math::

        r' = (r - \min(r)) / (\max(r) - \min(r))

    .. code-block:: python

        from d3rlpy.algos import CQL

        cql = CQL(reward_scaler="min_max")

    You can also initialize with :class:`d3rlpy.dataset.MDPDataset` object or
    manually.

    .. code-block:: python

        from d3rlpy.preprocessing import MinMaxRewardScaler

        # initialize manually
        scaler = MinMaxRewardScaler(minimum=0.0, maximum=10.0)

        cql = CQL(scaler=scaler)

    Args:
        minimum (float): minimum value.
        maximum (float): maximum value.
        multiplier (float): constant multiplication value.

    """
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    multiplier: float = 1.0

    def fit(self, episodes: Sequence[EpisodeBase]) -> None:
        if self.minimum is not None and self.maximum is not None:
            return

        rewards = [episode.rewards for episode in episodes]

        self.minimum = float(np.min(rewards))
        self.maximum = float(np.max(rewards))

    def transform(self, reward: torch.Tensor) -> torch.Tensor:
        assert self.minimum is not None and self.maximum is not None
        base = self.maximum - self.minimum
        return self.multiplier * (reward - self.minimum) / base

    def reverse_transform(self, reward: torch.Tensor) -> torch.Tensor:
        assert self.minimum is not None and self.maximum is not None
        base = self.maximum - self.minimum
        return reward * base / self.multiplier + self.minimum

    def transform_numpy(self, reward: np.ndarray) -> np.ndarray:
        assert self.minimum is not None and self.maximum is not None
        base = self.maximum - self.minimum
        return self.multiplier * (reward - self.minimum) / base

    @staticmethod
    def get_type() -> str:
        return "min_max"


@dataclasses.dataclass()
class StandardRewardScaler(RewardScaler):
    r"""Reward standardization preprocessing.

    .. math::

        r' = (r - \mu) / \sigma

    .. code-block:: python

        from d3rlpy.algos import CQL

        cql = CQL(reward_scaler="standard")

    You can also initialize with :class:`d3rlpy.dataset.MDPDataset` object or
    manually.

    .. code-block:: python

        from d3rlpy.preprocessing import StandardRewardScaler

        # initialize manually
        scaler = StandardRewardScaler(mean=0.0, std=1.0)

        cql = CQL(scaler=scaler)

    Args:
        mean (float): mean value.
        std (float): standard deviation value.
        eps (float): constant value to avoid zero-division.
        multiplier (float): constant multiplication value

    """
    mean: Optional[float] = None
    std: Optional[float] = None
    eps: float = 1e-3
    multiplier: float = 1.0

    def fit(self, episodes: Sequence[EpisodeBase]) -> None:
        if self.mean is not None and self.std is not None:
            return

        rewards = [episode.rewards for episode in episodes]

        self.mean = float(np.mean(rewards))
        self.std = float(np.std(rewards))

    def transform(self, reward: torch.Tensor) -> torch.Tensor:
        assert self.mean is not None and self.std is not None
        nonzero_std = self.std + self.eps
        return self.multiplier * (reward - self.mean) / nonzero_std

    def reverse_transform(self, reward: torch.Tensor) -> torch.Tensor:
        assert self.mean is not None and self.std is not None
        return reward * (self.std + self.eps) / self.multiplier + self.mean

    def transform_numpy(self, reward: np.ndarray) -> np.ndarray:
        assert self.mean is not None and self.std is not None
        nonzero_std = self.std + self.eps
        return self.multiplier * (reward - self.mean) / nonzero_std

    @staticmethod
    def get_type() -> str:
        return "standard"


class ReturnBasedRewardScaler(RewardScaler):
    r"""Reward normalization preprocessing based on return scale.

    .. math::

        r' = r / (R_{max} - R_{min})

    .. code-block:: python

        from d3rlpy.algos import CQL

        cql = CQL(reward_scaler="return")

    You can also initialize with :class:`d3rlpy.dataset.MDPDataset` object or
    manually.

    .. code-block:: python

        from d3rlpy.preprocessing import ReturnBasedRewardScaler

        # initialize manually
        scaler = ReturnBasedRewardScaler(return_max=100.0, return_min=1.0)

        cql = CQL(scaler=scaler)

    References:
        * `Kostrikov et al., Offline Reinforcement Learning with Implicit
          Q-Learning. <https://arxiv.org/abs/2110.06169>`_

    Args:
        return_max (float): the maximum return value.
        return_min (float): standard deviation value.
        multiplier (float): constant multiplication value

    """
    return_max: Optional[float] = None
    return_min: Optional[float] = None
    multiplier: float = 1.0

    def fit(self, episodes: Sequence[EpisodeBase]) -> None:
        if self.return_max is not None and self.return_min is not None:
            return

        # accumulate all rewards
        returns = []
        for episode in episodes:
            returns.append(episode.compute_return())

        self.return_max = float(np.max(returns))
        self.return_min = float(np.min(returns))

    def transform(self, reward: torch.Tensor) -> torch.Tensor:
        assert self.return_max is not None and self.return_min is not None
        return self.multiplier * reward / (self.return_max - self.return_min)

    def reverse_transform(self, reward: torch.Tensor) -> torch.Tensor:
        assert self.return_max is not None and self.return_min is not None
        return reward * (self.return_max + self.return_min) / self.multiplier

    def transform_numpy(self, reward: np.ndarray) -> np.ndarray:
        assert self.return_max is not None and self.return_min is not None
        return self.multiplier * reward / (self.return_max - self.return_min)

    @staticmethod
    def get_type() -> str:
        return "return"


@dataclasses.dataclass(frozen=True)
class ConstantShiftRewardScaler(RewardScaler):
    r"""Reward shift preprocessing.

    .. math::

        r' = r + c

    You need to initialize manually.

    .. code-block:: python

        from d3rlpy.preprocessing import ConstantShiftRewardScaler

        # initialize manually
        scaler = ConstantShiftRewardScaler(shift=-1.0)

        cql = CQL(reward_scaler=scaler)

    References:
        * `Kostrikov et al., Offline Reinforcement Learning with Implicit
          Q-Learning. <https://arxiv.org/abs/2110.06169>`_

    Args:
        shift (float): constant shift value

    """
    shift: float

    def fit(self, episodes: Sequence[EpisodeBase]) -> None:
        pass

    def transform(self, reward: torch.Tensor) -> torch.Tensor:
        return self.shift + reward

    def reverse_transform(self, reward: torch.Tensor) -> torch.Tensor:
        return reward - self.shift

    def transform_numpy(self, reward: np.ndarray) -> np.ndarray:
        return self.shift + reward

    @staticmethod
    def get_type() -> str:
        return "shift"


(
    register_reward_scaler,
    make_reward_scaler_field,
) = generate_optional_config_generation(RewardScaler)


register_reward_scaler(MultiplyRewardScaler)
register_reward_scaler(ClipRewardScaler)
register_reward_scaler(MinMaxRewardScaler)
register_reward_scaler(StandardRewardScaler)
register_reward_scaler(ReturnBasedRewardScaler)
register_reward_scaler(ConstantShiftRewardScaler)

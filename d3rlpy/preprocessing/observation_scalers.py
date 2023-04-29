import dataclasses
from typing import Any, Optional, Sequence

import gym
import numpy as np
import torch

from ..dataset import EpisodeBase
from ..serializable_config import (
    DynamicConfig,
    generate_optional_config_generation,
    make_optional_numpy_field,
)

__all__ = [
    "ObservationScaler",
    "PixelObservationScaler",
    "MinMaxObservationScaler",
    "StandardObservationScaler",
    "register_observation_scaler",
    "make_observation_scaler_field",
]


class ObservationScaler(DynamicConfig):
    def fit(self, episodes: Sequence[EpisodeBase]) -> None:
        """Estimates scaling parameters from dataset.

        Args:
            episodes: list of episodes.

        """
        raise NotImplementedError

    def fit_with_env(self, env: gym.Env[Any, Any]) -> None:
        """Gets scaling parameters from environment.

        Args:
            env: gym environment.

        """
        raise NotImplementedError

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Returns processed observations.

        Args:
            x: observation.

        Returns:
            processed observation.

        """
        raise NotImplementedError

    def reverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Returns reversely transformed observations.

        Args:
            x: observation.

        Returns:
            reversely transformed observation.

        """
        raise NotImplementedError


class PixelObservationScaler(ObservationScaler):
    """Pixel normalization preprocessing.

    .. math::

        x' = x / 255

    .. code-block:: python

        from d3rlpy.dataset import MDPDataset
        from d3rlpy.algos import CQL

        dataset = MDPDataset(observations, actions, rewards, terminals)

        # initialize algorithm with PixelScaler
        cql = CQL(scaler='pixel')

        cql.fit(dataset.episodes)

    """

    def fit(self, episodes: Sequence[EpisodeBase]) -> None:
        pass

    def fit_with_env(self, env: gym.Env[Any, Any]) -> None:
        pass

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return x.float() / 255.0

    def reverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x * 255.0).long()

    @staticmethod
    def get_type() -> str:
        return "pixel"


@dataclasses.dataclass()
class MinMaxObservationScaler(ObservationScaler):
    r"""Min-Max normalization preprocessing.

    .. math::

        x' = (x - \min{x}) / (\max{x} - \min{x})

    .. code-block:: python

        from d3rlpy.dataset import MDPDataset
        from d3rlpy.algos import CQL

        dataset = MDPDataset(observations, actions, rewards, terminals)

        # initialize algorithm with MinMaxScaler
        cql = CQL(scaler='min_max')

        # scaler is initialized from the given transitions
        transitions = []
        for episode in dataset.episodes:
            transitions += episode.transitions
        cql.fit(transitions)

    You can also initialize with :class:`d3rlpy.dataset.MDPDataset` object or
    manually.

    .. code-block:: python

        from d3rlpy.preprocessing import MinMaxScaler

        # initialize manually
        minimum = observations.min(axis=0)
        maximum = observations.max(axis=0)
        scaler = MinMaxScaler(minimum=minimum, maximum=maximum)

        cql = CQL(scaler=scaler)

    Args:
        minimum (numpy.ndarray): minimum values at each entry.
        maximum (numpy.ndarray): maximum values at each entry.

    """
    minimum: Optional[np.ndarray] = make_optional_numpy_field()
    maximum: Optional[np.ndarray] = make_optional_numpy_field()

    def __post_init__(self) -> None:
        if self.minimum is not None:
            self.minimum = np.asarray(self.minimum)
        if self.maximum is not None:
            self.maximum = np.asarray(self.maximum)

    def fit(self, episodes: Sequence[EpisodeBase]) -> None:
        if self.minimum is not None and self.maximum is not None:
            return

        maximum = np.zeros(episodes[0].observation_shape)
        minimum = np.zeros(episodes[0].observation_shape)
        for i, episode in enumerate(episodes):
            observations = np.asarray(episode.observations)
            max_observation = np.max(observations, axis=0)
            min_observation = np.min(observations, axis=0)
            if i == 0:
                minimum = min_observation
                maximum = max_observation
            else:
                minimum = np.minimum(minimum, min_observation)
                maximum = np.maximum(maximum, max_observation)

        self.minimum = minimum.reshape((1,) + minimum.shape)
        self.maximum = maximum.reshape((1,) + maximum.shape)

    def fit_with_env(self, env: gym.Env[Any, Any]) -> None:
        if self.minimum is not None and self.maximum is not None:
            return

        assert isinstance(env.observation_space, gym.spaces.Box)
        shape = env.observation_space.shape
        low = np.asarray(env.observation_space.low)
        high = np.asarray(env.observation_space.high)
        self.minimum = low.reshape((1,) + shape)
        self.maximum = high.reshape((1,) + shape)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        assert self.minimum is not None and self.maximum is not None
        minimum = torch.tensor(
            self.minimum, dtype=torch.float32, device=x.device
        )
        maximum = torch.tensor(
            self.maximum, dtype=torch.float32, device=x.device
        )
        return (x - minimum) / (maximum - minimum)

    def reverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        assert self.minimum is not None and self.maximum is not None
        minimum = torch.tensor(
            self.minimum, dtype=torch.float32, device=x.device
        )
        maximum = torch.tensor(
            self.maximum, dtype=torch.float32, device=x.device
        )
        return ((maximum - minimum) * x) + minimum

    @staticmethod
    def get_type() -> str:
        return "min_max"


@dataclasses.dataclass()
class StandardObservationScaler(ObservationScaler):
    r"""Standardization preprocessing.

    .. math::

        x' = (x - \mu) / \sigma

    .. code-block:: python

        from d3rlpy.dataset import MDPDataset
        from d3rlpy.algos import CQL

        dataset = MDPDataset(observations, actions, rewards, terminals)

        # initialize algorithm with StandardScaler
        cql = CQL(scaler='standard')

        # scaler is initialized from the given episodes
        transitions = []
        for episode in dataset.episodes:
            transitions += episode.transitions
        cql.fit(transitions)

    You can initialize with :class:`d3rlpy.dataset.MDPDataset` object or
    manually.

    .. code-block:: python

        from d3rlpy.preprocessing import StandardScaler

        # initialize manually
        mean = observations.mean(axis=0)
        std = observations.std(axis=0)
        scaler = StandardScaler(mean=mean, std=std)

        cql = CQL(scaler=scaler)

    Args:
        mean (numpy.ndarray): mean values at each entry.
        std (numpy.ndarray): standard deviation at each entry.
        eps (float): small constant value to avoid zero-division.

    """
    mean: Optional[np.ndarray] = make_optional_numpy_field()
    std: Optional[np.ndarray] = make_optional_numpy_field()
    eps: float = 1e-3

    def __post_init__(self) -> None:
        if self.mean is not None:
            self.mean = np.asarray(self.mean)
        if self.std is not None:
            self.std = np.asarray(self.std)

    def fit(self, episodes: Sequence[EpisodeBase]) -> None:
        if self.mean is not None and self.std is not None:
            return

        # compute mean
        total_sum = np.zeros(episodes[0].observation_shape)
        total_count = 0
        for episode in episodes:
            total_sum += np.sum(episode.observations, axis=0)
            total_count += episode.size()
        mean = total_sum / total_count

        # compute stdandard deviation
        total_sqsum = np.zeros(episodes[0].observation_shape)
        expanded_mean = mean.reshape((1,) + mean.shape)
        for episode in episodes:
            observations = np.asarray(episode.observations)
            total_sqsum += np.sum((observations - expanded_mean) ** 2, axis=0)
        std = np.sqrt(total_sqsum / total_count)

        self.mean = mean.reshape((1,) + mean.shape)
        self.std = std.reshape((1,) + std.shape)

    def fit_with_env(self, env: gym.Env[Any, Any]) -> None:
        if self.mean is not None and self.std is not None:
            return
        raise NotImplementedError(
            "standard scaler does not support fit_with_env."
        )

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        assert self.mean is not None and self.std is not None
        mean = torch.tensor(self.mean, dtype=torch.float32, device=x.device)
        std = torch.tensor(self.std, dtype=torch.float32, device=x.device)
        return (x - mean) / (std + self.eps)

    def reverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        assert self.mean is not None and self.std is not None
        mean = torch.tensor(self.mean, dtype=torch.float32, device=x.device)
        std = torch.tensor(self.std, dtype=torch.float32, device=x.device)
        return ((std + self.eps) * x) + mean

    @staticmethod
    def get_type() -> str:
        return "standard"


(
    register_observation_scaler,
    make_observation_scaler_field,
) = generate_optional_config_generation(ObservationScaler)

register_observation_scaler(PixelObservationScaler)
register_observation_scaler(MinMaxObservationScaler)
register_observation_scaler(StandardObservationScaler)

import dataclasses
from typing import Optional, Sequence

import numpy as np
import torch
from gym.spaces import Box
from gymnasium.spaces import Box as GymnasiumBox

from ..dataset import (
    EpisodeBase,
    TrajectorySlicerProtocol,
    TransitionPickerProtocol,
)
from ..envs import GymEnv
from ..serializable_config import (
    generate_optional_config_generation,
    make_optional_numpy_field,
)
from .base import Scaler, add_leading_dims, add_leading_dims_numpy

__all__ = [
    "ObservationScaler",
    "PixelObservationScaler",
    "MinMaxObservationScaler",
    "StandardObservationScaler",
    "register_observation_scaler",
    "make_observation_scaler_field",
]


class ObservationScaler(Scaler):
    pass


class PixelObservationScaler(ObservationScaler):
    """Pixel normalization preprocessing.

    .. math::

        x' = x / 255

    .. code-block:: python

        from d3rlpy.preprocessing import PixelObservationScaler
        from d3rlpy.algos import CQLConfig

        cql = CQLConfig(observation_scaler=PixelObservationScaler()).create()
    """

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

    def fit_with_env(self, env: GymEnv) -> None:
        pass

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return x.float() / 255.0

    def reverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x * 255.0).long()

    def transform_numpy(self, x: np.ndarray) -> np.ndarray:
        return x / 255.0

    def reverse_transform_numpy(self, x: np.ndarray) -> np.ndarray:
        return x * 255.0

    @staticmethod
    def get_type() -> str:
        return "pixel"

    @property
    def built(self) -> bool:
        return True


@dataclasses.dataclass()
class MinMaxObservationScaler(ObservationScaler):
    r"""Min-Max normalization preprocessing.

    Observations will be normalized in range ``[-1.0, 1.0]``.

    .. math::

        x' = (x - \min{x}) / (\max{x} - \min{x}) * 2 - 1

    .. code-block:: python

        from d3rlpy.preprocessing import MinMaxObservationScaler
        from d3rlpy.algos import CQLConfig

        # normalize based on datasets or environments
        cql = CQLConfig(observation_scaler=MinMaxObservationScaler()).create()

        # manually initialize
        minimum = observations.min(axis=0)
        maximum = observations.max(axis=0)
        observation_scaler = MinMaxObservationScaler(
            minimum=minimum,
            maximum=maximum,
        )
        cql = CQLConfig(observation_scaler=observation_scaler).create()

    Args:
        minimum (numpy.ndarray): Minimum values at each entry.
        maximum (numpy.ndarray): Maximum values at each entry.
    """
    minimum: Optional[np.ndarray] = make_optional_numpy_field()
    maximum: Optional[np.ndarray] = make_optional_numpy_field()

    def __post_init__(self) -> None:
        if self.minimum is not None:
            self.minimum = np.asarray(self.minimum)
        if self.maximum is not None:
            self.maximum = np.asarray(self.maximum)
        self._torch_minimum: Optional[torch.Tensor] = None
        self._torch_maximum: Optional[torch.Tensor] = None

    def fit_with_transition_picker(
        self,
        episodes: Sequence[EpisodeBase],
        transition_picker: TransitionPickerProtocol,
    ) -> None:
        assert not self.built
        maximum = np.zeros(episodes[0].observation_signature.shape[0])
        minimum = np.zeros(episodes[0].observation_signature.shape[0])
        for i, episode in enumerate(episodes):
            for j in range(episode.transition_count):
                transition = transition_picker(episode, j)
                observation = np.asarray(transition.observation)
                if i == 0 and j == 0:
                    minimum = observation
                    maximum = observation
                else:
                    minimum = np.minimum(minimum, observation)
                    maximum = np.maximum(maximum, observation)
        self.minimum = minimum
        self.maximum = maximum

    def fit_with_trajectory_slicer(
        self,
        episodes: Sequence[EpisodeBase],
        trajectory_slicer: TrajectorySlicerProtocol,
    ) -> None:
        assert not self.built
        maximum = np.zeros(episodes[0].observation_signature.shape[0])
        minimum = np.zeros(episodes[0].observation_signature.shape[0])
        for i, episode in enumerate(episodes):
            traj = trajectory_slicer(
                episode, episode.size() - 1, episode.size()
            )
            observations = np.asarray(traj.observations)
            max_observation = np.max(observations, axis=0)
            min_observation = np.min(observations, axis=0)
            if i == 0:
                minimum = min_observation
                maximum = max_observation
            else:
                minimum = np.minimum(minimum, min_observation)
                maximum = np.maximum(maximum, max_observation)
        self.minimum = minimum
        self.maximum = maximum

    def fit_with_env(self, env: GymEnv) -> None:
        assert not self.built
        assert isinstance(env.observation_space, (Box, GymnasiumBox))
        low = np.asarray(env.observation_space.low)
        high = np.asarray(env.observation_space.high)
        self.minimum = low
        self.maximum = high

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        assert self.built
        if self._torch_maximum is None or self._torch_minimum is None:
            self._set_torch_value(x.device)
        assert (
            self._torch_minimum is not None and self._torch_maximum is not None
        )
        minimum = add_leading_dims(self._torch_minimum, target=x)
        maximum = add_leading_dims(self._torch_maximum, target=x)
        return (x - minimum) / (maximum - minimum) * 2.0 - 1.0

    def reverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        assert self.built
        if self._torch_maximum is None or self._torch_minimum is None:
            self._set_torch_value(x.device)
        assert (
            self._torch_minimum is not None and self._torch_maximum is not None
        )
        minimum = add_leading_dims(self._torch_minimum, target=x)
        maximum = add_leading_dims(self._torch_maximum, target=x)
        return ((maximum - minimum) * (x + 1.0) / 2.0) + minimum

    def transform_numpy(self, x: np.ndarray) -> np.ndarray:
        assert self.built
        assert self.minimum is not None and self.maximum is not None
        minimum = add_leading_dims_numpy(self.minimum, target=x)
        maximum = add_leading_dims_numpy(self.maximum, target=x)
        return (x - minimum) / (maximum - minimum) * 2.0 - 1.0

    def reverse_transform_numpy(self, x: np.ndarray) -> np.ndarray:
        assert self.built
        assert self.minimum is not None and self.maximum is not None
        minimum = add_leading_dims_numpy(self.minimum, target=x)
        maximum = add_leading_dims_numpy(self.maximum, target=x)
        return ((maximum - minimum) * (x + 1.0) / 2.0) + minimum

    def _set_torch_value(self, device: torch.device) -> None:
        self._torch_minimum = torch.tensor(
            self.minimum, dtype=torch.float32, device=device
        )
        self._torch_maximum = torch.tensor(
            self.maximum, dtype=torch.float32, device=device
        )

    @staticmethod
    def get_type() -> str:
        return "min_max"

    @property
    def built(self) -> bool:
        return self.minimum is not None and self.maximum is not None


@dataclasses.dataclass()
class StandardObservationScaler(ObservationScaler):
    r"""Standardization preprocessing.

    .. math::

        x' = (x - \mu) / \sigma

    .. code-block:: python

        from d3rlpy.preprocessing import StandardObservationScaler
        from d3rlpy.algos import CQLConfig

        # normalize based on datasets
        cql = CQLConfig(observation_scaler=StandardObservationScaler()).create()

        # manually initialize
        mean = observations.mean(axis=0)
        std = observations.std(axis=0)
        observation_scaler = StandardObservationScaler(mean=mean, std=std)
        cql = CQLConfig(observation_scaler=observation_scaler).create()

    Args:
        mean (numpy.ndarray): Mean values at each entry.
        std (numpy.ndarray): Standard deviation at each entry.
        eps (float): Small constant value to avoid zero-division.
    """
    mean: Optional[np.ndarray] = make_optional_numpy_field()
    std: Optional[np.ndarray] = make_optional_numpy_field()
    eps: float = 1e-3

    def __post_init__(self) -> None:
        if self.mean is not None:
            self.mean = np.asarray(self.mean)
        if self.std is not None:
            self.std = np.asarray(self.std)
        self._torch_mean: Optional[torch.Tensor] = None
        self._torch_std: Optional[torch.Tensor] = None

    def fit_with_transition_picker(
        self,
        episodes: Sequence[EpisodeBase],
        transition_picker: TransitionPickerProtocol,
    ) -> None:
        assert not self.built
        # compute mean
        total_sum = np.zeros(episodes[0].observation_signature.shape[0])
        total_count = 0
        for episode in episodes:
            for i in range(episode.transition_count):
                transition = transition_picker(episode, i)
                total_sum += transition.observation
            total_count += episode.transition_count
        mean = total_sum / total_count

        # compute stdandard deviation
        total_sqsum = np.zeros(episodes[0].observation_signature.shape[0])
        for episode in episodes:
            for i in range(episode.transition_count):
                transition = transition_picker(episode, i)
                total_sqsum += (transition.observation - mean) ** 2
        std = np.sqrt(total_sqsum / total_count)

        self.mean = mean
        self.std = std

    def fit_with_trajectory_slicer(
        self,
        episodes: Sequence[EpisodeBase],
        trajectory_slicer: TrajectorySlicerProtocol,
    ) -> None:
        assert not self.built
        # compute mean
        total_sum = np.zeros(episodes[0].observation_signature.shape[0])
        total_count = 0
        for episode in episodes:
            traj = trajectory_slicer(
                episode, episode.size() - 1, episode.size()
            )
            total_sum += np.sum(traj.observations, axis=0)
            total_count += episode.size()
        mean = total_sum / total_count

        # compute stdandard deviation
        total_sqsum = np.zeros(episodes[0].observation_signature.shape[0])
        expanded_mean = mean.reshape((1,) + mean.shape)
        for episode in episodes:
            traj = trajectory_slicer(
                episode, episode.size() - 1, episode.size()
            )
            observations = np.asarray(traj.observations)
            total_sqsum += np.sum((observations - expanded_mean) ** 2, axis=0)
        std = np.sqrt(total_sqsum / total_count)

        self.mean = mean
        self.std = std

    def fit_with_env(self, env: GymEnv) -> None:
        raise NotImplementedError(
            "standard scaler does not support fit_with_env."
        )

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        assert self.built
        if self._torch_mean is None or self._torch_std is None:
            self._set_torch_value(x.device)
        assert self._torch_mean is not None and self._torch_std is not None
        mean = add_leading_dims(self._torch_mean, target=x)
        std = add_leading_dims(self._torch_std, target=x)
        return (x - mean) / (std + self.eps)

    def reverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        assert self.built
        if self._torch_mean is None or self._torch_std is None:
            self._set_torch_value(x.device)
        assert self._torch_mean is not None and self._torch_std is not None
        mean = add_leading_dims(self._torch_mean, target=x)
        std = add_leading_dims(self._torch_std, target=x)
        return ((std + self.eps) * x) + mean

    def transform_numpy(self, x: np.ndarray) -> np.ndarray:
        assert self.built
        assert self.mean is not None and self.std is not None
        mean = add_leading_dims_numpy(self.mean, target=x)
        std = add_leading_dims_numpy(self.std, target=x)
        return (x - mean) / (std + self.eps)

    def reverse_transform_numpy(self, x: np.ndarray) -> np.ndarray:
        assert self.built
        assert self.mean is not None and self.std is not None
        mean = add_leading_dims_numpy(self.mean, target=x)
        std = add_leading_dims_numpy(self.std, target=x)
        return ((std + self.eps) * x) + mean

    def _set_torch_value(self, device: torch.device) -> None:
        self._torch_mean = torch.tensor(
            self.mean, dtype=torch.float32, device=device
        )
        self._torch_std = torch.tensor(
            self.std, dtype=torch.float32, device=device
        )

    @staticmethod
    def get_type() -> str:
        return "standard"

    @property
    def built(self) -> bool:
        return self.mean is not None and self.std is not None


(
    register_observation_scaler,
    make_observation_scaler_field,
) = generate_optional_config_generation(
    ObservationScaler  # type: ignore
)


register_observation_scaler(PixelObservationScaler)
register_observation_scaler(MinMaxObservationScaler)
register_observation_scaler(StandardObservationScaler)

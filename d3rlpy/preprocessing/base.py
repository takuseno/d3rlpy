from abc import ABCMeta, abstractmethod
from typing import Sequence

import numpy as np
import torch

from ..dataset import (
    EpisodeBase,
    TrajectorySlicerProtocol,
    TransitionPickerProtocol,
)
from ..serializable_config import DynamicConfig
from ..types import GymEnv, NDArray

__all__ = ["Scaler", "add_leading_dims", "add_leading_dims_numpy"]


class Scaler(DynamicConfig, metaclass=ABCMeta):
    @abstractmethod
    def fit_with_transition_picker(
        self,
        episodes: Sequence[EpisodeBase],
        transition_picker: TransitionPickerProtocol,
    ) -> None:
        """Estimates scaling parameters from dataset.

        Args:
            episodes: List of episodes.
            transition_picker: Transition picker to process mini-batch.
        """
        raise NotImplementedError

    @abstractmethod
    def fit_with_trajectory_slicer(
        self,
        episodes: Sequence[EpisodeBase],
        trajectory_slicer: TrajectorySlicerProtocol,
    ) -> None:
        """Estimates scaling parameters from dataset.

        Args:
            episodes: List of episodes.
            trajectory_slicer: Trajectory slicer to process mini-batch.
        """
        raise NotImplementedError

    @abstractmethod
    def fit_with_env(self, env: GymEnv) -> None:
        """Gets scaling parameters from environment.

        Args:
            env: Gym environment.
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Returns processed output.

        Args:
            x: Input.

        Returns:
            Processed output.
        """
        raise NotImplementedError

    @abstractmethod
    def reverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Returns reversely transformed output.

        Args:
            x: input.

        Returns:
            Inversely transformed output.
        """
        raise NotImplementedError

    @abstractmethod
    def transform_numpy(self, x: NDArray) -> NDArray:
        """Returns processed output in numpy.

        Args:
            x: Input.

        Returns:
            Processed output.
        """
        raise NotImplementedError

    @abstractmethod
    def reverse_transform_numpy(self, x: NDArray) -> NDArray:
        """Returns reversely transformed output in numpy.

        Args:
            x: Input.

        Returns:
            Inversely transformed output.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def built(self) -> bool:
        """Returns a flag to represent if scaler is already built.

        Returns:
            The flag will be True if scaler is already built.
        """
        raise NotImplementedError


def add_leading_dims(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    assert x.ndim <= target.ndim
    dim_diff = target.ndim - x.ndim
    assert x.shape == target.shape[dim_diff:]
    return torch.reshape(x, [1] * dim_diff + list(x.shape))


def add_leading_dims_numpy(x: NDArray, target: NDArray) -> NDArray:
    assert x.ndim <= target.ndim
    dim_diff = target.ndim - x.ndim
    assert x.shape == target.shape[dim_diff:]
    return np.reshape(x, [1] * dim_diff + list(x.shape))

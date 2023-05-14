from abc import ABCMeta, abstractmethod
from typing import Any, Sequence

import gym
import numpy as np
import torch

from ..dataset import (
    EpisodeBase,
    TrajectorySlicerProtocol,
    TransitionPickerProtocol,
)
from ..serializable_config import DynamicConfig

__all__ = ["Scaler"]


class Scaler(DynamicConfig, metaclass=ABCMeta):
    @abstractmethod
    def fit_with_transition_picker(
        self,
        episodes: Sequence[EpisodeBase],
        transition_picker: TransitionPickerProtocol,
    ) -> None:
        """Estimates scaling parameters from dataset.

        Args:
            episodes: list of episodes.
            transition_picker: transition picker to process mini-batch.

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
            episodes: list of episodes.
            trajectory_slicer: trajectory slicer to process mini-batch.

        """
        raise NotImplementedError

    @abstractmethod
    def fit_with_env(self, env: gym.Env[Any, Any]) -> None:
        """Gets scaling parameters from environment.

        Args:
            env: gym environment.

        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Returns processed output.

        Args:
            x: input.

        Returns:
            processed output.

        """
        raise NotImplementedError

    @abstractmethod
    def reverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Returns reversely transformed output.

        Args:
            x: input.

        Returns:
            reversely transformed output.

        """
        raise NotImplementedError

    @abstractmethod
    def transform_numpy(self, x: np.ndarray) -> np.ndarray:
        """Returns processed output in numpy.

        Args:
            x: input.

        Returns:
            processed output.

        """
        raise NotImplementedError

    @abstractmethod
    def reverse_transform_numpy(self, x: np.ndarray) -> np.ndarray:
        """Returns reversely transformed output in numpy.

        Args:
            x: input.

        Returns:
            reversely transformed output.

        """
        raise NotImplementedError

    @property
    @abstractmethod
    def built(self) -> bool:
        """Returns a flag to represent if scaler is already built.

        Returns:
            the flag will be True if scaler is already built.

        """
        raise NotImplementedError

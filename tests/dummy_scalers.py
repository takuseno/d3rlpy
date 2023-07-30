from typing import Sequence

import numpy as np
import torch

from d3rlpy.dataset import (
    EpisodeBase,
    TrajectorySlicerProtocol,
    TransitionPickerProtocol,
)
from d3rlpy.envs import GymEnv
from d3rlpy.preprocessing import ActionScaler, ObservationScaler, RewardScaler


class DummyObservationScaler(ObservationScaler):
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
        return x + 0.1

    def reverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return x - 0.1

    def transform_numpy(self, x: np.ndarray) -> np.ndarray:
        return x + 0.1

    def reverse_transform_numpy(self, x: np.ndarray) -> np.ndarray:
        return x - 0.1

    @property
    def built(self) -> bool:
        return True


class DummyActionScaler(ActionScaler):
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
        return x + 0.2

    def reverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return x - 0.2

    def transform_numpy(self, x: np.ndarray) -> np.ndarray:
        return x + 0.2

    def reverse_transform_numpy(self, x: np.ndarray) -> np.ndarray:
        return x - 0.2

    @property
    def built(self) -> bool:
        return True


class DummyRewardScaler(RewardScaler):
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
        return x + 0.3

    def reverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return x - 0.3

    def transform_numpy(self, x: np.ndarray) -> np.ndarray:
        return x + 0.3

    def reverse_transform_numpy(self, x: np.ndarray) -> np.ndarray:
        return x - 0.3

    @property
    def built(self) -> bool:
        return True

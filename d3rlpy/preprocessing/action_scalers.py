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
from ..serializable_config import (
    generate_optional_config_generation,
    make_optional_numpy_field,
)
from .base import Scaler, add_leading_dims, add_leading_dims_numpy

__all__ = [
    "ActionScaler",
    "MinMaxActionScaler",
    "register_action_scaler",
    "make_action_scaler_field",
]


class ActionScaler(Scaler):
    pass


@dataclasses.dataclass()
class MinMaxActionScaler(ActionScaler):
    r"""Min-Max normalization action preprocessing.

    Actions will be normalized in range ``[-1.0, 1.0]``.

    .. math::

        a' = (a - \min{a}) / (\max{a} - \min{a}) * 2 - 1

    .. code-block:: python

        from d3rlpy.preprocessing import MinMaxActionScaler
        from d3rlpy.algos import CQLConfig

        # normalize based on datasets or environments
        cql = CQLConfig(action_scaler=MinMaxActionScaler()).create()

        # manually initialize
        minimum = actions.min(axis=0)
        maximum = actions.max(axis=0)
        action_scaler = MinMaxActionScaler(minimum=minimum, maximum=maximum)
        cql = CQLConfig(action_scaler=action_scaler).create()

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

    def fit_with_transition_picker(
        self,
        episodes: Sequence[EpisodeBase],
        transition_picker: TransitionPickerProtocol,
    ) -> None:
        assert not self.built
        minimum = np.zeros(episodes[0].action_signature.shape[0])
        maximum = np.zeros(episodes[0].action_signature.shape[0])
        for i, episode in enumerate(episodes):
            for j in range(episode.transition_count):
                transition = transition_picker(episode, j)
                if i == 0 and j == 0:
                    minimum = transition.action
                    maximum = transition.action
                else:
                    minimum = np.minimum(minimum, transition.action)
                    maximum = np.maximum(maximum, transition.action)
        self.minimum = minimum
        self.maximum = maximum

    def fit_with_trajectory_slicer(
        self,
        episodes: Sequence[EpisodeBase],
        trajectory_slicer: TrajectorySlicerProtocol,
    ) -> None:
        assert not self.built
        minimum = np.zeros(episodes[0].action_signature.shape[0])
        maximum = np.zeros(episodes[0].action_signature.shape[0])
        for i, episode in enumerate(episodes):
            traj = trajectory_slicer(
                episode, episode.size() - 1, episode.size()
            )
            actions = np.asarray(traj.actions)
            min_action = np.min(actions, axis=0)
            max_action = np.max(actions, axis=0)
            if i == 0:
                minimum = min_action
                maximum = max_action
            else:
                minimum = np.minimum(minimum, min_action)
                maximum = np.maximum(maximum, max_action)
        self.minimum = minimum
        self.maximum = maximum

    def fit_with_env(self, env: gym.Env[Any, Any]) -> None:
        assert not self.built
        assert isinstance(env.action_space, gym.spaces.Box)
        low = np.asarray(env.action_space.low)
        high = np.asarray(env.action_space.high)
        self.minimum = low
        self.maximum = high

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        assert self.built
        minimum = add_leading_dims(
            torch.tensor(self.minimum, dtype=torch.float32, device=x.device),
            target=x,
        )
        maximum = add_leading_dims(
            torch.tensor(self.maximum, dtype=torch.float32, device=x.device),
            target=x,
        )
        # transform action into [-1.0, 1.0]
        return ((x - minimum) / (maximum - minimum)) * 2.0 - 1.0

    def reverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        assert self.built
        minimum = add_leading_dims(
            torch.tensor(self.minimum, dtype=torch.float32, device=x.device),
            target=x,
        )
        maximum = add_leading_dims(
            torch.tensor(self.maximum, dtype=torch.float32, device=x.device),
            target=x,
        )
        # transform action from [-1.0, 1.0]
        return ((maximum - minimum) * ((x + 1.0) / 2.0)) + minimum

    def transform_numpy(self, x: np.ndarray) -> np.ndarray:
        assert self.built
        assert self.maximum is not None and self.minimum is not None
        minimum = add_leading_dims_numpy(self.minimum, target=x)
        maximum = add_leading_dims_numpy(self.maximum, target=x)
        # transform action into [-1.0, 1.0]
        return ((x - minimum) / (maximum - minimum)) * 2.0 - 1.0

    def reverse_transform_numpy(self, x: np.ndarray) -> np.ndarray:
        assert self.built
        assert self.maximum is not None and self.minimum is not None
        minimum = add_leading_dims_numpy(self.minimum, target=x)
        maximum = add_leading_dims_numpy(self.maximum, target=x)
        # transform action from [-1.0, 1.0]
        return ((maximum - minimum) * ((x + 1.0) / 2.0)) + minimum

    @staticmethod
    def get_type() -> str:
        return "min_max"

    @property
    def built(self) -> bool:
        return self.minimum is not None and self.maximum is not None


(
    register_action_scaler,
    make_action_scaler_field,
) = generate_optional_config_generation(
    ActionScaler  # type: ignore
)


register_action_scaler(MinMaxActionScaler)

import dataclasses
from typing import Any, Dict, Optional, Sequence

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
    "ActionScaler",
    "MinMaxActionScaler",
    "register_action_scaler",
    "make_action_scaler_field",
]


class ActionScaler(DynamicConfig):
    def fit(self, episodes: Sequence[EpisodeBase]) -> None:
        """Estimates scaling parameters from dataset.

        Args:
            episodes: a list of episode objects.

        """
        raise NotImplementedError

    def fit_with_env(self, env: gym.Env[Any, Any]) -> None:
        """Gets scaling parameters from environment.

        Args:
            env: gym environment.

        """
        raise NotImplementedError

    def transform(self, action: torch.Tensor) -> torch.Tensor:
        """Returns processed action.

        Args:
            action: action vector.

        Returns:
            processed action.

        """
        raise NotImplementedError

    def reverse_transform(self, action: torch.Tensor) -> torch.Tensor:
        """Returns reversely transformed action.

        Args:
            action: action vector.

        Returns:
            reversely transformed action.

        """
        raise NotImplementedError

    def reverse_transform_numpy(self, action: np.ndarray) -> np.ndarray:
        """Returns reversely transformed action in numpy array.

        Args:
            action: action vector.

        Returns:
            reversely transformed action.

        """
        raise NotImplementedError

    @staticmethod
    def get_type() -> str:
        """Returns action scaler type.

        Returns:
            action scaler type.

        """
        raise NotImplementedError

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        """Returns action scaler params.

        Args:
            deep: flag to deepcopy parameters.

        Returns:
            action scaler parameters.

        """
        raise NotImplementedError


@dataclasses.dataclass()
class MinMaxActionScaler(ActionScaler):
    r"""Min-Max normalization action preprocessing.

    Actions will be normalized in range ``[-1.0, 1.0]``.

    .. math::

        a' = (a - \min{a}) / (\max{a} - \min{a}) * 2 - 1

    .. code-block:: python

        from d3rlpy.dataset import MDPDataset
        from d3rlpy.algos import CQL

        dataset = MDPDataset(observations, actions, rewards, terminals)

        # initialize algorithm with MinMaxActionScaler
        cql = CQL(action_scaler='min_max')

        # scaler is initialized from the given transitions
        transitions = []
        for episode in dataset.episodes:
            transitions += episode.transitions
        cql.fit(transitions)

    You can also initialize with :class:`d3rlpy.dataset.MDPDataset` object or
    manually.

    .. code-block:: python

        from d3rlpy.preprocessing import MinMaxActionScaler

        # initialize manually
        minimum = actions.min(axis=0)
        maximum = actions.max(axis=0)
        action_scaler = MinMaxActionScaler(minimum=minimum, maximum=maximum)

        cql = CQL(action_scaler=action_scaler)

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

        minimum = np.zeros(episodes[0].action_shape)
        maximum = np.zeros(episodes[0].action_shape)
        for i, episode in enumerate(episodes):
            actions = np.asarray(episode.actions)
            min_action = np.min(actions, axis=0)
            max_action = np.max(actions, axis=0)
            if i == 0:
                minimum = min_action
                maximum = max_action
            else:
                minimum = np.minimum(minimum, min_action)
                maximum = np.maximum(maximum, max_action)

        self.minimum = minimum.reshape((1,) + minimum.shape)
        self.maximum = maximum.reshape((1,) + maximum.shape)

    def fit_with_env(self, env: gym.Env[Any, Any]) -> None:
        if self.minimum is not None and self.maximum is not None:
            return

        assert isinstance(env.action_space, gym.spaces.Box)
        shape = env.action_space.shape
        low = np.asarray(env.action_space.low)
        high = np.asarray(env.action_space.high)
        self.minimum = low.reshape((1,) + shape)
        self.maximum = high.reshape((1,) + shape)

    def transform(self, action: torch.Tensor) -> torch.Tensor:
        assert self.minimum is not None and self.maximum is not None
        minimum = torch.tensor(
            self.minimum, dtype=torch.float32, device=action.device
        )
        maximum = torch.tensor(
            self.maximum, dtype=torch.float32, device=action.device
        )
        # transform action into [-1.0, 1.0]
        return ((action - minimum) / (maximum - minimum)) * 2.0 - 1.0

    def reverse_transform(self, action: torch.Tensor) -> torch.Tensor:
        assert self.minimum is not None and self.maximum is not None
        minimum = torch.tensor(
            self.minimum, dtype=torch.float32, device=action.device
        )
        maximum = torch.tensor(
            self.maximum, dtype=torch.float32, device=action.device
        )
        # transform action from [-1.0, 1.0]
        return ((maximum - minimum) * ((action + 1.0) / 2.0)) + minimum

    def reverse_transform_numpy(self, action: np.ndarray) -> np.ndarray:
        assert self.minimum is not None and self.maximum is not None
        minimum, maximum = self.minimum, self.maximum
        # transform action from [-1.0, 1.0]
        return ((maximum - minimum) * ((action + 1.0) / 2.0)) + minimum

    @staticmethod
    def get_type() -> str:
        return "min_max"


(
    register_action_scaler,
    make_action_scaler_field,
) = generate_optional_config_generation(ActionScaler)


register_action_scaler(MinMaxActionScaler)

from dataclasses import field
from typing import Any, Dict, Optional, Sequence, Type

import gym
import numpy as np
import torch
from dataclasses_json import config

from ..dataset import EpisodeBase

__all__ = [
    "ActionScaler",
    "MinMaxActionScaler",
    "ACTION_SCALER_LIST",
    "register_action_scaler",
    "create_action_scaler",
    "make_action_scaler_field",
]


class ActionScaler:
    def fit(self, episodes: Sequence[EpisodeBase]) -> None:
        """Estimates scaling parameters from dataset.

        Args:
            episodes: a list of episode objects.

        """
        raise NotImplementedError

    def fit_with_env(self, env: gym.Env) -> None:
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
        min (numpy.ndarray): minimum values at each entry.
        max (numpy.ndarray): maximum values at each entry.

    """
    _minimum: Optional[np.ndarray]
    _maximum: Optional[np.ndarray]

    def __init__(
        self,
        maximum: Optional[np.ndarray] = None,
        minimum: Optional[np.ndarray] = None,
    ):
        self._minimum = None
        self._maximum = None
        if maximum is not None and minimum is not None:
            self._minimum = np.asarray(minimum)
            self._maximum = np.asarray(maximum)

    def fit(self, episodes: Sequence[EpisodeBase]) -> None:
        if self._minimum is not None and self._maximum is not None:
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

        self._minimum = minimum.reshape((1,) + minimum.shape)
        self._maximum = maximum.reshape((1,) + maximum.shape)

    def fit_with_env(self, env: gym.Env) -> None:
        if self._minimum is not None and self._maximum is not None:
            return

        assert isinstance(env.action_space, gym.spaces.Box)
        shape = env.action_space.shape
        low = np.asarray(env.action_space.low)
        high = np.asarray(env.action_space.high)
        self._minimum = low.reshape((1,) + shape)
        self._maximum = high.reshape((1,) + shape)

    def transform(self, action: torch.Tensor) -> torch.Tensor:
        assert self._minimum is not None and self._maximum is not None
        minimum = torch.tensor(
            self._minimum, dtype=torch.float32, device=action.device
        )
        maximum = torch.tensor(
            self._maximum, dtype=torch.float32, device=action.device
        )
        # transform action into [-1.0, 1.0]
        return ((action - minimum) / (maximum - minimum)) * 2.0 - 1.0

    def reverse_transform(self, action: torch.Tensor) -> torch.Tensor:
        assert self._minimum is not None and self._maximum is not None
        minimum = torch.tensor(
            self._minimum, dtype=torch.float32, device=action.device
        )
        maximum = torch.tensor(
            self._maximum, dtype=torch.float32, device=action.device
        )
        # transform action from [-1.0, 1.0]
        return ((maximum - minimum) * ((action + 1.0) / 2.0)) + minimum

    def reverse_transform_numpy(self, action: np.ndarray) -> np.ndarray:
        assert self._minimum is not None and self._maximum is not None
        minimum, maximum = self._minimum, self._maximum
        # transform action from [-1.0, 1.0]
        return ((maximum - minimum) * ((action + 1.0) / 2.0)) + minimum

    @staticmethod
    def get_type() -> str:
        return "min_max"

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        if self._minimum is not None:
            minimum = self._minimum.copy() if deep else self._minimum
        else:
            minimum = None

        if self._maximum is not None:
            maximum = self._maximum.copy() if deep else self._maximum
        else:
            maximum = None

        return {"minimum": minimum, "maximum": maximum}


ACTION_SCALER_LIST: Dict[str, Type[ActionScaler]] = {}


def register_action_scaler(cls: Type[ActionScaler]) -> None:
    """Registers action scaler class.

    Args:
        cls: action scaler class inheriting ``ActionScaler``.

    """
    type_name = cls.get_type()
    is_registered = type_name in ACTION_SCALER_LIST
    assert not is_registered, f"{type_name} seems to be already registered"
    ACTION_SCALER_LIST[type_name] = cls


def create_action_scaler(name: str, **kwargs: Any) -> ActionScaler:
    """Returns registered action scaler object.

    Args:
        name: regsitered scaler type name.
        kwargs: scaler arguments.

    Returns:
        scaler object.

    """
    assert name in ACTION_SCALER_LIST, f"{name} seems not to be registered."
    scaler = ACTION_SCALER_LIST[name](**kwargs)
    assert isinstance(scaler, ActionScaler)
    return scaler


def _encoder(scaler: Optional[ActionScaler]) -> Dict[str, Any]:
    if scaler is None:
        return {"type": "none", "params": {}}
    return {"type": scaler.get_type(), "params": scaler.get_params()}


def _decoder(dict_config: Dict[str, Any]) -> Optional[ActionScaler]:
    if dict_config["type"] == "none":
        return None
    return create_action_scaler(dict_config["type"], **dict_config["params"])


def make_action_scaler_field() -> Optional[ActionScaler]:
    return field(
        metadata=config(encoder=_encoder, decoder=_decoder), default=None
    )


register_action_scaler(MinMaxActionScaler)

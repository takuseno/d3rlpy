from typing import List, Union

import numpy as np
from typing_extensions import Protocol

from .buffers import BufferProtocol
from .components import Episode
from .types import Observation

__all__ = ["ExperienceWriter"]


class _ActiveEpisode:
    _observations: List[Observation]
    _actions: List[np.ndarray]
    _rewards: List[np.ndarray]

    def __init__(self):
        self._observations = []
        self._actions = []
        self._rewards = []

    def append(
        self,
        observation: Observation,
        action: Union[int, np.ndarray],
        reward: Union[float, np.ndarray],
    ) -> None:
        if isinstance(action, int):
            action = np.array([action])
        if isinstance(reward, (float, int)):
            reward = np.array([reward])
        self._observations.append(observation)
        self._actions.append(action)
        self._rewards.append(reward)

    def to_episode(self, terminated: bool) -> Episode:
        if isinstance(self._observations[0], (list, tuple)):
            obs_kinds = len(self._observations[0])
            observations = [
                np.array([obs[i] for obs in self._observations])
                for i in range(obs_kinds)
            ]
        elif isinstance(self._observations[0], np.ndarray):
            observations = np.array(self._observations)
        else:
            raise ValueError(
                f"invalid observations type: {type(self._observations[0])}"
            )
        return Episode(
            observations=observations,
            actions=np.array(self._actions),
            rewards=np.array(self._rewards),
            terminated=terminated,
        )


class ExperienceWriter:
    _buffer: BufferProtocol
    _active_episode: _ActiveEpisode

    def __init__(self, buffer: BufferProtocol):
        self._buffer = buffer
        self._active_episode = _ActiveEpisode()

    def write(
        self,
        observation: Observation,
        action: Union[int, np.ndarray],
        reward: Union[float, np.ndarray],
    ) -> None:
        self._active_episode.append(observation, action, reward)

    def clip_episode(self, terminated: bool) -> None:
        episode = self._active_episode.to_episode(terminated)
        self._active_episode = _ActiveEpisode()
        self._buffer.append(episode)

from typing import List, Union

import numpy as np
from typing_extensions import Protocol

from .buffers import BufferProtocol
from .components import Episode
from .types import Observation

__all__ = [
    "WriterPreprocessProtocol",
    "BasicWriterPreprocess",
    "LastFrameWriterPreprocess",
    "ExperienceWriter",
]


class WriterPreprocessProtocol(Protocol):
    def process_observation(self, observation: Observation) -> Observation:
        raise NotImplementedError

    def process_action(self, action: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def process_reward(self, reward: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class BasicWriterPreprocess(WriterPreprocessProtocol):
    def process_observation(self, observation: Observation) -> Observation:
        return observation

    def process_action(self, action: np.ndarray) -> np.ndarray:
        return action

    def process_reward(self, reward: np.ndarray) -> np.ndarray:
        return reward


class LastFrameWriterPreprocess(BasicWriterPreprocess):
    def process_observation(self, observation: Observation) -> Observation:
        if isinstance(observation, (list, tuple)):
            return [np.expand_dims(obs[-1], axis=0) for obs in observation]
        else:
            return np.expand_dims(observation[-1], axis=0)


class _ActiveEpisode:
    _preprocessor: WriterPreprocessProtocol
    _observations: List[Observation]
    _actions: List[np.ndarray]
    _rewards: List[np.ndarray]

    def __init__(self, preprocessor: WriterPreprocessProtocol) -> None:
        self._preprocessor = preprocessor
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

        # preprocess
        observation = self._preprocessor.process_observation(observation)
        action = self._preprocessor.process_action(action)
        reward = self._preprocessor.process_reward(reward)

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
    _preprocessor: WriterPreprocessProtocol
    _buffer: BufferProtocol
    _active_episode: _ActiveEpisode

    def __init__(
        self, buffer: BufferProtocol, preprocessor: WriterPreprocessProtocol
    ):
        self._buffer = buffer
        self._preprocessor = preprocessor
        self._active_episode = _ActiveEpisode(preprocessor)

    def write(
        self,
        observation: Observation,
        action: Union[int, np.ndarray],
        reward: Union[float, np.ndarray],
    ) -> None:
        self._active_episode.append(observation, action, reward)

    def clip_episode(self, terminated: bool) -> None:
        episode = self._active_episode.to_episode(terminated)
        self._active_episode = _ActiveEpisode(self._preprocessor)
        self._buffer.append(episode)

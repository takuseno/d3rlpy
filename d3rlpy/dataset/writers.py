from typing import Any, Dict, Sequence, Union

import numpy as np
from typing_extensions import Protocol

from .buffers import BufferProtocol
from .components import Episode, EpisodeBase, Signature
from .types import Observation, ObservationSequence
from .utils import get_dtype_from_observation, get_shape_from_observation

__all__ = [
    "WriterPreprocessProtocol",
    "BasicWriterPreprocess",
    "LastFrameWriterPreprocess",
    "ExperienceWriter",
]


class WriterPreprocessProtocol(Protocol):
    def process_observation(self, observation: Observation) -> Observation:
        r"""Processes observation.

        Args:
            observation: Observation.

        Returns:
            Processed observation.
        """
        raise NotImplementedError

    def process_action(self, action: np.ndarray) -> np.ndarray:
        r"""Processes action.

        Args:
            action: Action.

        Returns:
            Processed action.
        """
        raise NotImplementedError

    def process_reward(self, reward: np.ndarray) -> np.ndarray:
        r"""Processes reward.

        Args:
            reward: Reward.

        Returns:
            Processed reward.
        """
        raise NotImplementedError


class BasicWriterPreprocess(WriterPreprocessProtocol):
    """Stanard data writer.

    This class implements identity preprocess.
    """

    def process_observation(self, observation: Observation) -> Observation:
        return observation

    def process_action(self, action: np.ndarray) -> np.ndarray:
        return action

    def process_reward(self, reward: np.ndarray) -> np.ndarray:
        return reward


class LastFrameWriterPreprocess(BasicWriterPreprocess):
    """Data writer that writes the last channel of observation.

    This class is designed to be used with ``FrameStackTransitionPicker``.
    """

    def process_observation(self, observation: Observation) -> Observation:
        if isinstance(observation, (list, tuple)):
            return [np.expand_dims(obs[-1], axis=0) for obs in observation]
        else:
            return np.expand_dims(observation[-1], axis=0)


class _ActiveEpisode(EpisodeBase):
    _preprocessor: WriterPreprocessProtocol
    _cache_size: int
    _cursor: int
    _observation_signature: Signature
    _action_signature: Signature
    _reward_signature: Signature
    _observations: Sequence[np.ndarray]
    _actions: np.ndarray
    _rewards: np.ndarray
    _terminated: bool
    _frozen: bool

    def __init__(
        self,
        preprocessor: WriterPreprocessProtocol,
        cache_size: int,
        observation_signature: Signature,
        action_signature: Signature,
        reward_signature: Signature,
    ) -> None:
        self._preprocessor = preprocessor
        self._cache_size = cache_size
        self._cursor = 0
        shapes = observation_signature.shape
        dtypes = observation_signature.dtype
        self._observations = [
            np.empty((cache_size, *shape), dtype=dtype)
            for shape, dtype in zip(shapes, dtypes)
        ]
        self._actions = np.empty(
            (cache_size, *action_signature.shape[0]),
            dtype=action_signature.dtype[0],
        )
        self._rewards = np.empty(
            (cache_size, *reward_signature.shape[0]),
            dtype=reward_signature.dtype[0],
        )
        self._terminated = False
        self._observation_signature = observation_signature
        self._action_signature = action_signature
        self._reward_signature = reward_signature
        self._frozen = True

    def append(
        self,
        observation: Observation,
        action: Union[int, np.ndarray],
        reward: Union[float, np.ndarray],
    ) -> None:
        assert self._frozen, "This episode is already shrinked."
        assert (
            self._cursor < self._cache_size
        ), "episode length exceeds cache_size."

        if not isinstance(action, np.ndarray) or action.ndim == 0:
            action = np.array([action], dtype=self._action_signature.dtype[0])
        if not isinstance(reward, np.ndarray) or reward.ndim == 0:
            reward = np.array([reward], dtype=self._reward_signature.dtype[0])

        # preprocess
        observation = self._preprocessor.process_observation(observation)
        action = self._preprocessor.process_action(action)
        reward = self._preprocessor.process_reward(reward)

        if isinstance(observation, (list, tuple)):
            for i, obs in enumerate(observation):
                self._observations[i][self._cursor] = obs
        else:
            self._observations[0][self._cursor] = observation
        self._actions[self._cursor] = action
        self._rewards[self._cursor] = reward
        self._cursor += 1

    def to_episode(self, terminated: bool) -> Episode:
        if len(self._observations) == 1:
            observations = self._observations[0][: self._cursor].copy()
        else:
            observations = [
                obs[: self._cursor].copy() for obs in self._observations
            ]
        return Episode(
            observations=observations,
            actions=self._actions[: self._cursor].copy(),
            rewards=self._rewards[: self._cursor].copy(),
            terminated=terminated,
        )

    def shrink(self, terminated: bool) -> None:
        episode = self.to_episode(terminated)
        if isinstance(episode.observations, np.ndarray):
            self._observations = [episode.observations]
        else:
            self._observations = episode.observations
        self._actions = episode.actions
        self._rewards = episode.rewards
        self._terminated = terminated
        self._frozen = True

    def size(self) -> int:
        return self._cursor

    @property
    def observations(self) -> ObservationSequence:
        if len(self._observations) == 1:
            return self._observations[0][: self._cursor]
        else:
            return [obs[: self._cursor] for obs in self._observations]

    @property
    def actions(self) -> np.ndarray:
        return self._actions[: self._cursor]

    @property
    def rewards(self) -> np.ndarray:
        return self._rewards[: self._cursor]

    @property
    def terminated(self) -> bool:
        return self._terminated

    @property
    def observation_signature(self) -> Signature:
        return self._observation_signature

    @property
    def action_signature(self) -> Signature:
        return self._action_signature

    @property
    def reward_signature(self) -> Signature:
        return self._reward_signature

    def compute_return(self) -> float:
        return float(np.sum(self.rewards[: self._cursor]))

    def serialize(self) -> Dict[str, Any]:
        return {
            "observations": self.observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "terminated": self.terminated,
        }

    @classmethod
    def deserialize(cls, serializedData: Dict[str, Any]) -> "EpisodeBase":
        raise NotImplementedError("_ActiveEpisode cannot be deserialized.")

    def __len__(self) -> int:
        return self.size()

    @property
    def transition_count(self) -> int:
        return self.size() if self.terminated else self.size() - 1


class ExperienceWriter:
    """Experience writer.

    Args:
        buffer: Buffer.
        preprocessor: Writer preprocess.
        observation_signature: Signature of unprocessed observation.
        action_signature: Signature of unprocessed action.
        reward_signature: Signature of unprocessed reward.
        cache_size: Size of data in active episode. This needs to be larger
            than the maximum length of episodes.
    """

    _preprocessor: WriterPreprocessProtocol
    _buffer: BufferProtocol
    _cache_size: int
    _observation_signature: Signature
    _action_signature: Signature
    _reward_signature: Signature
    _active_episode: _ActiveEpisode
    _step: int

    def __init__(
        self,
        buffer: BufferProtocol,
        preprocessor: WriterPreprocessProtocol,
        observation_signature: Signature,
        action_signature: Signature,
        reward_signature: Signature,
        cache_size: int = 10000,
    ):
        self._buffer = buffer
        self._preprocessor = preprocessor
        self._cache_size = cache_size

        # preprocessed signatures
        if len(observation_signature.dtype) == 1:
            processed_observation = preprocessor.process_observation(
                observation_signature.sample()[0]
            )
            assert isinstance(processed_observation, np.ndarray)
            observation_signature = Signature(
                shape=[processed_observation.shape],
                dtype=[processed_observation.dtype],
            )
        else:
            processed_observation = preprocessor.process_observation(
                observation_signature.sample()
            )
            observation_shape = get_shape_from_observation(
                processed_observation
            )
            assert isinstance(observation_shape[0], (list, tuple))
            observation_dtype = get_dtype_from_observation(
                processed_observation
            )
            assert isinstance(observation_dtype, (list, tuple))
            observation_signature = Signature(
                shape=observation_shape,  # type: ignore
                dtype=observation_dtype,
            )
        processed_action = preprocessor.process_action(
            action_signature.sample()[0]
        )
        if (
            not isinstance(processed_action, np.ndarray)
            or processed_action.ndim == 0
        ):
            action_shape = (1,)
        else:
            action_shape = processed_action.shape
        action_signature = Signature(
            shape=[action_shape],
            dtype=[processed_action.dtype],
        )
        processed_reward = preprocessor.process_reward(
            reward_signature.sample()[0]
        )
        if (
            not isinstance(processed_reward, np.ndarray)
            or processed_reward.ndim == 0
        ):
            reward_shape = (1,)
        else:
            reward_shape = processed_reward.shape
        reward_signature = Signature(
            shape=[reward_shape],
            dtype=[processed_reward.dtype],
        )

        self._observation_signature = observation_signature
        self._action_signature = action_signature
        self._reward_signature = reward_signature
        self._active_episode = _ActiveEpisode(
            preprocessor,
            cache_size=cache_size,
            observation_signature=observation_signature,
            action_signature=action_signature,
            reward_signature=reward_signature,
        )

    def write(
        self,
        observation: Observation,
        action: Union[int, np.ndarray],
        reward: Union[float, np.ndarray],
    ) -> None:
        r"""Writes state tuple to buffer.

        Args:
            observation: Observation.
            action: Action.
            reward: Reward.
        """
        self._active_episode.append(observation, action, reward)
        if self._active_episode.transition_count > 0:
            self._buffer.append(
                episode=self._active_episode,
                index=self._active_episode.transition_count - 1,
            )

    def clip_episode(self, terminated: bool) -> None:
        r"""Clips the current episode.

        Args:
            terminated: Flag to represent environment termination.
        """
        if self._active_episode.transition_count == 0:
            return

        # shrink heap memory
        self._active_episode.shrink(terminated)

        # append terminal state if necessary
        if terminated:
            self._buffer.append(
                self._active_episode,
                self._active_episode.transition_count - 1,
            )

        # prepare next active episode
        self._active_episode = _ActiveEpisode(
            self._preprocessor,
            cache_size=self._cache_size,
            observation_signature=self._observation_signature,
            action_signature=self._action_signature,
            reward_signature=self._reward_signature,
        )

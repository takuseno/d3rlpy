import dataclasses
from typing import Any, Dict, Sequence

import numpy as np
from typing_extensions import Protocol

from ..constants import ActionSpace
from .types import Observation, ObservationSequence
from .utils import (
    detect_action_space,
    get_dtype_from_observation,
    get_dtype_from_observation_sequence,
    get_shape_from_observation,
    get_shape_from_observation_sequence,
)

__all__ = [
    "Signature",
    "Transition",
    "PartialTrajectory",
    "EpisodeBase",
    "Episode",
    "DatasetInfo",
]


@dataclasses.dataclass(frozen=True)
class Signature:
    dtype: Sequence[np.dtype]
    shape: Sequence[Sequence[int]]

    def sample(self) -> Sequence[np.ndarray]:
        return [
            np.random.random(shape).astype(dtype)
            for shape, dtype in zip(self.shape, self.dtype)
        ]


@dataclasses.dataclass(frozen=True)
class Transition:
    observation: Observation  # (...)
    action: np.ndarray  # (...)
    reward: np.ndarray  # (1,)
    next_observation: Observation  # (...)
    terminal: float
    interval: int

    @property
    def observation_signature(self) -> Signature:
        shape = get_shape_from_observation(self.observation)
        dtype = get_dtype_from_observation(self.observation)
        if isinstance(self.observation, np.ndarray):
            shape = [shape]  # type: ignore
            dtype = [dtype]
        return Signature(dtype=dtype, shape=shape)  # type: ignore

    @property
    def action_signature(self) -> Signature:
        return Signature(
            dtype=[self.action.dtype],
            shape=[self.action.shape],
        )

    @property
    def reward_signature(self) -> Signature:
        return Signature(
            dtype=[self.reward.dtype],
            shape=[self.reward.shape],
        )


@dataclasses.dataclass(frozen=True)
class PartialTrajectory:
    observations: ObservationSequence  # (L, ...)
    actions: np.ndarray  # (L, ...)
    rewards: np.ndarray  # (L, 1)
    returns_to_go: np.ndarray  # (L, 1)
    terminals: np.ndarray  # (L, 1)
    timesteps: np.ndarray  # (L,)
    masks: np.ndarray  # (L,)
    length: int

    @property
    def observation_signature(self) -> Signature:
        shape = get_shape_from_observation_sequence(self.observations)
        dtype = get_dtype_from_observation_sequence(self.observations)
        if isinstance(self.observations, np.ndarray):
            shape = [shape]  # type: ignore
            dtype = [dtype]
        return Signature(dtype=dtype, shape=shape)  # type: ignore

    @property
    def action_signature(self) -> Signature:
        return Signature(
            dtype=[self.actions.dtype],
            shape=[self.actions.shape[1:]],
        )

    @property
    def reward_signature(self) -> Signature:
        return Signature(
            dtype=[self.rewards.dtype],
            shape=[self.rewards.shape[1:]],
        )

    def __len__(self) -> int:
        return self.length


class EpisodeBase(Protocol):
    @property
    def observations(self) -> ObservationSequence:
        raise NotImplementedError

    @property
    def actions(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def rewards(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def terminated(self) -> bool:
        raise NotImplementedError

    @property
    def observation_signature(self) -> Signature:
        raise NotImplementedError

    @property
    def action_signature(self) -> Signature:
        raise NotImplementedError

    @property
    def reward_signature(self) -> Signature:
        raise NotImplementedError

    def size(self) -> int:
        raise NotImplementedError

    def compute_return(self) -> float:
        raise NotImplementedError

    def serialize(self) -> Dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def deserialize(cls, serializedData: Dict[str, Any]) -> "EpisodeBase":
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    @property
    def transition_count(self) -> int:
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class Episode:
    observations: ObservationSequence
    actions: np.ndarray
    rewards: np.ndarray
    terminated: bool

    @property
    def observation_signature(self) -> Signature:
        shape = get_shape_from_observation_sequence(self.observations)
        dtype = get_dtype_from_observation_sequence(self.observations)
        if isinstance(self.observations, np.ndarray):
            shape = [shape]  # type: ignore
            dtype = [dtype]
        return Signature(dtype=dtype, shape=shape)  # type: ignore

    @property
    def action_signature(self) -> Signature:
        return Signature(
            dtype=[self.actions.dtype],
            shape=[self.actions.shape[1:]],
        )

    @property
    def reward_signature(self) -> Signature:
        return Signature(
            dtype=[self.rewards.dtype],
            shape=[self.rewards.shape[1:]],
        )

    def size(self) -> int:
        return int(self.actions.shape[0])

    def compute_return(self) -> float:
        return float(np.sum(self.rewards))

    def serialize(self) -> Dict[str, Any]:
        return {
            "observations": self.observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "terminated": self.terminated,
        }

    @classmethod
    def deserialize(cls, serializedData: Dict[str, Any]) -> "Episode":
        return cls(
            observations=serializedData["observations"],
            actions=serializedData["actions"],
            rewards=serializedData["rewards"],
            terminated=serializedData["terminated"],
        )

    def __len__(self) -> int:
        return self.actions.shape[0]  # type: ignore

    @property
    def transition_count(self) -> int:
        return self.size() if self.terminated else self.size() - 1


@dataclasses.dataclass(frozen=True)
class DatasetInfo:
    observation_signature: Signature
    action_signature: Signature
    reward_signature: Signature
    action_space: ActionSpace
    action_size: int

    @classmethod
    def from_episodes(cls, episodes: Sequence[EpisodeBase]) -> "DatasetInfo":
        action_space = detect_action_space(episodes[0].actions)
        if action_space == ActionSpace.CONTINUOUS:
            action_size = episodes[0].action_signature.shape[0][0]
        else:
            max_action = 0
            for episode in episodes:
                max_action = max(int(np.max(episode.actions)), max_action)
            action_size = max_action + 1  # index should start from 0
        return DatasetInfo(
            observation_signature=episodes[0].observation_signature,
            action_signature=episodes[0].action_signature,
            reward_signature=episodes[0].reward_signature,
            action_space=action_space,
            action_size=action_size,
        )

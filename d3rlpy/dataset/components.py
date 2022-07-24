import dataclasses
from typing import Any, Dict, Sequence

import numpy as np

from .types import Observation, ObservationSequence, Shape
from .utils import (
    get_shape_from_observation,
    get_shape_from_observation_sequence,
)

__all__ = ["Transition", "PartialTrajectory", "EpisodeBase", "Episode"]


@dataclasses.dataclass(frozen=True)
class Transition:
    observation: Observation  # (...)
    action: np.ndarray  # (...)
    reward: np.ndarray  # (1,)
    next_observation: Observation  # (...)
    terminal: float
    interval: int

    @property
    def observation_shape(self) -> Shape:
        return get_shape_from_observation(self.observation)

    @property
    def action_shape(self) -> Sequence[int]:
        return self.action.shape

    @property
    def reward_shape(self) -> Sequence[int]:
        return self.reward.shape


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
    def observation_shape(self) -> Shape:
        return get_shape_from_observation_sequence(self.observations)

    @property
    def action_shape(self) -> Sequence[int]:
        return self.actions.shape[1:]

    @property
    def reward_shape(self) -> Sequence[int]:
        return self.rewards.shape[1:]


class EpisodeBase:
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
    def observation_shape(self) -> Shape:
        return get_shape_from_observation_sequence(self.observations)

    @property
    def action_shape(self) -> Sequence[int]:
        return self.actions.shape[1:]

    @property
    def reward_shape(self) -> Sequence[int]:
        return self.rewards.shape[1:]

    def size(self) -> int:
        return self.actions.shape[0]

    def compute_return(self) -> float:
        return np.sum(self.rewards)

    def serialize(self) -> Dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def deserialize(cls, serializedData: Dict[str, Any]) -> "EpisodeBase":
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class Episode(EpisodeBase):
    observations: ObservationSequence
    actions: np.ndarray
    rewards: np.ndarray
    terminated: bool

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

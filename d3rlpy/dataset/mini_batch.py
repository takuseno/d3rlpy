import dataclasses
from typing import Sequence, Union

import numpy as np

from .components import PartialTrajectory, Transition
from .types import Shape
from .utils import (
    cast_recursively,
    check_dtype,
    check_non_1d_array,
    get_shape_from_observation_sequence,
    stack_observations,
)

__all__ = ["TransitionMiniBatch", "TrajectoryMiniBatch"]


@dataclasses.dataclass(frozen=True)
class TransitionMiniBatch:
    observations: Union[np.ndarray, Sequence[np.ndarray]]  # (B, ...)
    actions: np.ndarray  # (B, ...)
    rewards: np.ndarray  # (B, 1)
    next_observations: Union[np.ndarray, Sequence[np.ndarray]]  # (B, ...)
    terminals: np.ndarray  # (B, 1)
    intervals: np.ndarray  # (B, 1)

    def __post_init__(self) -> None:
        assert check_non_1d_array(self.observations)
        assert check_dtype(self.observations, np.float32)
        assert check_non_1d_array(self.actions)
        assert check_dtype(self.actions, np.float32)
        assert check_non_1d_array(self.rewards)
        assert check_dtype(self.rewards, np.float32)
        assert check_non_1d_array(self.next_observations)
        assert check_dtype(self.next_observations, np.float32)
        assert check_non_1d_array(self.terminals)
        assert check_dtype(self.terminals, np.float32)
        assert check_non_1d_array(self.intervals)
        assert check_dtype(self.intervals, np.float32)

    @classmethod
    def from_transitions(
        cls, transitions: Sequence[Transition]
    ) -> "TransitionMiniBatch":
        observations = stack_observations(
            [transition.observation for transition in transitions]
        )
        actions = np.stack(
            [transition.action for transition in transitions], axis=0
        )
        rewards = np.stack(
            [transition.reward for transition in transitions], axis=0
        )
        next_observations = stack_observations(
            [transition.next_observation for transition in transitions]
        )
        terminals = np.reshape(
            np.array([transition.terminal for transition in transitions]),
            [-1, 1],
        )
        intervals = np.reshape(
            np.array([transition.terminal for transition in transitions]),
            [-1, 1],
        )
        return TransitionMiniBatch(
            observations=cast_recursively(observations, np.float32),
            actions=cast_recursively(actions, np.float32),
            rewards=cast_recursively(rewards, np.float32),
            next_observations=cast_recursively(next_observations, np.float32),
            terminals=cast_recursively(terminals, np.float32),
            intervals=cast_recursively(intervals, np.float32),
        )

    @property
    def observation_shape(self) -> Shape:
        return get_shape_from_observation_sequence(self.observations)

    @property
    def action_shape(self) -> Sequence[int]:
        return self.actions.shape[1:]  # type: ignore

    @property
    def reward_shape(self) -> Sequence[int]:
        return self.rewards.shape[1:]  # type: ignore

    def __len__(self) -> int:
        return int(self.actions.shape[0])


@dataclasses.dataclass(frozen=True)
class TrajectoryMiniBatch:
    observations: Union[np.ndarray, Sequence[np.ndarray]]  # (B, L, ...)
    actions: np.ndarray  # (B, L, ...)
    rewards: np.ndarray  # (B, L, 1)
    returns_to_go: np.ndarray  # (B, L, 1)
    terminals: np.ndarray  # (B, L, 1)
    timesteps: np.ndarray  # (B, L)
    masks: np.ndarray  # (B, L)
    length: int

    def __post_init__(self) -> None:
        assert check_dtype(self.observations, np.float32)
        assert check_dtype(self.actions, np.float32)
        assert check_dtype(self.rewards, np.float32)
        assert check_dtype(self.returns_to_go, np.float32)
        assert check_dtype(self.terminals, np.float32)
        assert check_dtype(self.timesteps, np.float32)
        assert check_dtype(self.masks, np.float32)

    @classmethod
    def from_partial_trajectories(
        cls, trajectories: Sequence[PartialTrajectory]
    ) -> "TrajectoryMiniBatch":
        observations = stack_observations(
            [traj.observations for traj in trajectories]
        )
        actions = np.stack([traj.actions for traj in trajectories], axis=0)
        rewards = np.stack([traj.rewards for traj in trajectories], axis=0)
        returns_to_go = np.stack(
            [traj.returns_to_go for traj in trajectories], axis=0
        )
        terminals = np.stack([traj.terminals for traj in trajectories], axis=0)
        timesteps = np.stack([traj.timesteps for traj in trajectories], axis=0)
        masks = np.stack([traj.masks for traj in trajectories], axis=0)
        return TrajectoryMiniBatch(
            observations=cast_recursively(observations, np.float32),
            actions=cast_recursively(actions, np.float32),
            rewards=cast_recursively(rewards, np.float32),
            returns_to_go=cast_recursively(returns_to_go, np.float32),
            terminals=cast_recursively(terminals, np.float32),
            timesteps=cast_recursively(timesteps, np.float32),
            masks=cast_recursively(masks, np.float32),
            length=trajectories[0].length,
        )

    @property
    def observation_shape(self) -> Shape:
        return get_shape_from_observation_sequence(self.observations)

    @property
    def action_shape(self) -> Sequence[int]:
        return self.actions.shape[1:]  # type: ignore

    @property
    def reward_shape(self) -> Sequence[int]:
        return self.rewards.shape[1:]  # type: ignore

    def __len__(self) -> int:
        return int(self.actions.shape[0])

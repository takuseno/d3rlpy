import dataclasses
from typing import Sequence, Union

import numpy as np

from ..types import Float32NDArray, Shape
from .components import PartialTrajectory, Transition
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
    r"""Mini-batch of transitions.

    Args:
        observations: Batched observations.
        actions: Batched actions.
        rewards: Batched rewards.
        next_observations: Batched next observations.
        returns_to_go: Batched returns-to-go.
        terminals: Batched environment terminal flags.
        intervals: Batched timesteps between observations and next
            observations.
        transitions: List of transitions.
    """

    observations: Union[Float32NDArray, Sequence[Float32NDArray]]  # (B, ...)
    actions: Float32NDArray  # (B, ...)
    rewards: Float32NDArray  # (B, 1)
    next_observations: Union[
        Float32NDArray, Sequence[Float32NDArray]
    ]  # (B, ...)
    next_actions: Float32NDArray  # (B, ...)
    terminals: Float32NDArray  # (B, 1)
    intervals: Float32NDArray  # (B, 1)
    transitions: Sequence[Transition]
    embeddings: Float32NDArray # (B, 1)

    def __post_init__(self) -> None:
        assert check_non_1d_array(self.observations)
        assert check_dtype(self.observations, np.float32)
        assert check_non_1d_array(self.actions)
        assert check_dtype(self.actions, np.float32)
        assert check_non_1d_array(self.next_actions)
        assert check_dtype(self.next_actions, np.float32)
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
        r"""Constructs mini-batch from list of transitions.

        Args:
            transitions: List of transitions.

        Returns:
            Mini-batch.
        """
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
        next_actions = np.stack(
            [transition.next_action for transition in transitions], axis=0
        )
        terminals = np.reshape(
            np.array([transition.terminal for transition in transitions]),
            [-1, 1],
        )
        intervals = np.reshape(
            np.array([transition.interval for transition in transitions]),
            [-1, 1],
        )
        embeddings = np.stack(
            [transition.embedding for transition in transitions], axis=0
        )

        return TransitionMiniBatch(
            observations=cast_recursively(observations, np.float32),
            actions=cast_recursively(actions, np.float32),
            rewards=cast_recursively(rewards, np.float32),
            next_observations=cast_recursively(next_observations, np.float32),
            next_actions=cast_recursively(next_actions, np.float32),
            terminals=cast_recursively(terminals, np.float32),
            intervals=cast_recursively(intervals, np.float32),
            transitions=transitions,
            embeddings=cast_recursively(embeddings, np.float32),
        )

    @property
    def observation_shape(self) -> Shape:
        r"""Returns observation shape.

        Returns:
            Observation shape.
        """
        return get_shape_from_observation_sequence(self.observations)

    @property
    def action_shape(self) -> Sequence[int]:
        r"""Returns action shape.

        Returns:
            Action shape.
        """
        return self.actions.shape[1:]

    @property
    def reward_shape(self) -> Sequence[int]:
        r"""Returns reward shape.

        Returns:
            Reward shape.
        """
        return self.rewards.shape[1:]

    def __len__(self) -> int:
        return int(self.actions.shape[0])


@dataclasses.dataclass(frozen=True)
class TrajectoryMiniBatch:
    r"""Mini-batch of trajectories.

    Args:
        observations: Batched sequence of observations.
        actions: Batched sequence of actions.
        rewards: Batched sequence of rewards.
        returns_to_go: Batched sequence of returns-to-go.
        terminals: Batched sequence of environment terminal flags.
        timesteps: Batched sequence of environment timesteps.
        masks: Batched masks that represent padding.
        length: Length of trajectories.
    """

    observations: Union[Float32NDArray, Sequence[Float32NDArray]]  # (B, L, ...)
    actions: Float32NDArray  # (B, L, ...)
    rewards: Float32NDArray  # (B, L, 1)
    returns_to_go: Float32NDArray  # (B, L, 1)
    terminals: Float32NDArray  # (B, L, 1)
    timesteps: Float32NDArray  # (B, L)
    masks: Float32NDArray  # (B, L)
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
        r"""Constructs mini-batch from list of trajectories.

        Args:
            trajectories: List of trajectories.

        Returns:
            Mini-batch of trajectories.
        """
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
        r"""Returns observation shape.

        Returns:
            Observation shape.
        """
        return get_shape_from_observation_sequence(self.observations)

    @property
    def action_shape(self) -> Sequence[int]:
        r"""Returns action shape.

        Returns:
            Action shape.
        """
        return self.actions.shape[1:]

    @property
    def reward_shape(self) -> Sequence[int]:
        r"""Returns reward shape.

        Returns:
            Reward shape.
        """
        return self.rewards.shape[1:]

    def __len__(self) -> int:
        return int(self.actions.shape[0])

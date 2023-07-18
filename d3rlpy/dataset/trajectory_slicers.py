import numpy as np
from typing_extensions import Protocol

from .components import EpisodeBase, PartialTrajectory
from .utils import batch_pad_array, batch_pad_observations, slice_observations

__all__ = ["TrajectorySlicerProtocol", "BasicTrajectorySlicer"]


class TrajectorySlicerProtocol(Protocol):
    r"""Interface of TrajectorySlicer."""

    def __call__(
        self, episode: EpisodeBase, end_index: int, size: int
    ) -> PartialTrajectory:
        r"""Slice trajectory.

        This method returns a partial trajectory from ``t=end_index-size`` to
        ``t=end_index``. If ``end_index-size`` is smaller than 0, those parts
        will be padded by zeros.

        Args:
            episode: Episode.
            end_index: Index at the end of the sliced trajectory.
            size: Length of the sliced trajectory.

        Returns:
            Sliced trajectory.
        """
        raise NotImplementedError


class BasicTrajectorySlicer(TrajectorySlicerProtocol):
    r"""Standard trajectory slicer.

    This class implements a basic trajectory slicing.
    """

    def __call__(
        self, episode: EpisodeBase, end_index: int, size: int
    ) -> PartialTrajectory:
        end = end_index + 1
        start = max(end - size, 0)
        actual_size = end - start

        # prepare terminal flags
        terminals = np.zeros((actual_size, 1), dtype=np.float32)
        if episode.terminated and end_index == episode.size() - 1:
            terminals[-1][0] = 1.0

        # slice data
        observations = slice_observations(episode.observations, start, end)
        actions = episode.actions[start:end]
        rewards = episode.rewards[start:end]
        ret = np.sum(episode.rewards[start:])
        all_returns_to_go = ret - np.cumsum(episode.rewards[start:], axis=0)
        returns_to_go = all_returns_to_go[:actual_size].reshape((-1, 1))

        # prepare metadata
        timesteps = np.arange(start, end)
        masks = np.ones(end - start, dtype=np.float32)

        # compute backward padding size
        pad_size = size - actual_size

        if pad_size == 0:
            return PartialTrajectory(
                observations=observations,
                actions=actions,
                rewards=rewards,
                returns_to_go=returns_to_go,
                terminals=terminals,
                timesteps=timesteps,
                masks=masks,
                length=size,
            )

        return PartialTrajectory(
            observations=batch_pad_observations(observations, pad_size),
            actions=batch_pad_array(actions, pad_size),
            rewards=batch_pad_array(rewards, pad_size),
            returns_to_go=batch_pad_array(returns_to_go, pad_size),
            terminals=batch_pad_array(terminals, pad_size),
            timesteps=batch_pad_array(timesteps, pad_size),
            masks=batch_pad_array(masks, pad_size),
            length=size,
        )

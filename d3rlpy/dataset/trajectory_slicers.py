import numpy as np
from typing_extensions import Protocol

from .components import EpisodeBase, PartialTrajectory
from .utils import batch_pad_array, batch_pad_observations, slice_observations

__all__ = ["TrajectorySlicerProtocol", "BasicTrajectorySlicer"]


class TrajectorySlicerProtocol(Protocol):
    def __call__(
        self, episode: EpisodeBase, start: int, size: int
    ) -> PartialTrajectory:
        ...


class BasicTrajectorySlicer(TrajectorySlicerProtocol):
    def __call__(
        self, episode: EpisodeBase, start: int, size: int
    ) -> PartialTrajectory:
        end = min(start + size, episode.size())

        # prepare terminal flags
        terminals = np.zeros(end - start, dtype=np.float32)
        if episode.terminated and end == episode.size():
            terminals[-1] = 1.0

        # slice data
        observations = slice_observations(episode.observations, start, end)
        actions = episode.actions[start:end]
        rewards = episode.rewards[start:end]
        returns_to_go = np.cumsum(rewards)

        # prepare metadata
        timesteps = np.arange(start, end)
        masks = np.ones(end - start, dtype=np.float32)

        # compute padding size
        pad_size = size - (end - start)

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

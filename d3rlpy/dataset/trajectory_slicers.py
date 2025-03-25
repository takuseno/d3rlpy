import numpy as np
from typing_extensions import Protocol

from ..types import Float32NDArray, Int32NDArray
from .components import EpisodeBase, PartialTrajectory
from .utils import batch_pad_array, batch_pad_observations, slice_observations

__all__ = [
    "TrajectorySlicerProtocol",
    "BasicTrajectorySlicer",
    "FrameStackTrajectorySlicer",
]


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
        terminals: Float32NDArray = np.zeros((actual_size, 1), dtype=np.float32)
        if episode.terminated and end_index == episode.size() - 1:
            terminals[-1][0] = 1.0

        # slice data
        observations = slice_observations(episode.observations, start, end)
        actions = episode.actions[start:end]
        rewards = episode.rewards[start:end]
        ret = np.sum(episode.rewards[start:])
        # cumsum includes the current timestep
        all_returns_to_go = (
            ret
            - np.cumsum(episode.rewards[start:], axis=0)
            + episode.rewards[start:]
        )
        returns_to_go = all_returns_to_go[:actual_size].reshape((-1, 1))

        # prepare metadata
        timesteps: Int32NDArray = np.arange(start, end) + 1
        masks: Float32NDArray = np.ones(end - start, dtype=np.float32)

        # compute backward padding size
        pad_size = size - actual_size

        embeddings = episode.embeddings[start:end] if episode.embeddings is not None else None

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
                embeddings=embeddings,
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
            embeddings=None if embeddings is None else batch_pad_array(embeddings, pad_size),
        )


class FrameStackTrajectorySlicer(TrajectorySlicerProtocol):
    r"""Frame-stacking trajectory slicer.

    This class implements the frame-stacking logic. The observations are
    stacked with the last ``n_frames-1`` frames. When ``index`` specifies
    timestep below ``n_frames``, those frames are padded by zeros.

    .. code-block:: python

        episode = Episode(
            observations=np.random.random((100, 1, 84, 84)),
            actions=np.random.random((100, 2)),
            rewards=np.random.random((100, 1)),
            terminated=False,
        )

        frame_stacking_slicer = FrameStackTrajectorySlicer(n_frames=4)
        trajectory = frame_stacking_slicer(episode, 0, 10)

        trajectory.observations.shape == (10, 4, 84, 84)

    Args:
        n_frames: Number of frames to stack.
    """

    _n_frames: int

    def __init__(self, n_frames: int):
        assert n_frames > 0
        self._n_frames = n_frames

    def __call__(
        self, episode: EpisodeBase, end_index: int, size: int
    ) -> PartialTrajectory:
        end = end_index + 1
        start = max(end - size, 0)
        actual_size = end - start

        # prepare terminal flags
        terminals: Float32NDArray = np.zeros((actual_size, 1), dtype=np.float32)
        if episode.terminated and end_index == episode.size() - 1:
            terminals[-1][0] = 1.0

        # prepare stacked observation data with zero initialization
        assert (
            len(episode.observation_signature.shape) == 1
        ), "Tuple observations are not supported yet."
        stacked_shape = list(episode.observation_signature.shape[0])
        channel_size = stacked_shape[0]
        image_shape = stacked_shape[1:]
        stacked_observations = np.zeros(
            (self._n_frames, actual_size, channel_size, *image_shape)
        )

        # fill stacked observations
        for i in range(self._n_frames):
            offset = self._n_frames - i - 1
            frame_start = max(start - offset, 0)
            frame_end = max(end - offset, 0)
            pad_size = actual_size - (frame_end - frame_start)
            observations = slice_observations(
                episode.observations, frame_start, frame_end
            )
            stacked_observations[i, pad_size:] = observations

        # (N, T, C, W, H) -> (T, N, C, W, H)
        stacked_observations = np.swapaxes(stacked_observations, 0, 1)
        # (T, N, C, W, H) -> (T, N * C, W, H)
        stacked_observations = np.reshape(
            stacked_observations,
            [actual_size, channel_size * self._n_frames, *image_shape],
        )

        # slice data
        actions = episode.actions[start:end]
        rewards = episode.rewards[start:end]
        ret = np.sum(episode.rewards[start:])
        # cumsum includes the current timestep
        all_returns_to_go = (
            ret
            - np.cumsum(episode.rewards[start:], axis=0)
            + episode.rewards[start:]
        )
        returns_to_go = all_returns_to_go[:actual_size].reshape((-1, 1))

        # prepare metadata
        timesteps: Int32NDArray = np.arange(start, end) + 1
        masks: Float32NDArray = np.ones(end - start, dtype=np.float32)

        # compute backward padding size
        pad_size = size - actual_size

        if pad_size == 0:
            return PartialTrajectory(
                observations=stacked_observations,
                actions=actions,
                rewards=rewards,
                returns_to_go=returns_to_go,
                terminals=terminals,
                timesteps=timesteps,
                masks=masks,
                length=size,
            )

        return PartialTrajectory(
            observations=batch_pad_observations(stacked_observations, pad_size),
            actions=batch_pad_array(actions, pad_size),
            rewards=batch_pad_array(rewards, pad_size),
            returns_to_go=batch_pad_array(returns_to_go, pad_size),
            terminals=batch_pad_array(terminals, pad_size),
            timesteps=batch_pad_array(timesteps, pad_size),
            masks=batch_pad_array(masks, pad_size),
            length=size,
        )

from typing import BinaryIO, List, Optional, Sequence, Type, Union

import numpy as np

from .buffers import BufferProtocol, FIFOBuffer, InfiniteBuffer
from .components import Episode, EpisodeBase, PartialTrajectory, Transition
from .episode_generator import EpisodeGeneratorProtocol
from .io import dump, load
from .mini_batch import TrajectoryMiniBatch, TransitionMiniBatch
from .trajectory_slicers import TrajectorySlicerProtocol
from .transition_pickers import TransitionPickerProtocol
from .types import Observation
from .writers import ExperienceWriter

__all__ = [
    "ReplayBuffer",
    "create_fifo_replay_buffer",
    "create_infinite_replay_buffer",
]


class ReplayBuffer:
    _buffer: BufferProtocol
    _writer: ExperienceWriter
    _episodes: List[EpisodeBase]

    def __init__(
        self,
        buffer: BufferProtocol,
        episodes: Optional[Sequence[EpisodeBase]] = None,
    ):
        self._buffer = buffer
        self._writer = ExperienceWriter(buffer)
        if episodes:
            for episode in episodes:
                self.append_episode(episode)

    def append(
        self,
        observation: Observation,
        action: Union[int, np.ndarray],
        reward: Union[float, np.ndarray],
    ) -> None:
        self._writer.write(observation, action, reward)

    def append_episode(self, episode: EpisodeBase) -> None:
        self._buffer.append(episode)

    def clip_episode(self, terminated: bool) -> None:
        self._writer.clip_episode(terminated)

    def sample_transition(
        self, sampler: TransitionPickerProtocol
    ) -> Transition:
        episode_index = np.random.randint(len(self.episodes))
        episode = self.episodes[episode_index]
        transition_index = np.random.randint(episode.size())
        return sampler(episode, transition_index)

    def sample_transition_batch(
        self, sampler: TransitionPickerProtocol, batch_size: int
    ) -> TransitionMiniBatch:
        return TransitionMiniBatch.from_transitions(
            [self.sample_transition(sampler) for _ in range(batch_size)]
        )

    def sample_trajectory(
        self, slicer: TrajectorySlicerProtocol, length: int
    ) -> PartialTrajectory:
        episode_index = np.random.randint(len(self.episodes))
        episode = self.episodes[episode_index]
        transition_index = np.random.randint(episode.size())
        return slicer(episode, transition_index, length)

    def sample_trajectory_batch(
        self, slicer: TrajectorySlicerProtocol, batch_size: int, length: int
    ) -> TrajectoryMiniBatch:
        return TrajectoryMiniBatch.from_partial_trajectories(
            [self.sample_trajectory(slicer, length) for _ in range(batch_size)]
        )

    def dump(self, f: BinaryIO) -> None:
        dump(self._episodes, f)

    @classmethod
    def from_episode_generator(
        cls, episode_generator: EpisodeGeneratorProtocol, buffer: BufferProtocol
    ) -> "ReplayBuffer":
        return cls(buffer, episode_generator())

    @classmethod
    def load(
        cls,
        f: BinaryIO,
        buffer: BufferProtocol,
        episode_cls: Type[EpisodeBase] = Episode,
    ) -> "ReplayBuffer":
        return cls(buffer, load(episode_cls, f))

    @property
    def episodes(self) -> Sequence[EpisodeBase]:
        return self._buffer.episodes

    def size(self) -> int:
        return len(self._buffer.episodes)


def create_fifo_replay_buffer(
    limit: int, episode_generator: Optional[EpisodeGeneratorProtocol] = None
) -> ReplayBuffer:
    buffer = FIFOBuffer(limit)
    episodes = episode_generator() if episode_generator else []
    return ReplayBuffer(buffer, episodes)


def create_infinite_replay_buffer(
    episode_generator: Optional[EpisodeGeneratorProtocol] = None,
) -> ReplayBuffer:
    buffer = InfiniteBuffer()
    episodes = episode_generator() if episode_generator else []
    return ReplayBuffer(buffer, episodes)

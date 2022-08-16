from typing import BinaryIO, List, Optional, Sequence, Type, Union

import numpy as np

from .buffers import BufferProtocol, FIFOBuffer, InfiniteBuffer
from .components import Episode, EpisodeBase, PartialTrajectory, Transition
from .episode_generator import EpisodeGeneratorProtocol
from .io import dump, load
from .mini_batch import TrajectoryMiniBatch, TransitionMiniBatch
from .trajectory_slicers import BasicTrajectorySlicer, TrajectorySlicerProtocol
from .transition_pickers import BasicTransitionPicker, TransitionPickerProtocol
from .types import Observation
from .writers import ExperienceWriter

__all__ = [
    "ReplayBuffer",
    "create_fifo_replay_buffer",
    "create_infinite_replay_buffer",
]


class ReplayBuffer:
    _buffer: BufferProtocol
    _transition_picker: TransitionPickerProtocol
    _trajectory_slicer: TrajectorySlicerProtocol
    _writer: ExperienceWriter
    _episodes: List[EpisodeBase]

    def __init__(
        self,
        buffer: BufferProtocol,
        transition_picker: Optional[TransitionPickerProtocol] = None,
        trajectory_slicer: Optional[TrajectorySlicerProtocol] = None,
        episodes: Optional[Sequence[EpisodeBase]] = None,
    ):
        self._buffer = buffer
        self._writer = ExperienceWriter(buffer)

        if transition_picker:
            self._transition_picker = transition_picker
        else:
            self._transition_picker = BasicTransitionPicker()

        if trajectory_slicer:
            self._trajectory_slicer = trajectory_slicer
        else:
            self._trajectory_slicer = BasicTrajectorySlicer()

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

    def sample_transition(self) -> Transition:
        episode_index = np.random.randint(len(self.episodes))
        episode = self.episodes[episode_index]
        transition_index = np.random.randint(episode.transition_count)
        return self._transition_picker(episode, transition_index)

    def sample_transition_batch(self, batch_size: int) -> TransitionMiniBatch:
        return TransitionMiniBatch.from_transitions(
            [self.sample_transition() for _ in range(batch_size)]
        )

    def sample_trajectory(self, length: int) -> PartialTrajectory:
        episode_index = np.random.randint(len(self.episodes))
        episode = self.episodes[episode_index]
        transition_index = np.random.randint(episode.size())
        return self._trajectory_slicer(episode, transition_index, length)

    def sample_trajectory_batch(
        self, batch_size: int, length: int
    ) -> TrajectoryMiniBatch:
        return TrajectoryMiniBatch.from_partial_trajectories(
            [self.sample_trajectory(length) for _ in range(batch_size)]
        )

    def dump(self, f: BinaryIO) -> None:
        dump(self._buffer.episodes, f)

    @classmethod
    def from_episode_generator(
        cls,
        episode_generator: EpisodeGeneratorProtocol,
        buffer: BufferProtocol,
        transition_picker: Optional[TransitionPickerProtocol] = None,
        trajectory_slicer: Optional[TrajectorySlicerProtocol] = None,
    ) -> "ReplayBuffer":
        return cls(
            buffer,
            episodes=episode_generator(),
            transition_picker=transition_picker,
            trajectory_slicer=trajectory_slicer,
        )

    @classmethod
    def load(
        cls,
        f: BinaryIO,
        buffer: BufferProtocol,
        episode_cls: Type[EpisodeBase] = Episode,
        transition_picker: Optional[TransitionPickerProtocol] = None,
        trajectory_slicer: Optional[TrajectorySlicerProtocol] = None,
    ) -> "ReplayBuffer":
        return cls(
            buffer,
            episodes=load(episode_cls, f),
            transition_picker=transition_picker,
            trajectory_slicer=trajectory_slicer,
        )

    @property
    def episodes(self) -> Sequence[EpisodeBase]:
        return self._buffer.episodes

    def size(self) -> int:
        return len(self._buffer.episodes)

    @property
    def buffer(self) -> BufferProtocol:
        return self._buffer

    @property
    def transition_count(self) -> int:
        return self._buffer.transition_count

    @property
    def transition_picker(self) -> TransitionPickerProtocol:
        return self._transition_picker

    @property
    def trajectory_slcier(self) -> TrajectorySlicerProtocol:
        return self._trajectory_slicer


def create_fifo_replay_buffer(
    limit: int,
    episode_generator: Optional[EpisodeGeneratorProtocol] = None,
    transition_picker: Optional[TransitionPickerProtocol] = None,
    trajectory_slicer: Optional[TrajectorySlicerProtocol] = None,
) -> ReplayBuffer:
    buffer = FIFOBuffer(limit)
    episodes = episode_generator() if episode_generator else []
    return ReplayBuffer(
        buffer,
        episodes=episodes,
        transition_picker=transition_picker,
        trajectory_slicer=trajectory_slicer,
    )


def create_infinite_replay_buffer(
    episode_generator: Optional[EpisodeGeneratorProtocol] = None,
    transition_picker: Optional[TransitionPickerProtocol] = None,
    trajectory_slicer: Optional[TrajectorySlicerProtocol] = None,
) -> ReplayBuffer:
    buffer = InfiniteBuffer()
    episodes = episode_generator() if episode_generator else []
    return ReplayBuffer(
        buffer,
        episodes=episodes,
        transition_picker=transition_picker,
        trajectory_slicer=trajectory_slicer,
    )

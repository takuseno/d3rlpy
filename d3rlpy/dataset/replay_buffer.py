from abc import ABC, abstractmethod
from typing import BinaryIO, List, Optional, Sequence, Type, Union

import numpy as np

from ..constants import ActionSpace
from ..logging import LOG
from ..types import GymEnv, NDArray, Observation
from .buffers import BufferProtocol, FIFOBuffer, InfiniteBuffer
from .components import (
    DatasetInfo,
    Episode,
    EpisodeBase,
    PartialTrajectory,
    Signature,
    Transition,
)
from .episode_generator import EpisodeGeneratorProtocol
from .io import dump, load
from .mini_batch import TrajectoryMiniBatch, TransitionMiniBatch
from .trajectory_slicers import BasicTrajectorySlicer, TrajectorySlicerProtocol
from .transition_pickers import BasicTransitionPicker, TransitionPickerProtocol
from .utils import (
    detect_action_size_from_env,
    detect_action_space,
    detect_action_space_from_env,
)
from .writers import (
    BasicWriterPreprocess,
    ExperienceWriter,
    WriterPreprocessProtocol,
)

__all__ = [
    "ReplayBufferBase",
    "ReplayBuffer",
    "MixedReplayBuffer",
    "create_fifo_replay_buffer",
    "create_infinite_replay_buffer",
]


class ReplayBufferBase(ABC):
    """An interface of ReplayBuffer."""

    @abstractmethod
    def append(
        self,
        observation: Observation,
        action: Union[int, NDArray],
        reward: Union[float, NDArray],
    ) -> None:
        r"""Appends observation, action and reward to buffer.

        Args:
            observation: Observation.
            action: Action.
            reward: Reward.
        """
        raise NotImplementedError

    @abstractmethod
    def append_episode(self, episode: EpisodeBase) -> None:
        r"""Appends episode to buffer.

        Args:
            episode: Episode.
        """
        raise NotImplementedError

    @abstractmethod
    def clip_episode(self, terminated: bool) -> None:
        r"""Clips current episode.

        Args:
            terminated: Flag to represent environmental termination. This flag
                should be ``False`` if the episode is terminated by timeout.
        """
        raise NotImplementedError

    @abstractmethod
    def sample_transition(self) -> Transition:
        r"""Samples a transition.

        Returns:
            Transition.
        """
        raise NotImplementedError

    @abstractmethod
    def sample_transition_batch(self, batch_size: int) -> TransitionMiniBatch:
        r"""Samples a mini-batch of transitions.

        Args:
            batch_size: Mini-batch size.

        Returns:
            Mini-batch.
        """
        raise NotImplementedError

    @abstractmethod
    def sample_trajectory(self, length: int) -> PartialTrajectory:
        r"""Samples a partial trajectory.

        Args:
            length: Length of partial trajectory.

        Returns:
            Partial trajectory.
        """
        raise NotImplementedError

    @abstractmethod
    def sample_trajectory_batch(
        self, batch_size: int, length: int
    ) -> TrajectoryMiniBatch:
        r"""Samples a mini-batch of partial trajectories.

        Args:
            batch_size: Mini-batch size.
            length: Length of partial trajectories.

        Returns:
            Mini-batch.
        """
        raise NotImplementedError

    @abstractmethod
    def dump(self, f: BinaryIO) -> None:
        """Dumps buffer data.

        .. code-block:: python

            with open('dataset.h5', 'w+b') as f:
                replay_buffer.dump(f)

        Args:
            f: IO object to write to.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_episode_generator(
        cls,
        episode_generator: EpisodeGeneratorProtocol,
        buffer: BufferProtocol,
        transition_picker: Optional[TransitionPickerProtocol] = None,
        trajectory_slicer: Optional[TrajectorySlicerProtocol] = None,
        writer_preprocessor: Optional[WriterPreprocessProtocol] = None,
    ) -> "ReplayBuffer":
        """Builds ReplayBuffer from episode generator.

        Args:
            episode_generator: Episode generator implementation.
            buffer: Buffer implementation.
            transition_picker: Transition picker implementation for
                Q-learning-based algorithms.
            trajectory_slicer: Trajectory slicer implementation for
                Transformer-based algorithms.
            writer_preprocessor: Writer preprocessor implementation.

        Returns:
            Replay buffer.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(
        cls,
        f: BinaryIO,
        buffer: BufferProtocol,
        episode_cls: Type[EpisodeBase] = Episode,
        transition_picker: Optional[TransitionPickerProtocol] = None,
        trajectory_slicer: Optional[TrajectorySlicerProtocol] = None,
        writer_preprocessor: Optional[WriterPreprocessProtocol] = None,
    ) -> "ReplayBuffer":
        """Builds ReplayBuffer from dumped data.

        This method reconstructs replay buffer dumped by ``dump`` method.

        .. code-block:: python

            with open('dataset.h5', 'rb') as f:
                replay_buffer = ReplayBuffer.load(f, buffer)

        Args:
            f: IO object to read from.
            buffer: Buffer implementation.
            episode_cls: Eisode class used to reconstruct data.
            transition_picker: Transition picker implementation for
                Q-learning-based algorithms.
            trajectory_slicer: Trajectory slicer implementation for
                Transformer-based algorithms.
            writer_preprocessor: Writer preprocessor implementation.

        Returns:
            Replay buffer.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def episodes(self) -> Sequence[EpisodeBase]:
        """Returns sequence of episodes.

        Returns:
            Sequence of episodes.
        """
        raise NotImplementedError

    @abstractmethod
    def size(self) -> int:
        """Returns number of episodes.

        Returns:
            Number of episodes.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def buffer(self) -> BufferProtocol:
        """Returns buffer.

        Returns:
            Buffer.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def transition_count(self) -> int:
        """Returns number of transitions.

        Returns:
            Number of transitions.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def transition_picker(self) -> TransitionPickerProtocol:
        """Returns transition picker.

        Returns:
            Transition picker.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def trajectory_slicer(self) -> TrajectorySlicerProtocol:
        """Returns trajectory slicer.

        Returns:
            Trajectory slicer.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def dataset_info(self) -> DatasetInfo:
        """Returns dataset information.

        Returns:
            Dataset information.
        """
        raise NotImplementedError


class ReplayBuffer(ReplayBufferBase):
    r"""Replay buffer for experience replay.

    This replay buffer implementation is used for both online and offline
    training in d3rlpy. To determine shapes of observations, actions and
    rewards, one of ``episodes``, ``env`` and signatures must be provided.

    .. code-block::

        from d3rlpy.dataset import FIFOBuffer, ReplayBuffer, Signature

        buffer = FIFOBuffer(limit=1000000)

        # initialize with pre-collected episodes
        replay_buffer = ReplayBuffer(buffer=buffer, episodes=<episodes>)

        # initialize with Gym
        replay_buffer = ReplayBuffer(buffer=buffer, env=<env>)

        # initialize with manually specified signatures
        replay_buffer = ReplayBuffer(
            buffer=buffer,
            observation_signature=Signature(dtype=[<dtype>], shape=[<shape>]),
            action_signature=Signature(dtype=[<dtype>], shape=[<shape>]),
            reward_signature=Signature(dtype=[<dtype>], shape=[<shape>]),
        )

    Args:
        buffer (d3rlpy.dataset.BufferProtocol): Buffer implementation.
        transition_picker (Optional[d3rlpy.dataset.TransitionPickerProtocol]):
            Transition picker implementation for Q-learning-based algorithms.
            If ``None`` is given, ``BasicTransitionPicker`` is used by default.
        trajectory_slicer (Optional[d3rlpy.dataset.TrajectorySlicerProtocol]):
            Trajectory slicer implementation for Transformer-based algorithms.
            If ``None`` is given, ``BasicTrajectorySlicer`` is used by default.
        writer_preprocessor (Optional[d3rlpy.dataset.WriterPreprocessProtocol]):
            Writer preprocessor implementation. If ``None`` is given,
            ``BasicWriterPreprocess`` is used by default.
        episodes (Optional[Sequence[d3rlpy.dataset.EpisodeBase]]):
            List of episodes to initialize replay buffer.
        env (Optional[GymEnv]): Gym environment to extract shapes of
            observations and action.
        observation_signature (Optional[d3rlpy.dataset.Signature]):
            Signature of observation.
        action_signature (Optional[d3rlpy.dataset.Signature]):
            Signature of action.
        reward_signature (Optional[d3rlpy.dataset.Signature]):
            Signature of reward.
        action_space (Optional[d3rlpy.constants.ActionSpace]):
            Action-space type.
        action_size (Optional[int]): Size of action-space. For continuous
            action-space, this represents dimension of action vectors. For
            discrete action-space, this represents the number of discrete
            actions.
        cache_size (int): Size of cache to record active episode history used
            for online training. ``cache_size`` needs to be greater than the
            maximum possible episode length.
    """

    _buffer: BufferProtocol
    _transition_picker: TransitionPickerProtocol
    _trajectory_slicer: TrajectorySlicerProtocol
    _writer: ExperienceWriter
    _episodes: List[EpisodeBase]
    _dataset_info: DatasetInfo

    def __init__(
        self,
        buffer: BufferProtocol,
        transition_picker: Optional[TransitionPickerProtocol] = None,
        trajectory_slicer: Optional[TrajectorySlicerProtocol] = None,
        writer_preprocessor: Optional[WriterPreprocessProtocol] = None,
        episodes: Optional[Sequence[EpisodeBase]] = None,
        env: Optional[GymEnv] = None,
        observation_signature: Optional[Signature] = None,
        action_signature: Optional[Signature] = None,
        reward_signature: Optional[Signature] = None,
        action_space: Optional[ActionSpace] = None,
        action_size: Optional[int] = None,
        cache_size: int = 10000,
    ):
        transition_picker = transition_picker or BasicTransitionPicker()
        trajectory_slicer = trajectory_slicer or BasicTrajectorySlicer()
        writer_preprocessor = writer_preprocessor or BasicWriterPreprocess()

        if not (
            observation_signature and action_signature and reward_signature
        ):
            if episodes:
                observation_signature = episodes[0].observation_signature
                action_signature = episodes[0].action_signature
                reward_signature = episodes[0].reward_signature
            elif env:
                observation_signature = Signature(
                    dtype=[env.observation_space.dtype],
                    shape=[env.observation_space.shape],  # type: ignore
                )
                action_signature = Signature(
                    dtype=[env.action_space.dtype],
                    shape=[env.action_space.shape],  # type: ignore
                )
                reward_signature = Signature(
                    dtype=[np.dtype(np.float32)],
                    shape=[[1]],
                )
            else:
                raise ValueError(
                    "Either episodes or env must be provided to determine signatures."
                    " Or specify signatures directly."
                )
            LOG.info(
                "Signatures have been automatically determined.",
                observation_signature=observation_signature,
                action_signature=action_signature,
                reward_signature=reward_signature,
            )

        if action_space is None:
            if episodes:
                action_space = detect_action_space(episodes[0].actions)
            elif env:
                action_space = detect_action_space_from_env(env)
            else:
                raise ValueError(
                    "Either episodes or env must be provided to determine action_space."
                    " Or specify action_space directly."
                )
            LOG.info(
                "Action-space has been automatically determined.",
                action_space=action_space,
            )

        if action_size is None:
            if episodes:
                if action_space == ActionSpace.CONTINUOUS:
                    action_size = action_signature.shape[0][0]
                else:
                    max_action = 0
                    for episode in episodes:
                        max_action = max(
                            int(np.max(episode.actions)), max_action
                        )
                    action_size = max_action + 1  # index should start from 0
            elif env:
                action_size = detect_action_size_from_env(env)
            else:
                raise ValueError(
                    "Either episodes or env must be provided to determine action_space."
                    " Or specify action_size directly."
                )
            LOG.info(
                "Action size has been automatically determined.",
                action_size=action_size,
            )

        self._buffer = buffer
        self._writer = ExperienceWriter(
            buffer,
            writer_preprocessor,
            observation_signature=observation_signature,
            action_signature=action_signature,
            reward_signature=reward_signature,
            cache_size=cache_size,
        )
        self._transition_picker = transition_picker
        self._trajectory_slicer = trajectory_slicer
        self._dataset_info = DatasetInfo(
            observation_signature=observation_signature,
            action_signature=action_signature,
            reward_signature=reward_signature,
            action_space=action_space,
            action_size=action_size,
        )

        if episodes:
            for episode in episodes:
                self.append_episode(episode)

    def append(
        self,
        observation: Observation,
        action: Union[int, NDArray],
        reward: Union[float, NDArray],
    ) -> None:
        self._writer.write(observation, action, reward)

    def append_episode(self, episode: EpisodeBase) -> None:
        for i in range(episode.transition_count):
            self._buffer.append(episode, i)

    def clip_episode(self, terminated: bool) -> None:
        self._writer.clip_episode(terminated)

    def sample_transition(self) -> Transition:
        index = np.random.randint(self._buffer.transition_count)
        episode, transition_index = self._buffer[index]
        return self._transition_picker(episode, transition_index)

    def sample_transition_batch(self, batch_size: int) -> TransitionMiniBatch:
        return TransitionMiniBatch.from_transitions(
            [self.sample_transition() for _ in range(batch_size)]
        )

    def sample_trajectory(self, length: int) -> PartialTrajectory:
        index = np.random.randint(self._buffer.transition_count)
        episode, transition_index = self._buffer[index]
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
        writer_preprocessor: Optional[WriterPreprocessProtocol] = None,
    ) -> "ReplayBuffer":
        return cls(
            buffer,
            episodes=episode_generator(),
            transition_picker=transition_picker,
            trajectory_slicer=trajectory_slicer,
            writer_preprocessor=writer_preprocessor,
        )

    @classmethod
    def load(
        cls,
        f: BinaryIO,
        buffer: BufferProtocol,
        episode_cls: Type[EpisodeBase] = Episode,
        transition_picker: Optional[TransitionPickerProtocol] = None,
        trajectory_slicer: Optional[TrajectorySlicerProtocol] = None,
        writer_preprocessor: Optional[WriterPreprocessProtocol] = None,
    ) -> "ReplayBuffer":
        return cls(
            buffer,
            episodes=load(episode_cls, f),
            transition_picker=transition_picker,
            trajectory_slicer=trajectory_slicer,
            writer_preprocessor=writer_preprocessor,
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
    def trajectory_slicer(self) -> TrajectorySlicerProtocol:
        return self._trajectory_slicer

    @property
    def dataset_info(self) -> DatasetInfo:
        return self._dataset_info


class MixedReplayBuffer(ReplayBufferBase):
    r"""A class combining two replay buffer instances.

    This replay buffer implementation combines two replay buffers
    (e.g. offline buffer and online buffer). The primary replay buffer is
    exposed to methods such as ``append``. Mini-batches are sampled from each
    replay buffer based on ``secondary_mix_ratio``.

    .. code-block::

        import d3rlpy

        # offline dataset
        dataset, env = d3rlpy.datasets.get_cartpole()

        # online replay buffer
        online_buffer = d3rlpy.dataset.create_fifo_replay_buffer(
            limit=100000,
            env=env,
        )

        # combine two replay buffers
        replay_buffer = d3rlpy.dataset.MixedReplayBuffer(
            primary_replay_buffer=online_buffer,
            secondary_replay_buffer=dataset,
            secondary_mix_ratio=0.5,
        )

    Args:
        primary_replay_buffer (d3rlpy.dataset.ReplayBufferBase):
            Primary replay buffer.
        secondary_replay_buffer (d3rlpy.dataset.ReplayBufferBase):
            Secondary replay buffer.
        secondary_mix_ratio (float): Ratio to sample mini-batches from the
            secondary replay buffer.
    """

    _primary_replay_buffer: ReplayBufferBase
    _secondary_replay_buffer: ReplayBufferBase
    _secondary_mix_ratio: float

    def __init__(
        self,
        primary_replay_buffer: ReplayBufferBase,
        secondary_replay_buffer: ReplayBufferBase,
        secondary_mix_ratio: float,
    ):
        assert 0.0 <= secondary_mix_ratio <= 1.0
        assert isinstance(
            primary_replay_buffer.transition_picker,
            type(secondary_replay_buffer.transition_picker),
        )
        assert isinstance(
            primary_replay_buffer.trajectory_slicer,
            type(secondary_replay_buffer.trajectory_slicer),
        )
        self._primary_replay_buffer = primary_replay_buffer
        self._secondary_replay_buffer = secondary_replay_buffer
        self._secondary_mix_ratio = secondary_mix_ratio

    def append(
        self,
        observation: Observation,
        action: Union[int, NDArray],
        reward: Union[float, NDArray],
    ) -> None:
        self._primary_replay_buffer.append(observation, action, reward)

    def append_episode(self, episode: EpisodeBase) -> None:
        self._primary_replay_buffer.append_episode(episode)

    def clip_episode(self, terminated: bool) -> None:
        self._primary_replay_buffer.clip_episode(terminated)

    def sample_transition(self) -> Transition:
        raise NotImplementedError(
            "MixedReplayBuffer does not support sample_transition."
        )

    def sample_transition_batch(self, batch_size: int) -> TransitionMiniBatch:
        primary_batch_size = int((1 - self._secondary_mix_ratio) * batch_size)
        secondary_batch_size = batch_size - primary_batch_size
        primary_batches = [
            self._primary_replay_buffer.sample_transition()
            for _ in range(primary_batch_size)
        ]
        secondary_batches = [
            self._secondary_replay_buffer.sample_transition()
            for _ in range(secondary_batch_size)
        ]
        return TransitionMiniBatch.from_transitions(
            primary_batches + secondary_batches
        )

    def sample_trajectory(self, length: int) -> PartialTrajectory:
        raise NotImplementedError(
            "MixedReplayBuffer does not support sample_trajectory."
        )

    def sample_trajectory_batch(
        self, batch_size: int, length: int
    ) -> TrajectoryMiniBatch:
        primary_batch_size = int((1 - self._secondary_mix_ratio) * batch_size)
        secondary_batch_size = batch_size - primary_batch_size
        primary_batches = [
            self._primary_replay_buffer.sample_trajectory(length)
            for _ in range(primary_batch_size)
        ]
        secondary_batches = [
            self._secondary_replay_buffer.sample_trajectory(length)
            for _ in range(secondary_batch_size)
        ]
        return TrajectoryMiniBatch.from_partial_trajectories(
            primary_batches + secondary_batches
        )

    def dump(self, f: BinaryIO) -> None:
        raise NotImplementedError("MixedReplayBuffer does not support dump.")

    @classmethod
    def from_episode_generator(
        cls,
        episode_generator: EpisodeGeneratorProtocol,
        buffer: BufferProtocol,
        transition_picker: Optional[TransitionPickerProtocol] = None,
        trajectory_slicer: Optional[TrajectorySlicerProtocol] = None,
        writer_preprocessor: Optional[WriterPreprocessProtocol] = None,
    ) -> "ReplayBuffer":
        raise NotImplementedError(
            "MixedReplayBuffer does not support from_episode_generator."
        )

    @classmethod
    def load(
        cls,
        f: BinaryIO,
        buffer: BufferProtocol,
        episode_cls: Type[EpisodeBase] = Episode,
        transition_picker: Optional[TransitionPickerProtocol] = None,
        trajectory_slicer: Optional[TrajectorySlicerProtocol] = None,
        writer_preprocessor: Optional[WriterPreprocessProtocol] = None,
    ) -> "ReplayBuffer":
        raise NotImplementedError("MixedReplayBuffer does not support load.")

    @property
    def episodes(self) -> Sequence[EpisodeBase]:
        return list(self._primary_replay_buffer.episodes) + list(
            self._secondary_replay_buffer.episodes
        )

    def size(self) -> int:
        return (
            self._primary_replay_buffer.size()
            + self._secondary_replay_buffer.size()
        )

    @property
    def buffer(self) -> BufferProtocol:
        return self._primary_replay_buffer.buffer

    @property
    def transition_count(self) -> int:
        return (
            self._primary_replay_buffer.transition_count
            + self._secondary_replay_buffer.transition_count
        )

    @property
    def transition_picker(self) -> TransitionPickerProtocol:
        return self._primary_replay_buffer.transition_picker

    @property
    def trajectory_slicer(self) -> TrajectorySlicerProtocol:
        return self._primary_replay_buffer.trajectory_slicer

    @property
    def dataset_info(self) -> DatasetInfo:
        return self._primary_replay_buffer.dataset_info

    @property
    def primary_replay_buffer(self) -> ReplayBufferBase:
        return self._primary_replay_buffer

    @property
    def secondary_replay_buffer(self) -> ReplayBufferBase:
        return self._secondary_replay_buffer


def create_fifo_replay_buffer(
    limit: int,
    episodes: Optional[Sequence[EpisodeBase]] = None,
    transition_picker: Optional[TransitionPickerProtocol] = None,
    trajectory_slicer: Optional[TrajectorySlicerProtocol] = None,
    writer_preprocessor: Optional[WriterPreprocessProtocol] = None,
    env: Optional[GymEnv] = None,
) -> ReplayBuffer:
    """Builds FIFO replay buffer.

    This function is a shortcut alias to build replay buffer with
    ``FIFOBuffer``.

    Args:
        limit: Maximum capacity of FIFO buffer.
        episodes: List of episodes to initialize replay buffer.
        transition_picker:
            Transition picker implementation for Q-learning-based algorithms.
            If ``None`` is given, ``BasicTransitionPicker`` is used by default.
        trajectory_slicer:
            Trajectory slicer implementation for Transformer-based algorithms.
            If ``None`` is given, ``BasicTrajectorySlicer`` is used by default.
        writer_preprocessor:
            Writer preprocessor implementation. If ``None`` is given,
            ``BasicWriterPreprocess`` is used by default.
        env: Gym environment to extract shapes of observations and action.

    Returns:
        Replay buffer.
    """
    buffer = FIFOBuffer(limit)
    return ReplayBuffer(
        buffer,
        episodes=episodes,
        transition_picker=transition_picker,
        trajectory_slicer=trajectory_slicer,
        writer_preprocessor=writer_preprocessor,
        env=env,
    )


def create_infinite_replay_buffer(
    episodes: Optional[Sequence[EpisodeBase]] = None,
    transition_picker: Optional[TransitionPickerProtocol] = None,
    trajectory_slicer: Optional[TrajectorySlicerProtocol] = None,
    writer_preprocessor: Optional[WriterPreprocessProtocol] = None,
    env: Optional[GymEnv] = None,
) -> ReplayBuffer:
    """Builds infinite replay buffer.

    This function is a shortcut alias to build replay buffer with
    ``InfiniteBuffer``.

    Args:
        episodes: List of episodes to initialize replay buffer.
        transition_picker:
            Transition picker implementation for Q-learning-based algorithms.
            If ``None`` is given, ``BasicTransitionPicker`` is used by default.
        trajectory_slicer:
            Trajectory slicer implementation for Transformer-based algorithms.
            If ``None`` is given, ``BasicTrajectorySlicer`` is used by default.
        writer_preprocessor:
            Writer preprocessor implementation. If ``None`` is given,
            ``BasicWriterPreprocess`` is used by default.
        env: Gym environment to extract shapes of observations and action.

    Returns:
        Replay buffer.
    """
    buffer = InfiniteBuffer()
    return ReplayBuffer(
        buffer,
        episodes=episodes,
        transition_picker=transition_picker,
        trajectory_slicer=trajectory_slicer,
        writer_preprocessor=writer_preprocessor,
        env=env,
    )

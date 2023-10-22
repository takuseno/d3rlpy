from typing import Optional

from ..constants import ActionSpace
from ..types import Float32NDArray, NDArray, ObservationSequence
from .buffers import InfiniteBuffer
from .episode_generator import EpisodeGenerator
from .replay_buffer import ReplayBuffer
from .trajectory_slicers import TrajectorySlicerProtocol
from .transition_pickers import TransitionPickerProtocol

__all__ = ["MDPDataset"]


class MDPDataset(ReplayBuffer):
    r"""Backward-compability class of MDPDataset.

    This is a wrapper class that has a backward-compatible constructor
    interface.

    Args:
        observations (ObservationSequence): Observations.
        actions (np.ndarray): Actions.
        rewards (np.ndarray): Rewards.
        terminals (np.ndarray): Environmental terminal flags.
        timeouts (np.ndarray): Timeouts.
        transition_picker (Optional[TransitionPickerProtocol]):
            Transition picker implementation for Q-learning-based algorithms.
            If ``None`` is given, ``BasicTransitionPicker`` is used by default.
        trajectory_slicer (Optional[TrajectorySlicerProtocol]):
            Trajectory slicer implementation for Transformer-based algorithms.
            If ``None`` is given, ``BasicTrajectorySlicer`` is used by default.
        action_space (Optional[d3rlpy.constants.ActionSpace]):
            Action-space type.
        action_size (Optional[int]): Size of action-space. For continuous
            action-space, this represents dimension of action vectors. For
            discrete action-space, this represents the number of discrete
            actions.
    """

    def __init__(
        self,
        observations: ObservationSequence,
        actions: NDArray,
        rewards: Float32NDArray,
        terminals: Float32NDArray,
        timeouts: Optional[Float32NDArray] = None,
        transition_picker: Optional[TransitionPickerProtocol] = None,
        trajectory_slicer: Optional[TrajectorySlicerProtocol] = None,
        action_space: Optional[ActionSpace] = None,
        action_size: Optional[int] = None,
    ):
        episode_generator = EpisodeGenerator(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
            timeouts=timeouts,
        )
        buffer = InfiniteBuffer()
        super().__init__(
            buffer,
            episodes=episode_generator(),
            transition_picker=transition_picker,
            trajectory_slicer=trajectory_slicer,
            action_space=action_space,
            action_size=action_size,
        )

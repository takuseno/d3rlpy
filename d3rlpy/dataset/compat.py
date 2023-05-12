from typing import Optional

import numpy as np

from .buffers import InfiniteBuffer
from .episode_generator import EpisodeGenerator
from .replay_buffer import ReplayBuffer
from .trajectory_slicers import TrajectorySlicerProtocol
from .transition_pickers import TransitionPickerProtocol
from .types import ObservationSequence

__all__ = ["MDPDataset"]


class MDPDataset(ReplayBuffer):
    def __init__(
        self,
        observations: ObservationSequence,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminals: np.ndarray,
        timeouts: Optional[np.ndarray] = None,
        transition_picker: Optional[TransitionPickerProtocol] = None,
        trajectory_slicer: Optional[TrajectorySlicerProtocol] = None,
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
        )

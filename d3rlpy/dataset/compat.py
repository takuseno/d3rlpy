from typing import Optional

import numpy as np

from .buffers import InfiniteBuffer
from .episode_generator import EpisodeGenerator
from .replay_buffer import ReplayBuffer
from .types import ObservationSequence

__all__ = ["MDPDataset"]


class MDPDataset(ReplayBuffer):
    def __init__(
        self,
        observations: ObservationSequence,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminals: np.ndarray,
        episode_terminals: Optional[np.ndarray] = None,
    ):
        episode_generator = EpisodeGenerator(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
            episode_terminals=episode_terminals,
        )
        buffer = InfiniteBuffer()
        super().__init__(buffer, episodes=episode_generator())

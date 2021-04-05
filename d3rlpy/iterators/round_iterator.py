from typing import List, cast

import numpy as np

from ..dataset import Episode, Transition
from .base import TransitionIterator


class RoundIterator(TransitionIterator):

    _shuffle: bool
    _indices: np.ndarray
    _index: int

    def __init__(
        self,
        episodes: List[Episode],
        batch_size: int,
        n_steps: int = 1,
        gamma: float = 0.99,
        n_frames: int = 1,
        shuffle: bool = True,
    ):
        super().__init__(episodes, batch_size, n_steps, gamma, n_frames)
        self._shuffle = shuffle
        self._indices = np.arange(self.size())
        self._index = 0

    def _reset(self) -> None:
        self._indices = np.arange(self.size())
        if self._shuffle:
            np.random.shuffle(self._indices)
        self._index = 0

    def _next(self) -> Transition:
        transition = self._transitions[cast(int, self._indices[self._index])]
        self._index += 1
        return transition

    def _has_finished(self) -> bool:
        return self._index >= self.size()

    def __len__(self) -> int:
        return len(self._transitions) // self._batch_size

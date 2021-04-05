from typing import List, cast

import numpy as np

from ..dataset import Episode, Transition
from .base import TransitionIterator


class RandomIterator(TransitionIterator):

    _n_steps_per_epoch: int
    _n_samples_per_epoch: int
    _index: int

    def __init__(
        self,
        episodes: List[Episode],
        n_steps_per_epoch: int,
        batch_size: int,
        n_steps: int = 1,
        gamma: float = 0.99,
        n_frames: int = 1,
    ):
        super().__init__(episodes, batch_size, n_steps, gamma, n_frames)
        self._n_steps_per_epoch = n_steps_per_epoch
        self._n_samples_per_epoch = batch_size * n_steps_per_epoch
        self._index = 0

    def _reset(self) -> None:
        self._index = 0

    def _next(self) -> Transition:
        index = cast(int, np.random.randint(self.size()))
        transition = self._transitions[index]
        self._index += 1
        return transition

    def _has_finished(self) -> bool:
        return self._index >= self._n_samples_per_epoch

    def __len__(self) -> int:
        return self._n_steps_per_epoch

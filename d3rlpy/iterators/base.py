from abc import ABCMeta, abstractmethod
from typing import Dict, Iterator, List

import numpy as np

from ..dataset import Episode, Transition, TransitionMiniBatch


class TransitionIterator(metaclass=ABCMeta):

    _episodes: List[Episode]
    _transitions: List[Transition]
    _orig_transitions: List[Transition]
    _ephemeral_transitions: List[Transition]
    _masks: Dict[Transition, np.ndarray]
    _batch_size: int
    _n_steps: int
    _gamma: float
    _n_frames: int
    _index: int

    def __init__(
        self,
        episodes: List[Episode],
        batch_size: int,
        n_steps: int = 1,
        gamma: float = 0.99,
        n_frames: int = 1,
    ):
        self._episodes = episodes
        self._orig_transitions = []
        self._ephemeral_transitions = []
        for episode in episodes:
            self._orig_transitions += episode.transitions
        self._transitions = self._orig_transitions
        self._batch_size = batch_size
        self._n_steps = n_steps
        self._gamma = gamma
        self._n_frames = n_frames

    def __iter__(self) -> Iterator[TransitionMiniBatch]:
        self.reset()
        return self

    def __next__(self) -> TransitionMiniBatch:
        transitions = [self.get_next() for _ in range(self._batch_size)]
        batch = TransitionMiniBatch(
            transitions,
            n_frames=self._n_frames,
            n_steps=self._n_steps,
            gamma=self._gamma,
        )
        return batch

    def reset(self) -> None:
        self._transitions = self._orig_transitions + self._ephemeral_transitions
        self._reset()

    @abstractmethod
    def _reset(self) -> None:
        pass

    @abstractmethod
    def _next(self) -> Transition:
        pass

    @abstractmethod
    def _has_finished(self) -> bool:
        pass

    def set_ephemeral_transitions(self, transitions: List[Transition]) -> None:
        self._ephemeral_transitions = transitions

    def get_next(self) -> Transition:
        if self._has_finished():
            raise StopIteration
        return self._next()

    @abstractmethod
    def __len__(self) -> int:
        pass

    def size(self) -> int:
        return len(self._transitions)

    @property
    def episodes(self) -> List[Episode]:
        return self._episodes

    @property
    def transitions(self) -> List[Transition]:
        return self._transitions

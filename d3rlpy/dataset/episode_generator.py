from typing import Optional, Sequence

import numpy as np
from typing_extensions import Protocol

from .components import Episode, EpisodeBase
from .types import ObservationSequence
from .utils import slice_observations

__all__ = ["EpisodeGeneratorProtocol", "EpisodeGenerator"]


class EpisodeGeneratorProtocol(Protocol):
    def __call__(self) -> Sequence[EpisodeBase]:
        ...


class EpisodeGenerator(EpisodeGeneratorProtocol):
    _observations: ObservationSequence
    _actions: np.ndarray
    _rewards: np.ndarray
    _terminals: np.ndarray
    _timeouts: np.ndarray

    def __init__(
        self,
        observations: ObservationSequence,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminals: np.ndarray,
        timeouts: Optional[np.ndarray] = None,
    ):
        if actions.ndim == 1:
            actions = np.reshape(actions, [-1, 1])

        if rewards.ndim == 1:
            rewards = np.reshape(rewards, [-1, 1])

        if terminals.ndim > 1:
            terminals = np.reshape(terminals, [-1])

        if timeouts is None:
            timeouts = np.zeros_like(terminals)

        assert (
            np.sum(np.logical_and(terminals, timeouts)) == 0
        ), "terminals and timeouts never become True at the same time"

        self._observations = observations
        self._actions = actions
        self._rewards = rewards
        self._terminals = terminals
        self._timeouts = timeouts

    def __call__(self) -> Sequence[Episode]:
        start = 0
        episodes = []
        for i in range(self._terminals.shape[0]):
            if self._terminals[i] or self._timeouts[i]:
                end = i + 1
                episode = Episode(
                    observations=slice_observations(
                        self._observations, start, end
                    ),
                    actions=self._actions[start:end],
                    rewards=self._rewards[start:end],
                    terminated=bool(self._terminals[i]),
                )
                episodes.append(episode)
                start = end
        return episodes

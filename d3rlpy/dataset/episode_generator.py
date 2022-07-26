from typing import Optional, Sequence

import numpy as np
from typing_extensions import Protocol

from .components import Episode, EpisodeBase, ObservationSequence
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
    _episode_terminals: np.ndarray

    def __init__(
        self,
        observations: ObservationSequence,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminals: np.ndarray,
        episode_terminals: Optional[np.ndarray] = None,
    ):
        if actions.ndim == 1:
            actions = np.reshape(actions, [-1, 1])

        if rewards.ndim == 1:
            rewards = np.reshape(rewards, [-1, 1])

        if terminals.ndim > 1:
            terminals = np.reshape(terminals, [-1])

        if episode_terminals is None:
            episode_terminals = terminals

        self._observations = observations
        self._actions = actions
        self._rewards = rewards
        self._terminals = terminals
        self._episode_terminals = episode_terminals

    def __call__(self) -> Sequence[Episode]:
        start = 0
        episodes = []
        for i, terminal in enumerate(self._episode_terminals):
            if terminal:
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

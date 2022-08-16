from typing import List, Sequence

from typing_extensions import Protocol

from .components import EpisodeBase


class BufferProtocol(Protocol):
    def append(self, episode: EpisodeBase) -> None:
        raise NotImplementedError

    @property
    def episodes(self) -> Sequence[EpisodeBase]:
        raise NotImplementedError

    @property
    def transition_count(self) -> int:
        raise NotImplementedError


class InfiniteBuffer(BufferProtocol):
    _episodes: List[EpisodeBase]
    _transition_count: int

    def __init__(self) -> None:
        self._episodes = []
        self._transition_count = 0

    def append(self, episode: EpisodeBase) -> None:
        self._episodes.append(episode)
        self._transition_count += episode.transition_count

    @property
    def episodes(self) -> Sequence[EpisodeBase]:
        return self._episodes

    def __len__(self) -> int:
        return len(self._episodes)

    @property
    def transition_count(self) -> int:
        return self._transition_count


class FIFOBuffer(BufferProtocol):
    _episodes: List[EpisodeBase]
    _limit: int
    _transition_count: int

    def __init__(self, limit: int):
        self._limit = limit
        self._episodes = []
        self._transition_count = 0

    def append(self, episode: EpisodeBase) -> None:
        self._transition_count += episode.transition_count
        self._episodes.append(episode)
        if self._transition_count > self._limit:
            dropped_episode = self._episodes.pop(0)
            self._transition_count -= dropped_episode.transition_count

    @property
    def episodes(self) -> Sequence[EpisodeBase]:
        return self._episodes

    def __len__(self) -> int:
        return len(self._episodes)

    @property
    def transition_count(self) -> int:
        return self._transition_count

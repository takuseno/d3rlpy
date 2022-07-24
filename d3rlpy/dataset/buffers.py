from typing import List, Sequence

from typing_extensions import Protocol

from .components import EpisodeBase


class BufferProtocol(Protocol):
    def append(self, episode: EpisodeBase) -> None:
        raise NotImplementedError

    @property
    def episodes(self) -> Sequence[EpisodeBase]:
        raise NotImplementedError


class InfiniteBuffer(BufferProtocol):
    _episodes: List[EpisodeBase]

    def __init__(self):
        self._episodes = []

    def append(self, episode: EpisodeBase) -> None:
        self._episodes.append(episode)

    @property
    def episodes(self) -> Sequence[EpisodeBase]:
        return self._episodes


class FIFOBuffer(BufferProtocol):
    _episodes: List[EpisodeBase]
    _limit: int
    _total_count: int

    def __init__(self, limit: int):
        self._limit = limit
        self._episodes = []
        self._total_count = 0

    def append(self, episode: EpisodeBase) -> None:
        self._total_count += episode.size()
        self._episodes.append(episode)
        if self._total_count > self._limit:
            dropped_episode = self._episodes.pop(0)
            self._total_count -= dropped_episode.size()

    @property
    def episodes(self) -> Sequence[EpisodeBase]:
        return self._episodes

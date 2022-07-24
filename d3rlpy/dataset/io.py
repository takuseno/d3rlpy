from typing import Any, BinaryIO, Dict, Sequence, Type, cast

import h5py

from .components import Episode, EpisodeBase
from .episode_generator import EpisodeGenerator

__all__ = ["dump", "load", "load_v1"]


def dump(episodes: Sequence[EpisodeBase], f: BinaryIO) -> None:
    data = [episode.serialize() for episode in episodes]
    with h5py.File(f, "w") as h5:
        h5.create_dataset("data", data=data)
        h5.create_dataset("version", data="2.0")
        h5.flush()


def load(episode_cls: Type[EpisodeBase], f: BinaryIO) -> Sequence[EpisodeBase]:
    with h5py.File(f, "r") as h5:
        data = cast(Sequence[Dict[str, Any]], h5["data"][()])  # type: ignore
        episodes = [episode_cls.deserialize(v) for v in data]
    return episodes


def load_v1(f: BinaryIO) -> Sequence[Episode]:
    with h5py.File(f, "r") as h5:
        observations = h5["observations"][()]  # type: ignore
        actions = h5["actions"][()]  # type: ignore
        rewards = h5["rewards"][()]  # type: ignore
        terminals = h5["terminals"][()]  # type: ignore

        if "episode_terminals" in h5:
            episode_terminals = h5["episode_terminals"][()]  # type: ignore
        else:
            episode_terminals = None

    episode_generator = EpisodeGenerator(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        episode_terminals=episode_terminals,
    )

    return episode_generator()

from typing import BinaryIO, Sequence, Type, cast

import h5py
import numpy as np

from .components import Episode, EpisodeBase
from .episode_generator import EpisodeGenerator

__all__ = ["dump", "load", "load_v1", "DATASET_VERSION"]


DATASET_VERSION = "2.1"


def dump(episodes: Sequence[EpisodeBase], f: BinaryIO) -> None:
    r"""Writes episode data to file-like object.

    Args:
        episodes: Sequence of episodes.
        f: Binary file-like object.
    """
    with h5py.File(f, "w") as h5:
        keys = list(episodes[0].serialize().keys())
        h5.create_dataset("columns", data=keys)
        h5.create_dataset("num_episodes", data=len(episodes))
        for i, episode in enumerate(episodes):
            serializedData = episode.serialize()
            for key in keys:
                if isinstance(serializedData[key], (list, tuple)):
                    for j in range(len(serializedData[key])):
                        elm = serializedData[key][j]
                        h5.create_dataset(f"{key}_{i}_{j}", data=elm)
                else:
                    h5.create_dataset(f"{key}_{i}", data=serializedData[key])
        h5.create_dataset("version", data=DATASET_VERSION)
        h5.flush()


def load(episode_cls: Type[EpisodeBase], f: BinaryIO) -> Sequence[EpisodeBase]:
    r"""Constructs episodes from file-like object.

    Args:
        episode_cls: Episode class.
        f: Binary file-like object.

    Returns:
        Sequence of episodes.
    """
    episodes = []
    with h5py.File(f, "r") as h5:
        version = cast(str, h5["version"][()])
        if version == "2.0":
            raise ValueError("version=2.0 has been obsolete.")
        keys = cast(
            Sequence[str],
            list(map(lambda s: s.decode("utf-8"), h5["columns"][()])),
        )
        num_episodes = cast(int, h5["num_episodes"][()])
        for i in range(num_episodes):
            data = {}
            for key in keys:
                path = f"{key}_{i}"
                if path in h5:
                    data[key] = h5[path][()]
                else:
                    j = 0
                    tuple_data = []
                    while True:
                        tuple_path = f"{key}_{i}_{j}"
                        if tuple_path in h5:
                            tuple_data.append(h5[tuple_path][()])
                        else:
                            break
                        j += 1
                    data[key] = tuple_data
            episode = episode_cls.deserialize(data)
            episodes.append(episode)
    return episodes


def load_v1(f: BinaryIO) -> Sequence[Episode]:
    r"""Loads v1 dataset data.

    Args:
        f: Binary file-like object.

    Returns:
        Sequence of episodes.
    """
    with h5py.File(f, "r") as h5:
        observations = h5["observations"][()]
        actions = h5["actions"][()]
        rewards = h5["rewards"][()]
        terminals = h5["terminals"][()]

        if "episode_terminals" in h5:
            episode_terminals = h5["episode_terminals"][()]
        else:
            episode_terminals = None

    if episode_terminals is None:
        timeouts = None
    else:
        timeouts = np.logical_and(np.logical_not(terminals), episode_terminals)

    episode_generator = EpisodeGenerator(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        timeouts=timeouts,
    )

    return episode_generator()

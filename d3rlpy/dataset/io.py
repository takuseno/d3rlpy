from typing import BinaryIO, Sequence, Type, cast

import h5py

from .components import Episode, EpisodeBase
from .episode_generator import EpisodeGenerator

__all__ = ["dump", "load", "load_v1", "DATASET_VERSION"]


DATASET_VERSION = "2.0"


def dump(episodes: Sequence[EpisodeBase], f: BinaryIO) -> None:
    serializedData = [episode.serialize() for episode in episodes]
    keys = list(serializedData[0].keys())
    with h5py.File(f, "w") as h5:
        h5.create_dataset("columns", data=keys)
        h5.create_dataset("total_size", data=len(episodes))
        for key in keys:
            if isinstance(serializedData[0][key], (list, tuple)):
                for i in range(len(serializedData[0][key])):
                    data = [d[key][i] for d in serializedData]
                    h5.create_dataset(f"{key}_{i}", data=data)
            else:
                data = [d[key] for d in serializedData]
                h5.create_dataset(key, data=data)
        h5.create_dataset("version", data=DATASET_VERSION)
        h5.flush()


def load(episode_cls: Type[EpisodeBase], f: BinaryIO) -> Sequence[EpisodeBase]:
    episodes = []
    with h5py.File(f, "r") as h5:
        keys = cast(
            Sequence[str], map(lambda s: s.decode("utf-8"), h5["columns"][()])
        )
        total_size = cast(int, h5["total_size"][()])
        for i in range(total_size):
            data = {}
            for key in keys:
                if key in h5:
                    data[key] = h5[key][()][i]
                else:
                    j = 0
                    tuple_data = []
                    while True:
                        tuple_key = f"{key}_{j}"
                        if tuple_key in h5:
                            tuple_data.append(h5[tuple_key][()][i])
                        else:
                            break
                        j += 1
                    data[key] = tuple_data
            episode = episode_cls.deserialize(data)
            episodes.append(episode)
    return episodes


def load_v1(f: BinaryIO) -> Sequence[Episode]:
    with h5py.File(f, "r") as h5:
        observations = h5["observations"][()]
        actions = h5["actions"][()]
        rewards = h5["rewards"][()]
        terminals = h5["terminals"][()]

        if "episode_terminals" in h5:
            episode_terminals = h5["episode_terminals"][()]
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

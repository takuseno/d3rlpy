import os

import numpy as np
import pytest

from d3rlpy.dataset import Episode, Shape, dump, load

from ..testing_utils import create_episode


@pytest.mark.parametrize("observation_shape", [(4,), ((4,), (8,))])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("length", [100])
def test_dump_and_load(
    observation_shape: Shape, action_size: int, length: int
) -> None:
    episode = create_episode(observation_shape, action_size, length)

    path = os.path.join("test_data", "data.h5")

    # dump
    with open(path, "w+b") as f:
        dump([episode], f)

    # load
    with open(path, "rb") as f:
        loaded_episode = load(Episode, f)[0]

    if isinstance(observation_shape[0], tuple):
        for i in range(len(observation_shape)):
            assert np.all(
                episode.observations[i] == loaded_episode.observations[i]
            )
    else:
        assert np.all(episode.observations == loaded_episode.observations)
    assert np.all(episode.actions == loaded_episode.actions)
    assert np.all(episode.rewards == loaded_episode.rewards)
    assert np.all(episode.terminated == loaded_episode.terminated)

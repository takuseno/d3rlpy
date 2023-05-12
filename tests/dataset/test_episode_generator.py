import numpy as np
import pytest

from d3rlpy.dataset import EpisodeGenerator

from ..testing_utils import create_observations


@pytest.mark.parametrize("observation_shape", [(4,), ((4,), (8,))])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("length", [1000])
@pytest.mark.parametrize("terminal", [False, True])
def test_episode_generator(observation_shape, action_size, length, terminal):
    observations = create_observations(observation_shape, length)
    actions = np.random.random((length, action_size))
    rewards = np.random.random((length, 1))
    terminals = np.zeros(length)
    timeouts = np.zeros(length)
    for i in range(length // 100):
        if terminal:
            terminals[(i + 1) * 100 - 1] = 1.0
        else:
            timeouts[(i + 1) * 100 - 1] = 1.0

    episode_generator = EpisodeGenerator(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        timeouts=timeouts,
    )

    episodes = episode_generator()
    assert len(episodes) == length // 100

    for episode in episodes:
        assert len(episode) == 100
        if isinstance(observation_shape[0], tuple):
            for i, shape in enumerate(observation_shape):
                assert episode.observations[i].shape == (100,) + shape
        else:
            assert episode.observations.shape == (100,) + observation_shape
        assert episode.actions.shape == (100, action_size)
        assert episode.rewards.shape == (100, 1)
        assert episode.terminated == terminal

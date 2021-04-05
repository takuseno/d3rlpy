import numpy as np
import pytest

from d3rlpy.dataset import Episode
from d3rlpy.iterators.random_iterator import RandomIterator


@pytest.mark.parametrize("episode_size", [100])
@pytest.mark.parametrize("n_steps_per_epoch", [10])
@pytest.mark.parametrize("n_episodes", [2])
@pytest.mark.parametrize("observation_size", [10])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("set_ephemeral", [False, True])
def test_random_iterator(
    episode_size,
    n_steps_per_epoch,
    n_episodes,
    observation_size,
    action_size,
    batch_size,
    set_ephemeral,
):
    episodes = []
    for _ in range(n_episodes):
        observations = np.random.random((episode_size, observation_size))
        actions = np.random.random((episode_size, action_size))
        rewards = np.random.random(episode_size)
        episode = Episode(
            (observation_size,), action_size, observations, actions, rewards
        )
        episodes.append(episode)

    iterator = RandomIterator(episodes, n_steps_per_epoch, batch_size)

    if set_ephemeral:
        iterator.set_ephemeral_transitions(episodes[0].transitions)

    count = 0
    for batch in iterator:
        assert batch.observations.shape == (batch_size, observation_size)
        assert batch.actions.shape == (batch_size, action_size)
        assert batch.rewards.shape == (batch_size, 1)
        count += 1

    if set_ephemeral:
        assert count == n_steps_per_epoch
        assert len(iterator) == n_steps_per_epoch
    else:
        assert count == n_steps_per_epoch
        assert len(iterator) == n_steps_per_epoch

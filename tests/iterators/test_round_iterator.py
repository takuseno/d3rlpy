import pytest
import numpy as np

from d3rlpy.dataset import Episode
from d3rlpy.iterators.round_iterator import RoundIterator


@pytest.mark.parametrize("episode_size", [100])
@pytest.mark.parametrize("n_episodes", [2])
@pytest.mark.parametrize("observation_size", [10])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("n_critics", [2])
@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("set_ephemeral", [False, True])
def test_round_iterator(
    episode_size,
    n_episodes,
    observation_size,
    action_size,
    batch_size,
    n_critics,
    shuffle,
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

    iterator = RoundIterator(
        episodes, batch_size, n_critics=n_critics, shuffle=shuffle
    )

    if set_ephemeral:
        iterator.set_ephemeral_transitions(episodes[0].transitions)

    count = 0
    for batch in iterator:
        assert batch.observations.shape == (batch_size, observation_size)
        assert batch.actions.shape == (batch_size, action_size)
        assert batch.rewards.shape == (batch_size, 1)
        mask = batch.get_additional_data("mask")
        assert mask.shape == (batch_size, n_critics)
        count += 1

    if set_ephemeral:
        assert count == episode_size * (n_episodes + 1) // batch_size
        assert len(iterator) == episode_size * (n_episodes + 1) // batch_size
    else:
        assert count == episode_size * n_episodes // batch_size
        assert len(iterator) == episode_size * n_episodes // batch_size

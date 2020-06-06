import numpy as np
import pytest

from skbrl.dataset import _compute_rewards, _load_images, read_csv
from skbrl.dataset import MDPDataset, Episode, Transition


@pytest.mark.parametrize('data_size', [100])
@pytest.mark.parametrize('observation_size', [4])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('n_episodes', [4])
def test_compute_rewards(data_size, observation_size, action_size, n_episodes):
    observations = np.random.random((data_size, observation_size))
    actions = np.random.random((data_size, action_size))
    n_steps = data_size // n_episodes
    terminals = np.array(([0] * (n_steps - 1) + [1]) * n_episodes)

    def reward_func(obs_tm1, obs_t, act_t, ter_t):
        if ter_t:
            return 100.0
        return (obs_tm1 + obs_t).sum() + act_t.sum()

    # calcualate base rewards
    ref_rewards = (observations[1:] + observations[:-1]).sum(axis=1)
    ref_rewards += actions[1:].sum(axis=1)
    # append 0.0 as the initial step
    ref_rewards = np.hstack([[0.0], ref_rewards])
    # set terminal rewards
    ref_rewards[terminals == 1.0] = 100.0
    # set 0.0 to the first steps
    ref_rewards[1:][terminals[:-1] == 1.0] = 0.0

    rewards = _compute_rewards(reward_func, observations, actions, terminals)

    assert np.all(rewards == ref_rewards)


@pytest.mark.parametrize('data_size', [100])
@pytest.mark.parametrize('observation_size', [4])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('n_episodes', [4])
@pytest.mark.parametrize('discrete_action', [True, False])
def test_mdp_dataset(data_size, observation_size, action_size, n_episodes,
                     discrete_action):
    observations = np.random.random((data_size, observation_size))
    rewards = np.random.random(data_size)
    n_steps = data_size // n_episodes
    terminals = np.array(([0] * (n_steps - 1) + [1]) * n_episodes)

    if discrete_action:
        actions = np.random.randint(10, size=data_size)
        ref_action_size = np.max(actions) + 1
    else:
        actions = np.random.random((data_size, action_size))
        ref_action_size = action_size

    dataset = MDPDataset(
        observations, actions, rewards, terminals, discrete_action)

    # check MDPDataset methods
    assert np.all(dataset.observations == observations)
    assert np.all(dataset.actions == actions)
    assert np.all(dataset.rewards == rewards)
    assert np.all(dataset.terminals == terminals)
    assert dataset.size() == data_size
    assert dataset.get_action_size() == ref_action_size
    assert dataset.get_observation_shape() == (observation_size,)
    assert dataset.is_action_discrete() == discrete_action

    # check episodes exported from dataset
    episodes = dataset.to_episodes()
    assert len(episodes) == n_episodes
    for i, episode in enumerate(episodes):
        assert isinstance(episode, Episode)
        assert episode.size() == n_steps
        head = i * n_steps
        tail = head + n_steps
        assert np.all(episode.observations == observations[head:tail])
        assert np.all(episode.actions == actions[head:tail])
        assert np.all(episode.rewards == rewards[head:tail])
        assert episode.get_observation_shape() == (observation_size,)
        assert episode.get_action_size() == ref_action_size


@pytest.mark.parametrize('data_size', [100])
@pytest.mark.parametrize('observation_size', [4])
@pytest.mark.parametrize('action_size', [2])
def test_episode(data_size, observation_size, action_size):
    observations = np.random.random((data_size, observation_size))
    actions = np.random.random((data_size, action_size))
    rewards = np.random.random((data_size, 1))

    episode = Episode((observation_size,), action_size, observations, actions,
                      rewards)

    # check Episode methods
    assert np.all(episode.observations == observations)
    assert np.all(episode.actions == actions)
    assert np.all(episode.rewards == rewards)
    assert episode.size() == data_size
    assert episode.get_observation_shape() == (observation_size,)
    assert episode.get_action_size() == action_size

    # check transitions exported from episode
    transitions = episode.to_transitions()
    assert len(transitions) == data_size - 1
    for i, transition in enumerate(transitions):
        assert isinstance(transition, Transition)
        assert transition.observation_shape == (observation_size,)
        assert transition.action_size == action_size
        assert np.all(transition.obs_t == observations[i])
        assert np.all(transition.act_t == actions[i])
        assert transition.rew_t == rewards[i]
        assert np.all(transition.obs_tp1 == observations[i + 1])
        assert np.all(transition.act_tp1 == actions[i + 1])
        assert transition.rew_tp1 == rewards[i + 1]
        assert transition.ter_tp1 == (1.0 if (i == data_size - 2) else 0.0)

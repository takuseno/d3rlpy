import numpy as np
import pytest

from sklearn.model_selection import train_test_split
from skbrl.dataset import _compute_rewards, _load_images, read_csv
from skbrl.dataset import MDPDataset, Episode, Transition, TransitionMiniBatch


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

    dataset = MDPDataset(observations, actions, rewards, terminals,
                         discrete_action)

    # check MDPDataset methods
    assert np.all(dataset.observations == observations)
    assert np.all(dataset.actions == actions)
    assert np.all(dataset.rewards == rewards)
    assert np.all(dataset.terminals == terminals)
    assert dataset.size() == n_episodes
    assert dataset.get_action_size() == ref_action_size
    assert dataset.get_observation_shape() == (observation_size, )
    assert dataset.is_action_discrete() == discrete_action

    # check stats
    ref_returns = []
    for i in range(n_episodes):
        episode_return = 0.0
        for j in range(1, n_steps):
            episode_return += rewards[j + i * n_steps]
        ref_returns.append(episode_return)

    stats = dataset.compute_stats()
    return_stats = stats['return']
    assert np.allclose(return_stats['mean'], np.mean(ref_returns))
    assert np.allclose(return_stats['std'], np.std(ref_returns))
    assert np.allclose(return_stats['min'], np.min(ref_returns))
    assert np.allclose(return_stats['max'], np.max(ref_returns))
    observation_stats = stats['observation']
    assert np.all(observation_stats['mean'] == np.mean(observations, axis=0))
    assert np.all(observation_stats['std'] == np.std(observations, axis=0))

    # check episodes exported from dataset
    episodes = dataset.episodes
    assert len(episodes) == n_episodes
    for i, e in enumerate(dataset.episodes):
        assert isinstance(e, Episode)
        assert e.size() == n_steps - 1
        head = i * n_steps
        tail = head + n_steps
        assert np.all(e.observations == observations[head:tail])
        assert np.all(e.actions == actions[head:tail])
        assert np.all(e.rewards == rewards[head:tail])
        assert e.get_observation_shape() == (observation_size, )
        assert e.get_action_size() == ref_action_size

    # check list-like behaviors
    assert len(dataset) == n_episodes
    assert dataset[0] is dataset.episodes[0]
    for i, episode in enumerate(dataset.episodes):
        assert isinstance(episode, Episode)
        assert episode is dataset.episodes[i]


@pytest.mark.parametrize('data_size', [100])
@pytest.mark.parametrize('observation_size', [4])
@pytest.mark.parametrize('action_size', [2])
def test_episode(data_size, observation_size, action_size):
    observations = np.random.random((data_size, observation_size))
    actions = np.random.random((data_size, action_size))
    rewards = np.random.random((data_size, 1))

    episode = Episode((observation_size, ), action_size, observations, actions,
                      rewards)

    # check Episode methods
    assert np.all(episode.observations == observations)
    assert np.all(episode.actions == actions)
    assert np.all(episode.rewards == rewards)
    assert episode.size() == data_size - 1
    assert episode.get_observation_shape() == (observation_size, )
    assert episode.get_action_size() == action_size
    assert episode.compute_return() == np.sum(rewards[1:])

    # check transitions exported from episode
    assert len(episode.transitions) == data_size - 1
    for i, t in enumerate(episode.transitions):
        assert isinstance(t, Transition)
        assert t.observation_shape == (observation_size, )
        assert t.action_size == action_size
        assert np.all(t.observation == observations[i])
        assert np.all(t.action == actions[i])
        assert t.reward == rewards[i]
        assert np.all(t.next_observation == observations[i + 1])
        assert np.all(t.next_action == actions[i + 1])
        assert t.next_reward == rewards[i + 1]
        assert t.terminal == (1.0 if (i == data_size - 2) else 0.0)

    # check list-like bahaviors
    assert len(episode) == data_size - 1
    assert episode[0] is episode.transitions[0]
    for i, transition in enumerate(episode):
        assert isinstance(transition, Transition)
        assert transition is episode.transitions[i]


@pytest.mark.parametrize('data_size', [100])
@pytest.mark.parametrize('observation_size', [4])
@pytest.mark.parametrize('action_size', [2])
def test_transition_minibatch(data_size, observation_size, action_size):
    observations = np.random.random((data_size, observation_size))
    actions = np.random.random((data_size, action_size))
    rewards = np.random.random((data_size, 1))

    episode = Episode((observation_size, ), action_size, observations, actions,
                      rewards)

    batch = TransitionMiniBatch(episode.transitions)
    for i, t in enumerate(episode.transitions):
        assert np.all(batch.observations[i] == t.observation)
        assert np.all(batch.actions[i] == t.action)
        assert np.all(batch.rewards[i] == t.reward)
        assert np.all(batch.next_observations[i] == t.next_observation)
        assert np.all(batch.next_actions[i] == t.next_action)
        assert np.all(batch.next_rewards[i] == t.next_reward)
        assert np.all(batch.terminals[i] == t.terminal)

    # check list-like behavior
    assert len(batch) == data_size - 1
    assert batch[0] is episode.transitions[0]
    for i, transition in enumerate(batch):
        assert isinstance(transition, Transition)
        assert transition is episode.transitions[i]


@pytest.mark.parametrize('data_size', [100])
@pytest.mark.parametrize('observation_size', [4])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('n_episodes', [10])
@pytest.mark.parametrize('test_size', [0.2])
def test_dataset_with_sklearn(data_size, observation_size, action_size,
                              n_episodes, test_size):
    observations = np.random.random((data_size, observation_size))
    actions = np.random.random((data_size, action_size))
    rewards = np.random.random(data_size)
    n_steps = data_size // n_episodes
    terminals = np.array(([0] * (n_steps - 1) + [1]) * n_episodes)

    dataset = MDPDataset(observations, actions, rewards, terminals)

    # check compatibility with train_test_split
    train_episodes, test_episodes = train_test_split(dataset,
                                                     test_size=test_size)
    assert len(train_episodes) == int(n_episodes * (1.0 - test_size))
    assert len(test_episodes) == int(n_episodes * test_size)

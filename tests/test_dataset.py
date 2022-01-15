import os
import warnings
from collections import deque

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from d3rlpy.dataset import (
    Episode,
    MDPDataset,
    Transition,
    TransitionMiniBatch,
    _check_discrete_action,
    compute_lambda_return,
)


@pytest.mark.parametrize("data_size", [100])
def test_check_discrete_action(data_size):
    # discrete action with int32
    discrete_actions = np.random.randint(100, size=data_size)
    assert _check_discrete_action(discrete_actions)

    # discrete action with float32
    assert _check_discrete_action(np.array(discrete_actions, dtype=np.float32))

    # continuous action
    continuous_actions = np.random.random(data_size)
    assert not _check_discrete_action(continuous_actions)


@pytest.mark.parametrize("data_size", [100])
@pytest.mark.parametrize("observation_size", [4])
@pytest.mark.parametrize("action_size", [2])
def test_check_discrete_action_with_mdp_dataset(
    data_size, observation_size, action_size
):
    observations = np.random.random((data_size, observation_size)).astype("f4")
    rewards = np.random.random(data_size)
    terminals = np.random.randint(2, size=data_size)

    # check discrete_action
    discrete_actions = np.random.randint(action_size, size=data_size)
    dataset = MDPDataset(observations, discrete_actions, rewards, terminals)
    assert dataset.is_action_discrete()

    # check continuous action
    continuous_actions = np.random.random((data_size, action_size))
    dataset = MDPDataset(observations, continuous_actions, rewards, terminals)
    assert not dataset.is_action_discrete()


@pytest.mark.parametrize("data_size", [100])
@pytest.mark.parametrize("observation_size", [4])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("n_episodes", [4])
@pytest.mark.parametrize("discrete_action", [True, False])
@pytest.mark.parametrize("add_actions", [1, 3])
def test_mdp_dataset(
    data_size,
    observation_size,
    action_size,
    n_episodes,
    discrete_action,
    add_actions,
):
    observations = np.random.random((data_size, observation_size)).astype("f4")
    rewards = np.random.uniform(-10.0, 10.0, size=data_size).astype("f4")
    n_steps = data_size // n_episodes
    terminals = np.array(([0] * (n_steps - 1) + [1]) * n_episodes)

    if discrete_action:
        actions = np.random.randint(action_size, size=data_size)
        ref_action_size = np.max(actions) + 1
    else:
        actions = np.random.random((data_size, action_size)).astype("f4")
        ref_action_size = action_size

    dataset = MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        discrete_action=discrete_action,
    )

    # check MDPDataset methods
    assert np.all(dataset.observations == observations)
    assert np.all(dataset.actions == actions)
    assert np.all(dataset.rewards == rewards)
    assert np.all(dataset.terminals == terminals)
    assert dataset.size() == n_episodes
    assert dataset.get_action_size() == action_size
    assert dataset.get_observation_shape() == (observation_size,)
    assert dataset.is_action_discrete() == discrete_action

    # check stats
    ref_returns = []
    for i in range(n_episodes):
        episode_return = 0.0
        for j in range(n_steps):
            episode_return += rewards[j + i * n_steps]
        ref_returns.append(episode_return)

    stats = dataset.compute_stats()
    return_stats = stats["return"]
    assert np.allclose(return_stats["mean"], np.mean(ref_returns))
    assert np.allclose(return_stats["std"], np.std(ref_returns))
    assert np.allclose(return_stats["min"], np.min(ref_returns))
    assert np.allclose(return_stats["max"], np.max(ref_returns))
    reward_stats = stats["reward"]
    assert np.allclose(reward_stats["mean"], np.mean(rewards))
    assert np.allclose(reward_stats["std"], np.std(rewards))
    assert np.allclose(reward_stats["min"], np.min(rewards))
    assert np.allclose(reward_stats["max"], np.max(rewards))
    observation_stats = stats["observation"]
    assert np.all(observation_stats["mean"] == np.mean(observations, axis=0))
    assert np.all(observation_stats["std"] == np.std(observations, axis=0))
    if discrete_action:
        freqs, action_ids = stats["action"]["histogram"]
        assert np.sum(freqs) == data_size
        assert list(action_ids) == [i for i in range(action_size)]
    else:
        action_stats = stats["action"]
        assert np.all(action_stats["mean"] == np.mean(actions, axis=0))
        assert np.all(action_stats["std"] == np.std(actions, axis=0))
        assert np.all(action_stats["min"] == np.min(actions, axis=0))
        assert np.all(action_stats["max"] == np.max(actions, axis=0))
        assert len(action_stats["histogram"]) == action_size
        for freqs, _ in action_stats["histogram"]:
            assert np.sum(freqs) == data_size

    # check episodes exported from dataset
    episodes = dataset.episodes
    assert len(episodes) == n_episodes
    for i, e in enumerate(dataset.episodes):
        assert isinstance(e, Episode)
        assert e.size() == n_steps
        head = i * n_steps
        tail = head + n_steps
        assert np.all(e.observations == observations[head:tail])
        assert np.all(e.actions == actions[head:tail])
        assert np.all(e.rewards == rewards[head:tail])
        assert e.get_observation_shape() == (observation_size,)
        assert e.get_action_size() == ref_action_size

    # check list-like behaviors
    assert len(dataset) == n_episodes
    assert dataset[0] is dataset.episodes[0]
    for i, episode in enumerate(dataset.episodes):
        assert isinstance(episode, Episode)
        assert episode is dataset.episodes[i]

    # check append
    new_size = 2
    dataset.append(observations, actions, rewards, terminals)
    assert len(dataset) == new_size * n_episodes
    assert dataset.observations.shape == (
        new_size * data_size,
        observation_size,
    )
    assert dataset.rewards.shape == (new_size * data_size,)
    assert dataset.terminals.shape == (new_size * data_size,)
    if discrete_action:
        assert dataset.actions.shape == (new_size * data_size,)
    else:
        assert dataset.actions.shape == (new_size * data_size, action_size)

    # check append if discrete action and number of actions grow
    if discrete_action:
        old_action_size = dataset.get_action_size()
        new_size += 1
        dataset.append(observations, actions + add_actions, rewards, terminals)
        assert dataset.get_action_size() == old_action_size + add_actions

    # check extend
    new_size += 1
    another_dataset = MDPDataset(
        observations,
        actions,
        rewards,
        terminals,
        discrete_action=discrete_action,
    )
    dataset.extend(another_dataset)
    assert len(dataset) == new_size * n_episodes
    assert dataset.observations.shape == (
        new_size * data_size,
        observation_size,
    )
    assert dataset.rewards.shape == (new_size * data_size,)
    assert dataset.terminals.shape == (new_size * data_size,)
    if discrete_action:
        assert dataset.actions.shape == (new_size * data_size,)
    else:
        assert dataset.actions.shape == (new_size * data_size, action_size)

    # check dump and load
    dataset.dump(os.path.join("test_data", "dataset.h5"))
    new_dataset = MDPDataset.load(os.path.join("test_data", "dataset.h5"))
    assert np.all(dataset.observations == new_dataset.observations)
    assert np.all(dataset.actions == new_dataset.actions)
    assert np.all(dataset.rewards == new_dataset.rewards)
    assert np.all(dataset.terminals == new_dataset.terminals)
    assert dataset.discrete_action == new_dataset.discrete_action
    assert len(dataset) == len(new_dataset)


@pytest.mark.parametrize("data_size", [100])
@pytest.mark.parametrize("observation_size", [4])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("terminal", [True, False])
def test_episode(data_size, observation_size, action_size, terminal):
    observations = np.random.random((data_size, observation_size)).astype("f4")
    actions = np.random.random((data_size, action_size)).astype("f4")
    rewards = np.random.random(data_size).astype("f4")

    episode = Episode(
        observation_shape=(observation_size,),
        action_size=action_size,
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminal=terminal,
    )

    # check Episode methods
    assert np.all(episode.observations == observations)
    assert np.all(episode.actions == actions)
    assert np.all(episode.rewards == rewards)
    assert episode.size() == (data_size if terminal else data_size - 1)
    assert episode.get_observation_shape() == (observation_size,)
    assert episode.get_action_size() == action_size
    assert episode.compute_return() == np.sum(rewards)

    # check transitions exported from episode
    if terminal:
        assert len(episode.transitions) == data_size
    else:
        assert len(episode.transitions) == data_size - 1
    for i, t in enumerate(episode.transitions):
        assert isinstance(t, Transition)
        assert t.get_observation_shape() == (observation_size,)
        assert t.get_action_size() == action_size
        assert np.all(t.observation == observations[i])
        assert np.all(t.action == actions[i])
        assert np.allclose(t.reward, rewards[i])
        if terminal:
            if i == data_size - 1:
                assert t.terminal == 1.0
                assert np.all(t.next_observation == 0.0)
            else:
                assert t.terminal == 0.0
                assert np.all(t.next_observation == observations[i + 1])
        else:
            assert t.terminal == 0.0

    # check forward pointers
    count = 1
    transition = episode[0]
    while transition.next_transition:
        transition = transition.next_transition
        count += 1
    assert count == (data_size if terminal else data_size - 1)

    # check backward pointers
    count = 1
    transition = episode[-1]
    while transition.prev_transition:
        transition = transition.prev_transition
        count += 1
    assert count == (data_size if terminal else data_size - 1)

    # check list-like bahaviors
    assert len(episode) == (data_size if terminal else data_size - 1)
    assert episode[0] is episode.transitions[0]
    for i, transition in enumerate(episode):
        assert isinstance(transition, Transition)
        assert transition is episode.transitions[i]


@pytest.mark.parametrize("data_size", [100])
@pytest.mark.parametrize("observation_size", [100])
@pytest.mark.parametrize("action_size", [2])
def test_episode_terminals(data_size, observation_size, action_size):
    observations = np.random.random((data_size, observation_size)).astype("f4")
    actions = np.random.random((data_size, action_size)).astype("f4")
    rewards = np.random.random(data_size).astype("f4")

    # check default
    terminals = np.zeros(data_size, dtype=np.float32)
    terminals[49] = 1.0
    terminals[-1] = 1.0
    dataset1 = MDPDataset(observations, actions, rewards, terminals)
    assert len(dataset1.episodes) == 2
    assert np.all(dataset1.terminals == dataset1.episode_terminals)
    assert dataset1.episodes[0].terminal
    assert dataset1.episodes[0][-1].terminal

    # check non-terminal episode
    terminals = np.zeros(data_size, dtype=np.float32)
    terminals[-1] = 1.0
    episode_terminals = np.zeros(data_size, dtype=np.float32)
    episode_terminals[49] = 1.0
    episode_terminals[-1] = 1.0
    dataset2 = MDPDataset(
        observations, actions, rewards, terminals, episode_terminals
    )
    assert len(dataset2.episodes) == 2
    assert not np.all(dataset2.terminals == dataset2.episode_terminals)
    assert not dataset2.episodes[0].terminal
    assert not dataset2.episodes[0][-1].terminal

    # check extend
    dataset1.extend(dataset2)
    assert len(dataset1) == 4
    assert not dataset1.episodes[2].terminal
    assert dataset1.episodes[3].terminal


@pytest.mark.parametrize("data_size", [100])
@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("n_frames", [1, 4])
@pytest.mark.parametrize("n_steps", [1, 3])
@pytest.mark.parametrize("gamma", [0.99])
@pytest.mark.parametrize("discrete_action", [False, True])
def test_transition_minibatch(
    data_size,
    observation_shape,
    action_size,
    n_frames,
    n_steps,
    gamma,
    discrete_action,
):
    if len(observation_shape) == 3:
        observations = np.random.randint(
            256, size=(data_size, *observation_shape), dtype=np.uint8
        )
    else:
        observations = np.random.random(
            (data_size,) + observation_shape
        ).astype("f4")
    if discrete_action:
        actions = np.random.randint(action_size, size=data_size)
    else:
        actions = np.random.random((data_size, action_size)).astype("f4")
    rewards = np.random.random((data_size, 1)).astype("f4")

    episode = Episode(
        observation_shape=observation_shape,
        action_size=action_size,
        observations=observations,
        actions=actions,
        rewards=rewards,
    )

    if len(observation_shape) == 3:
        n_channels = n_frames * observation_shape[0]
        image_size = observation_shape[1:]
        batched_observation_shape = (data_size, n_channels, *image_size)
    else:
        batched_observation_shape = (data_size, *observation_shape)

    batch = TransitionMiniBatch(episode.transitions, n_frames, n_steps, gamma)
    assert batch.observations.shape == batched_observation_shape
    assert batch.next_observations.shape == batched_observation_shape

    for i, t in enumerate(episode.transitions):
        observation = batch.observations[i]
        next_observation = batch.next_observations[i]
        n = int(batch.n_steps[i][0])

        assert n == min(data_size - i, n_steps)

        if n_frames > 1 and len(observation_shape) == 3:
            # create padded observations for check stacking
            pad = ((n_frames - 1, 1), (0, 0), (0, 0), (0, 0))
            padded_observations = np.pad(observations, pad, "edge")

            # check frame stacking
            head_index = i
            tail_index = head_index + n_frames
            window = padded_observations[head_index:tail_index]
            next_window = padded_observations[head_index + n : tail_index + n]
            ref_observation = np.vstack(window)
            ref_next_observation = np.vstack(next_window)
            assert observation.shape == ref_observation.shape
            assert next_observation.shape == ref_next_observation.shape
            assert np.all(observation == ref_observation)
            if i >= data_size - n_steps:
                assert np.all(next_observation == 0)
            else:
                assert np.all(next_observation == ref_next_observation)
        else:
            next_t = t
            for _ in range(n - 1):
                next_t = next_t.next_transition
            assert np.allclose(observation, t.observation)
            assert np.allclose(next_observation, next_t.next_observation)

        n_step_reward = 0.0
        terminal = 0.0
        next_t = t
        for j in range(n):
            n_step_reward += rewards[i + j] * gamma ** j
            terminal = next_t.terminal
            next_t = next_t.next_transition

        assert np.all(batch.actions[i] == t.action)
        assert np.allclose(batch.rewards[i][0], n_step_reward)
        assert np.all(batch.terminals[i][0] == terminal)

    # check list-like behavior
    assert len(batch) == data_size
    assert batch[0] is episode.transitions[0]
    for i, transition in enumerate(batch):
        assert isinstance(transition, Transition)
        assert transition is episode.transitions[i]

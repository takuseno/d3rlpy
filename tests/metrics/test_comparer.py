import numpy as np
import pytest

from d3rlpy.dataset import BasicTransitionPicker, Episode, TransitionMiniBatch
from d3rlpy.metrics.comparer import (
    compare_continuous_action_diff,
    compare_discrete_action_match,
)

from .test_scorer import DummyAlgo


def _convert_episode_to_batch(episode):
    transition_picker = BasicTransitionPicker()
    transitions = [
        transition_picker(episode, index)
        for index in range(episode.transition_count)
    ]
    return TransitionMiniBatch.from_transitions(transitions)


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("n_episodes", [100])
@pytest.mark.parametrize("episode_length", [10])
def test_compare_continuous_action_diff(
    observation_shape, action_size, n_episodes, episode_length
):
    episodes = []
    for _ in range(n_episodes):
        observations = np.random.random((episode_length,) + observation_shape)
        actions = np.random.random((episode_length, action_size))
        rewards = np.random.random((episode_length, 1))
        episode = Episode(
            observations.astype("f4"),
            actions,
            rewards,
            False,
        )
        episodes.append(episode)

    A1 = np.random.random(observation_shape + (action_size,))
    A2 = np.random.random(observation_shape + (action_size,))
    algo = DummyAlgo(A1, 0.0)
    base_algo = DummyAlgo(A2, 0.0)

    total_diffs = []
    for episode in episodes:
        batch = _convert_episode_to_batch(episode)
        actions = algo.predict(batch.observations)
        base_actions = base_algo.predict(batch.observations)
        diff = ((actions - base_actions) ** 2).sum(axis=1).tolist()
        total_diffs += diff

    score = compare_continuous_action_diff(base_algo)(
        algo, episodes, BasicTransitionPicker()
    )
    assert np.allclose(score, -np.mean(total_diffs))


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("n_episodes", [100])
@pytest.mark.parametrize("episode_length", [10])
def test_compare_discrete_action_diff(
    observation_shape, action_size, n_episodes, episode_length
):
    episodes = []
    for _ in range(n_episodes):
        observations = np.random.random((episode_length,) + observation_shape)
        actions = np.random.random((episode_length, action_size))
        rewards = np.random.random((episode_length, 1))
        episode = Episode(
            observations.astype("f4"),
            actions,
            rewards,
            False,
        )
        episodes.append(episode)

    A1 = np.random.random(observation_shape + (action_size,))
    A2 = np.random.random(observation_shape + (action_size,))
    algo = DummyAlgo(A1, 0.0, discrete=True)
    base_algo = DummyAlgo(A2, 0.0, discrete=True)

    total_matches = []
    for episode in episodes:
        batch = _convert_episode_to_batch(episode)
        actions = algo.predict(batch.observations)
        base_actions = base_algo.predict(batch.observations)
        match = (actions == base_actions).tolist()
        total_matches += match

    score = compare_discrete_action_match(base_algo)(
        algo, episodes, BasicTransitionPicker()
    )
    assert np.allclose(score, np.mean(total_matches))

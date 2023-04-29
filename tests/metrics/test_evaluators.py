import numpy as np
import pytest

from d3rlpy.dataset import (
    BasicTransitionPicker,
    Episode,
    InfiniteBuffer,
    ReplayBuffer,
    TransitionMiniBatch,
)
from d3rlpy.metrics.evaluators import (
    AverageValueEstimationEvaluator,
    CompareContinuousActionDiffEvaluator,
    CompareDiscreteActionMatchEvaluator,
    ContinuousActionDiffEvaluator,
    DiscountedSumOfAdvantageEvaluator,
    DiscreteActionMatchEvaluator,
    InitialStateValueEstimationEvaluator,
    SoftOPCEvaluator,
    TDErrorEvaluator,
)
from d3rlpy.preprocessing import ClipRewardScaler


def _convert_episode_to_batch(episode):
    transition_picker = BasicTransitionPicker()
    transitions = [
        transition_picker(episode, index)
        for index in range(episode.transition_count)
    ]
    return TransitionMiniBatch.from_transitions(transitions)


def _create_replay_buffer(episodes):
    return ReplayBuffer(InfiniteBuffer(), episodes=episodes)


# dummy algorithm with deterministic outputs
class DummyAlgo:
    def __init__(self, A, gamma, discrete=False, reward_scaler=None):
        self.A = A
        self.gamma = gamma
        self.discrete = discrete
        self.n_frames = 1
        self.reward_scaler = reward_scaler

    def predict(self, x):
        x = np.array(x)
        y = np.matmul(x.reshape(x.shape[0], -1), self.A)
        if self.discrete:
            return y.argmax(axis=1)
        return y

    def predict_value(self, x, action, with_std=False):
        values = np.mean(x, axis=1) + np.mean(action, axis=1)
        if with_std:
            return values.reshape(-1), values.reshape(-1) + 0.1
        return values.reshape(-1)


def ref_td_error_score(
    predict_value,
    observations,
    actions,
    rewards,
    next_observations,
    next_actions,
    terminals,
    gamma,
    reward_scaler,
):
    if reward_scaler:
        rewards = reward_scaler.transform_numpy(rewards)
    values = predict_value(observations, actions)
    next_values = predict_value(next_observations, next_actions)
    y = rewards + gamma * next_values * (1.0 - terminals)
    return ((y - values) ** 2).reshape(-1).tolist()


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("n_episodes", [100])
@pytest.mark.parametrize("episode_length", [10])
@pytest.mark.parametrize("gamma", [0.99])
@pytest.mark.parametrize(
    "reward_scaler", [None, ClipRewardScaler(low=0.2, high=0.5)]
)
def test_td_error_scorer(
    observation_shape,
    action_size,
    n_episodes,
    episode_length,
    gamma,
    reward_scaler,
):
    # projection matrix for deterministic action
    A = np.random.random(observation_shape + (action_size,))
    episodes = []
    for _ in range(n_episodes):
        observations = np.random.random((episode_length,) + observation_shape)
        actions = np.matmul(observations, A).astype("f4")
        rewards = np.random.random((episode_length, 1)).astype("f4")
        episode = Episode(
            observations.astype("f4"),
            actions,
            rewards,
            False,
        )
        episodes.append(episode)

    algo = DummyAlgo(A, gamma, reward_scaler=reward_scaler)

    ref_errors = []
    for episode in episodes:
        batch = _convert_episode_to_batch(episode)
        ref_error = ref_td_error_score(
            algo.predict_value,
            batch.observations,
            batch.actions,
            np.asarray(batch.rewards).reshape(-1),
            batch.next_observations,
            algo.predict(batch.next_observations),
            np.asarray(batch.terminals).reshape(-1),
            gamma,
            reward_scaler,
        )
        ref_errors += ref_error

    score = TDErrorEvaluator()(algo, _create_replay_buffer(episodes))
    assert np.allclose(score, np.mean(ref_errors))


def ref_discounted_sum_of_advantage_score(
    predict_value, observations, dataset_actions, policy_actions, gamma
):
    dataset_values = predict_value(observations, dataset_actions)
    policy_values = predict_value(observations, policy_actions)
    advantages = (dataset_values - policy_values).reshape(-1).tolist()
    rets = []
    for i in range(len(advantages)):
        sum_advangage = 0.0
        for j, advantage in enumerate(advantages[i:]):
            sum_advangage += (gamma**j) * advantage
        rets.append(sum_advangage)
    return rets


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("n_episodes", [100])
@pytest.mark.parametrize("episode_length", [10])
@pytest.mark.parametrize("gamma", [0.99])
def test_discounted_sum_of_advantage_scorer(
    observation_shape, action_size, n_episodes, episode_length, gamma
):
    # projection matrix for deterministic action
    A = np.random.random(observation_shape + (action_size,))
    episodes = []
    for _ in range(n_episodes):
        observations = np.random.random((episode_length,) + observation_shape)
        # make difference between algorithm outputs and dataset
        noise = 100 * np.random.random((episode_length, action_size))
        actions = (np.matmul(observations, A) + noise).astype("f4")
        rewards = np.random.random((episode_length, 1)).astype("f4")
        episode = Episode(
            observations.astype("f4"),
            actions,
            rewards,
            False,
        )
        episodes.append(episode)

    algo = DummyAlgo(A, gamma)

    ref_sums = []
    for episode in episodes:
        batch = _convert_episode_to_batch(episode)
        policy_actions = algo.predict(batch.observations)
        ref_sum = ref_discounted_sum_of_advantage_score(
            algo.predict_value,
            batch.observations,
            batch.actions,
            policy_actions,
            gamma,
        )
        ref_sums += ref_sum

    score = DiscountedSumOfAdvantageEvaluator()(
        algo, _create_replay_buffer(episodes)
    )
    assert np.allclose(score, np.mean(ref_sums))


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("n_episodes", [100])
@pytest.mark.parametrize("episode_length", [10])
def test_average_value_estimation_scorer(
    observation_shape, action_size, n_episodes, episode_length
):
    # projection matrix for deterministic action
    A = np.random.random(observation_shape + (action_size,))
    episodes = []
    for _ in range(n_episodes):
        observations = np.random.random((episode_length,) + observation_shape)
        actions = np.matmul(observations, A).astype("f4")
        rewards = np.random.random((episode_length, 1)).astype("f4")
        episode = Episode(
            observations.astype("f4"),
            actions,
            rewards,
            False,
        )
        episodes.append(episode)

    algo = DummyAlgo(A, 0.0)

    total_values = []
    for episode in episodes:
        batch = _convert_episode_to_batch(episode)
        policy_actions = algo.predict(batch.observations)
        values = algo.predict_value(batch.observations, policy_actions)
        total_values += values.tolist()

    score = AverageValueEstimationEvaluator()(
        algo, _create_replay_buffer(episodes)
    )
    assert np.allclose(score, np.mean(total_values))


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("n_episodes", [100])
@pytest.mark.parametrize("episode_length", [10])
def test_initial_state_value_estimation_scorer(
    observation_shape, action_size, n_episodes, episode_length
):
    # projection matrix for deterministic action
    A = np.random.random(observation_shape + (action_size,))
    episodes = []
    for _ in range(n_episodes):
        observations = np.random.random((episode_length,) + observation_shape)
        actions = np.matmul(observations, A).astype("f4")
        rewards = np.random.random((episode_length, 1)).astype("f4")
        episode = Episode(
            observations.astype("f4"),
            actions,
            rewards,
            False,
        )
        episodes.append(episode)

    algo = DummyAlgo(A, 0.0)

    total_values = []
    for episode in episodes:
        observation = episode.observations[0].reshape(1, -1)
        policy_actions = algo.predict(observation)
        values = algo.predict_value(observation, policy_actions)
        total_values.append(values)

    score = InitialStateValueEstimationEvaluator()(
        algo, _create_replay_buffer(episodes)
    )
    assert np.allclose(score, np.mean(total_values))


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("n_episodes", [100])
@pytest.mark.parametrize("episode_length", [10])
@pytest.mark.parametrize("threshold", [5.0])
def test_soft_opc_scorer(
    observation_shape, action_size, n_episodes, episode_length, threshold
):
    # projection matrix for deterministic action
    A = np.random.random(observation_shape + (action_size,))
    episodes = []
    for _ in range(n_episodes):
        observations = np.random.random((episode_length,) + observation_shape)
        actions = np.matmul(observations, A).astype("f4")
        rewards = np.random.random((episode_length, 1)).astype("f4")
        episode = Episode(
            observations.astype("f4"),
            actions,
            rewards,
            False,
        )
        episodes.append(episode)

    algo = DummyAlgo(A, 0.0)
    success_values = []
    all_values = []
    for episode in episodes:
        is_success = episode.compute_return() >= threshold
        batch = _convert_episode_to_batch(episode)
        values = algo.predict_value(batch.observations, batch.actions)
        if is_success:
            success_values += values.tolist()
        all_values += values.tolist()

    scorer = SoftOPCEvaluator(threshold)
    score = scorer(algo, _create_replay_buffer(episodes))
    assert np.allclose(score, np.mean(success_values) - np.mean(all_values))


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("n_episodes", [100])
@pytest.mark.parametrize("episode_length", [10])
def test_continuous_action_diff_scorer(
    observation_shape, action_size, n_episodes, episode_length
):
    # projection matrix for deterministic action
    A = np.random.random(observation_shape + (action_size,))
    episodes = []
    for _ in range(n_episodes):
        observations = np.random.random((episode_length,) + observation_shape)
        actions = np.random.random((episode_length, action_size)).astype("f4")
        rewards = np.random.random((episode_length, 1)).astype("f4")
        episode = Episode(
            observations.astype("f4"),
            actions,
            rewards,
            False,
        )
        episodes.append(episode)

    algo = DummyAlgo(A, 0.0)

    total_diffs = []
    for episode in episodes:
        batch = _convert_episode_to_batch(episode)
        policy_actions = algo.predict(batch.observations)
        diff = ((batch.actions - policy_actions) ** 2).sum(axis=1).tolist()
        total_diffs += diff
    score = ContinuousActionDiffEvaluator()(
        algo, _create_replay_buffer(episodes)
    )
    assert np.allclose(score, np.mean(total_diffs))


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("n_episodes", [100])
@pytest.mark.parametrize("episode_length", [10])
def test_discrete_action_match_scorer(
    observation_shape, action_size, n_episodes, episode_length
):
    # projection matrix for deterministic action
    A = np.random.random(observation_shape + (action_size,))
    episodes = []
    for _ in range(n_episodes):
        observations = np.random.random((episode_length,) + observation_shape)
        actions = np.random.randint(action_size, size=(episode_length, 1))
        rewards = np.random.random((episode_length, 1)).astype("f4")
        episode = Episode(
            observations.astype("f4"),
            actions,
            rewards,
            False,
        )
        episodes.append(episode)

    algo = DummyAlgo(A, 0.0, discrete=True)

    total_matches = []
    for episode in episodes:
        batch = _convert_episode_to_batch(episode)
        policy_actions = algo.predict(batch.observations)
        match = (batch.actions.reshape(-1) == policy_actions).tolist()
        total_matches += match
    score = DiscreteActionMatchEvaluator()(
        algo, _create_replay_buffer(episodes)
    )
    assert np.allclose(score, np.mean(total_matches))


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

    score = CompareContinuousActionDiffEvaluator(base_algo)(
        algo, _create_replay_buffer(episodes)
    )
    assert np.allclose(score, np.mean(total_diffs))


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

    score = CompareDiscreteActionMatchEvaluator(base_algo)(
        algo, _create_replay_buffer(episodes)
    )
    assert np.allclose(score, np.mean(total_matches))

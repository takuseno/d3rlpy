from typing import Callable, Optional, Sequence

import numpy as np
import pytest

from d3rlpy.algos import DQNConfig, SACConfig
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
from d3rlpy.preprocessing import (
    ActionScaler,
    ClipRewardScaler,
    ObservationScaler,
    RewardScaler,
)
from d3rlpy.types import Float32NDArray, NDArray, Observation

from ..testing_utils import create_episode


def _convert_episode_to_batch(episode: Episode) -> TransitionMiniBatch:
    transition_picker = BasicTransitionPicker()
    transitions = [
        transition_picker(episode, index)
        for index in range(episode.transition_count)
    ]
    return TransitionMiniBatch.from_transitions(transitions)


def _create_replay_buffer(episodes: Sequence[Episode]) -> ReplayBuffer:
    return ReplayBuffer(InfiniteBuffer(), episodes=episodes)


# dummy algorithm with deterministic outputs
class DummyAlgo:
    def __init__(
        self,
        A: NDArray,
        gamma: float,
        discrete: bool = False,
        reward_scaler: Optional[RewardScaler] = None,
    ):
        self.A = A
        self.gamma = gamma
        self.discrete = discrete
        self.n_frames = 1
        self.reward_scaler = reward_scaler

    def predict(self, x: Observation) -> NDArray:
        x = np.array(x)
        y = np.matmul(x.reshape(x.shape[0], -1), self.A)
        if self.discrete:
            return y.argmax(axis=1)  # type: ignore
        return y  # type: ignore

    def predict_value(self, x: Observation, action: NDArray) -> NDArray:
        values = np.mean(x, axis=1) + np.mean(action, axis=1)
        return values.reshape(-1)  # type: ignore

    def sample_action(self, x: Observation) -> NDArray:
        raise NotImplementedError

    @property
    def observation_scaler(self) -> Optional[ObservationScaler]:
        return None

    @property
    def action_scaler(self) -> Optional[ActionScaler]:
        return None

    @property
    def action_size(self) -> int:
        return 1


def ref_td_error_score(
    predict_value: Callable[[Observation, NDArray], NDArray],
    observations: Observation,
    actions: NDArray,
    rewards: NDArray,
    next_observations: Observation,
    next_actions: NDArray,
    terminals: NDArray,
    gamma: float,
    reward_scaler: Optional[RewardScaler],
) -> list[float]:
    if reward_scaler:
        rewards = reward_scaler.transform_numpy(rewards)
    values = predict_value(observations, actions)
    next_values = predict_value(next_observations, next_actions)
    y = rewards + gamma * next_values * (1.0 - terminals)
    return ((y - values) ** 2).reshape(-1).tolist()  # type: ignore


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("n_episodes", [100])
@pytest.mark.parametrize("episode_length", [10])
@pytest.mark.parametrize("gamma", [0.99])
@pytest.mark.parametrize(
    "reward_scaler", [None, ClipRewardScaler(low=0.2, high=0.5)]
)
def test_td_error_scorer(
    observation_shape: Sequence[int],
    action_size: int,
    n_episodes: int,
    episode_length: int,
    gamma: float,
    reward_scaler: Optional[RewardScaler],
) -> None:
    # projection matrix for deterministic action
    A = np.random.random((*observation_shape, action_size))
    episodes = []
    for _ in range(n_episodes):
        observations = np.random.random((episode_length, *observation_shape))
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

    ref_errors: list[float] = []
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


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("episode_length", [10])
def test_td_error_scorer_with_algos(
    observation_shape: Sequence[int],
    action_size: int,
    episode_length: int,
) -> None:
    # test with DQN
    discrete_episode = create_episode(
        observation_shape,
        action_size,
        length=episode_length,
        discrete_action=True,
    )
    discrete_replay_buffer = _create_replay_buffer([discrete_episode])
    dqn = DQNConfig().create()
    dqn.build_with_dataset(discrete_replay_buffer)
    TDErrorEvaluator()(dqn, discrete_replay_buffer)

    # test with SAC
    episode = create_episode(
        observation_shape,
        action_size,
        length=episode_length,
        discrete_action=False,
    )
    replay_buffer = _create_replay_buffer([episode])
    sac = SACConfig().create()
    sac.build_with_dataset(replay_buffer)
    TDErrorEvaluator()(sac, replay_buffer)


def ref_discounted_sum_of_advantage_score(
    predict_value: Callable[[Observation, NDArray], NDArray],
    observations: Observation,
    dataset_actions: NDArray,
    policy_actions: NDArray,
    gamma: float,
) -> list[float]:
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
    observation_shape: Sequence[int],
    action_size: int,
    n_episodes: int,
    episode_length: int,
    gamma: float,
) -> None:
    # projection matrix for deterministic action
    A = np.random.random((*observation_shape, action_size))
    episodes = []
    for _ in range(n_episodes):
        observations = np.random.random((episode_length, *observation_shape))
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
@pytest.mark.parametrize("episode_length", [10])
def test_discounted_sum_of_advantage_with_algos(
    observation_shape: Sequence[int],
    action_size: int,
    episode_length: int,
) -> None:
    # test with DQN
    discrete_episode = create_episode(
        observation_shape,
        action_size,
        length=episode_length,
        discrete_action=True,
    )
    discrete_replay_buffer = _create_replay_buffer([discrete_episode])
    dqn = DQNConfig().create()
    dqn.build_with_dataset(discrete_replay_buffer)
    DiscountedSumOfAdvantageEvaluator()(dqn, discrete_replay_buffer)

    # test with SAC
    episode = create_episode(
        observation_shape,
        action_size,
        length=episode_length,
        discrete_action=False,
    )
    replay_buffer = _create_replay_buffer([episode])
    sac = SACConfig().create()
    sac.build_with_dataset(replay_buffer)
    DiscountedSumOfAdvantageEvaluator()(sac, replay_buffer)


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("n_episodes", [100])
@pytest.mark.parametrize("episode_length", [10])
def test_average_value_estimation_scorer(
    observation_shape: Sequence[int],
    action_size: int,
    n_episodes: int,
    episode_length: int,
) -> None:
    # projection matrix for deterministic action
    A = np.random.random((*observation_shape, action_size))
    episodes = []
    for _ in range(n_episodes):
        observations = np.random.random((episode_length, *observation_shape))
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
@pytest.mark.parametrize("episode_length", [10])
def test_average_value_estimation_with_algos(
    observation_shape: Sequence[int],
    action_size: int,
    episode_length: int,
) -> None:
    # test with DQN
    discrete_episode = create_episode(
        observation_shape,
        action_size,
        length=episode_length,
        discrete_action=True,
    )
    discrete_replay_buffer = _create_replay_buffer([discrete_episode])
    dqn = DQNConfig().create()
    dqn.build_with_dataset(discrete_replay_buffer)
    AverageValueEstimationEvaluator()(dqn, discrete_replay_buffer)

    # test with SAC
    episode = create_episode(
        observation_shape,
        action_size,
        length=episode_length,
        discrete_action=False,
    )
    replay_buffer = _create_replay_buffer([episode])
    sac = SACConfig().create()
    sac.build_with_dataset(replay_buffer)
    AverageValueEstimationEvaluator()(sac, replay_buffer)


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("n_episodes", [100])
@pytest.mark.parametrize("episode_length", [10])
def test_initial_state_value_estimation_scorer(
    observation_shape: Sequence[int],
    action_size: int,
    n_episodes: int,
    episode_length: int,
) -> None:
    # projection matrix for deterministic action
    A = np.random.random((*observation_shape, action_size))
    episodes = []
    for _ in range(n_episodes):
        observations = np.random.random((episode_length, *observation_shape))
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
@pytest.mark.parametrize("episode_length", [10])
def test_initial_state_value_estimation_with_algos(
    observation_shape: Sequence[int],
    action_size: int,
    episode_length: int,
) -> None:
    # test with DQN
    discrete_episode = create_episode(
        observation_shape,
        action_size,
        length=episode_length,
        discrete_action=True,
    )
    discrete_replay_buffer = _create_replay_buffer([discrete_episode])
    dqn = DQNConfig().create()
    dqn.build_with_dataset(discrete_replay_buffer)
    InitialStateValueEstimationEvaluator()(dqn, discrete_replay_buffer)

    # test with SAC
    episode = create_episode(
        observation_shape,
        action_size,
        length=episode_length,
        discrete_action=False,
    )
    replay_buffer = _create_replay_buffer([episode])
    sac = SACConfig().create()
    sac.build_with_dataset(replay_buffer)
    InitialStateValueEstimationEvaluator()(sac, replay_buffer)


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("n_episodes", [100])
@pytest.mark.parametrize("episode_length", [10])
@pytest.mark.parametrize("threshold", [5.0])
def test_soft_opc_scorer(
    observation_shape: Sequence[int],
    action_size: int,
    n_episodes: int,
    episode_length: int,
    threshold: float,
) -> None:
    # projection matrix for deterministic action
    A = np.random.random((*observation_shape, action_size))
    episodes = []
    for _ in range(n_episodes):
        observations = np.random.random((episode_length, *observation_shape))
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
@pytest.mark.parametrize("episode_length", [10])
@pytest.mark.parametrize("threshold", [5.0])
def test_soft_opc_wtth_algos(
    observation_shape: Sequence[int],
    action_size: int,
    episode_length: int,
    threshold: float,
) -> None:
    # test with DQN
    discrete_episode = create_episode(
        observation_shape,
        action_size,
        length=episode_length,
        discrete_action=True,
    )
    discrete_replay_buffer = _create_replay_buffer([discrete_episode])
    dqn = DQNConfig().create()
    dqn.build_with_dataset(discrete_replay_buffer)
    SoftOPCEvaluator(threshold)(dqn, discrete_replay_buffer)

    # test with SAC
    episode = create_episode(
        observation_shape,
        action_size,
        length=episode_length,
        discrete_action=False,
    )
    replay_buffer = _create_replay_buffer([episode])
    sac = SACConfig().create()
    sac.build_with_dataset(replay_buffer)
    SoftOPCEvaluator(threshold)(sac, replay_buffer)


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("n_episodes", [100])
@pytest.mark.parametrize("episode_length", [10])
def test_continuous_action_diff_scorer(
    observation_shape: Sequence[int],
    action_size: int,
    n_episodes: int,
    episode_length: int,
) -> None:
    # projection matrix for deterministic action
    A = np.random.random((*observation_shape, action_size))
    episodes = []
    for _ in range(n_episodes):
        observations = np.random.random((episode_length, *observation_shape))
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
@pytest.mark.parametrize("episode_length", [10])
def test_continuous_action_diff_with_algos(
    observation_shape: Sequence[int],
    action_size: int,
    episode_length: int,
) -> None:
    # test with SAC
    episode = create_episode(
        observation_shape,
        action_size,
        length=episode_length,
        discrete_action=False,
    )
    replay_buffer = _create_replay_buffer([episode])
    sac = SACConfig().create()
    sac.build_with_dataset(replay_buffer)
    ContinuousActionDiffEvaluator()(sac, replay_buffer)


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("n_episodes", [100])
@pytest.mark.parametrize("episode_length", [10])
def test_discrete_action_match_scorer(
    observation_shape: Sequence[int],
    action_size: int,
    n_episodes: int,
    episode_length: int,
) -> None:
    # projection matrix for deterministic action
    A = np.random.random((*observation_shape, action_size))
    episodes = []
    for _ in range(n_episodes):
        observations = np.random.random((episode_length, *observation_shape))
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
@pytest.mark.parametrize("episode_length", [10])
def test_discrete_action_match_with_algos(
    observation_shape: Sequence[int],
    action_size: int,
    episode_length: int,
) -> None:
    # test with DQN
    discrete_episode = create_episode(
        observation_shape,
        action_size,
        length=episode_length,
        discrete_action=True,
    )
    discrete_replay_buffer = _create_replay_buffer([discrete_episode])
    dqn = DQNConfig().create()
    dqn.build_with_dataset(discrete_replay_buffer)
    DiscreteActionMatchEvaluator()(dqn, discrete_replay_buffer)


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("n_episodes", [100])
@pytest.mark.parametrize("episode_length", [10])
def test_compare_continuous_action_diff(
    observation_shape: Sequence[int],
    action_size: int,
    n_episodes: int,
    episode_length: int,
) -> None:
    episodes = []
    for _ in range(n_episodes):
        observations = np.random.random((episode_length, *observation_shape))
        actions = np.random.random((episode_length, action_size))
        rewards: Float32NDArray = np.random.random((episode_length, 1)).astype(
            np.float32
        )
        episode = Episode(
            observations.astype("f4"),
            actions,
            rewards,
            False,
        )
        episodes.append(episode)

    A1 = np.random.random((*observation_shape, action_size))
    A2 = np.random.random((*observation_shape, action_size))
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
@pytest.mark.parametrize("episode_length", [10])
def test_compare_continuous_action_diff_with_algos(
    observation_shape: Sequence[int],
    action_size: int,
    episode_length: int,
) -> None:
    # test with SAC
    episode = create_episode(
        observation_shape,
        action_size,
        length=episode_length,
        discrete_action=False,
    )
    replay_buffer = _create_replay_buffer([episode])
    sac1 = SACConfig().create()
    sac1.build_with_dataset(replay_buffer)
    sac2 = SACConfig().create()
    sac2.build_with_dataset(replay_buffer)
    CompareContinuousActionDiffEvaluator(sac1)(sac2, replay_buffer)


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("n_episodes", [100])
@pytest.mark.parametrize("episode_length", [10])
def test_compare_discrete_action_diff(
    observation_shape: Sequence[int],
    action_size: int,
    n_episodes: int,
    episode_length: int,
) -> None:
    episodes = []
    for _ in range(n_episodes):
        observations = np.random.random((episode_length, *observation_shape))
        actions = np.random.random((episode_length, action_size))
        rewards: Float32NDArray = np.random.random((episode_length, 1)).astype(
            np.float32
        )
        episode = Episode(
            observations.astype("f4"),
            actions,
            rewards,
            False,
        )
        episodes.append(episode)

    A1 = np.random.random((*observation_shape, action_size))
    A2 = np.random.random((*observation_shape, action_size))
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


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("episode_length", [10])
def test_compare_discrete_action_diff_with_algos(
    observation_shape: Sequence[int],
    action_size: int,
    episode_length: int,
) -> None:
    # test with DQN
    discrete_episode = create_episode(
        observation_shape,
        action_size,
        length=episode_length,
        discrete_action=True,
    )
    discrete_replay_buffer = _create_replay_buffer([discrete_episode])
    dqn1 = DQNConfig().create()
    dqn1.build_with_dataset(discrete_replay_buffer)
    dqn2 = DQNConfig().create()
    dqn2.build_with_dataset(discrete_replay_buffer)
    CompareDiscreteActionMatchEvaluator(dqn1)(dqn2, discrete_replay_buffer)
